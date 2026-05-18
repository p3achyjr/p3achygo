"""Zlib-compressed TFRecord support for tfrecord[torch].

The upstream `tfrecord` package only handles `compression_type` of `'gzip'`
or `None`. Our chunks are written by the C++ self-play pipeline using
RFC 1950 (zlib) framing — `tf.io.TFRecordOptions(compression_type='ZLIB')`.

Importing this module monkey-patches `tfrecord.reader.tfrecord_iterator`
so callers can pass `compression_type='zlib'` to `TFRecordDataset` and
friends.

Limitation: forward-only `seek`. Index-based sharding (`index_path != None`)
is unsupported; we use `index_path=None`.
"""

import functools
import io
import struct
import zlib

from tfrecord import reader as _reader


class ZlibFileReader(io.RawIOBase):
    """Streaming RFC 1950 (zlib) decoder with file-object semantics."""

    def __init__(self, path, chunk_size=1 << 16):
        self._fobj = open(path, "rb")
        self._dec = zlib.decompressobj(zlib.MAX_WBITS)
        self._chunk = chunk_size
        self._buf = bytearray()
        self._pos = 0
        self._eof = False

    def _refill(self, need):
        while not self._eof and len(self._buf) < need:
            raw = self._fobj.read(self._chunk)
            if not raw:
                self._buf += self._dec.flush()
                self._eof = True
                break
            self._buf += self._dec.decompress(raw)

    def readable(self):
        return True

    def readinto(self, b):
        n = len(b)
        self._refill(n)
        m = min(n, len(self._buf))
        b[:m] = self._buf[:m]
        del self._buf[:m]
        self._pos += m
        return m

    def tell(self):
        return self._pos

    def close(self):
        try:
            self._fobj.close()
        finally:
            super().close()


_original_tfrecord_iterator = _reader.tfrecord_iterator


@functools.wraps(_original_tfrecord_iterator)
def _zlib_aware_tfrecord_iterator(
    data_path, index_path=None, shard=None, compression_type=None
):
    if compression_type != "zlib":
        yield from _original_tfrecord_iterator(
            data_path,
            index_path=index_path,
            shard=shard,
            compression_type=compression_type,
        )
        return

    if index_path is not None:
        raise NotImplementedError(
            "tfrecord with compression_type='zlib' does not support index_path"
        )

    file = ZlibFileReader(data_path)
    length_bytes = bytearray(8)
    crc_bytes = bytearray(4)
    datum_bytes = bytearray(1024 * 1024)
    try:
        while True:
            n = file.readinto(length_bytes)
            if n == 0:
                break
            if n != 8:
                raise RuntimeError(f"truncated length header (got {n} bytes)")
            if file.readinto(crc_bytes) != 4:
                raise RuntimeError("truncated start CRC")
            (length,) = struct.unpack("<Q", length_bytes)
            if length > len(datum_bytes):
                datum_bytes = bytearray(int(length * 1.5))
            view = memoryview(datum_bytes)[:length]
            if file.readinto(view) != length:
                raise RuntimeError("truncated record body")
            if file.readinto(crc_bytes) != 4:
                raise RuntimeError("truncated end CRC")
            yield view
    finally:
        file.close()


if not getattr(_reader, "_p3achygo_zlib_patched", False):
    _reader.tfrecord_iterator = _zlib_aware_tfrecord_iterator
    _reader._p3achygo_zlib_patched = True
