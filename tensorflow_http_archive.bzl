"""
Copied from https://github.com/tensorflow/serving/blob/master/tensorflow_serving/repo.bzl
"""

def _tensorflow_http_archive(ctx):
    version = ctx.attr.version
    sha256 = ctx.attr.sha256

    strip_prefix = "tensorflow-%s" % version
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/refs/tags/v%s.tar.gz" % version,
    ]
    ctx.download_and_extract(
        urls,
        "",
        sha256,
        "",
        strip_prefix,
    )

tensorflow_http_archive = repository_rule(
    implementation = _tensorflow_http_archive,
    attrs = {
        "version": attr.string(mandatory = True),
        "sha256": attr.string(mandatory = True),
    },
)
