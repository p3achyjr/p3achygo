#include "cc/nn/engine/cuda_buffer_pool.h"

namespace nn {
namespace {
namespace nv = ::nvinfer1;
}

CudaBufferPool::CudaBufferPool() = default;
CudaBufferPool::~CudaBufferPool() = default;

// Adds new buffer keyed by `name`. Does not allocate resources.
void CudaBufferPool::AddBuffer(std::string name, nv::Dims dims) {
  CHECK(dims.nbDims > 0);
  BufferHandle buf;

  size_t num_elems = 1;
  nv::Dims dim_sizes;
  dim_sizes.nbDims = dims.nbDims;
  for (int i = dims.nbDims - 1; i >= 0; --i) {
    dim_sizes.d[i] = num_elems;
    num_elems *= dims.d[i];
  }

  size_t num_bytes = num_elems * sizeof(float);
  buf.size = num_bytes;
  buf.dims = dims;
  buf.dim_sizes = dim_sizes;

  buf_map_[name] = buf;
}

// Allocates host and device memory for buffer at `name`.
void CudaBufferPool::AllocateBuffer(std::string name) {
  CHECK(buf_map_.contains(name));
  size_t num_bytes = buf_map_[name].size;
  cudaMallocHost(&buf_map_[name].host_buf, num_bytes);
  cudaMalloc(&buf_map_[name].device_buf, num_bytes);
}

// Sets Buffer at `batch_id` to zero.
void CudaBufferPool::SetBatchZero(std::string name, int batch_id) {
  CHECK(buf_map_.contains(name));
  BufferHandle& buf = buf_map_[name];
  size_t num_bytes = buf.dim_sizes.d[0] * sizeof(float);
  memset(buf.host_buf, 0, num_bytes);
}

// Methods to copy from host to device.
void CudaBufferPool::CopyToDevice(std::string name) {
  CHECK(buf_map_.contains(name));
  BufferHandle& buf = buf_map_[name];
  cudaMemcpy(buf.device_buf, buf.host_buf, buf.size, cudaMemcpyHostToDevice);
}

void CudaBufferPool::CopyToDeviceAsync(std::string name, cudaStream_t& stream) {
  CHECK(buf_map_.contains(name));
  BufferHandle& buf = buf_map_[name];
  cudaMemcpyAsync(buf.device_buf, buf.host_buf, buf.size,
                  cudaMemcpyHostToDevice, stream);
}

// Methods to copy from device to host.
void CudaBufferPool::CopyFromDevice(std::string name) {
  CHECK(buf_map_.contains(name));
  BufferHandle& buf = buf_map_[name];
  cudaMemcpy(buf.host_buf, buf.device_buf, buf.size, cudaMemcpyDeviceToHost);
}

void CudaBufferPool::CopyFromDeviceAsync(std::string name,
                                         cudaStream_t& stream) {
  CHECK(buf_map_.contains(name));
  BufferHandle& buf = buf_map_[name];
  cudaMemcpyAsync(buf.host_buf, buf.device_buf, buf.size,
                  cudaMemcpyDeviceToHost, stream);
}

void CudaBufferPool::Cleanup() {
  for (auto& [name, buf] : buf_map_) {
    cudaFreeHost(buf.host_buf);
    cudaFree(buf.device_buf);
  }
}

}  // namespace nn
