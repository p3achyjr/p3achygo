#ifndef NN_ENGINE_CUDA_BUFFER_MANAGER_H_
#define NN_ENGINE_CUDA_BUFFER_MANAGER_H_

#include <NvInfer.h>

#include <string>
#include <vector>

#include "absl/container/node_hash_map.h"
#include "absl/log/check.h"
#include "cc/core/util.h"

namespace nn {

/*
 * Managed pool of CUDA buffers.
 */
class CudaBufferPool final {
 public:
  CudaBufferPool();
  ~CudaBufferPool();

  // Disable Copy and Move.
  CudaBufferPool(CudaBufferPool const&) = delete;
  CudaBufferPool& operator=(CudaBufferPool const&) = delete;
  CudaBufferPool(CudaBufferPool&&) = delete;
  CudaBufferPool& operator=(CudaBufferPool&&) = delete;

  // Adds new buffer keyed by `name`. Does not allocate resources.
  void AddBuffer(std::string name, nvinfer1::Dims dims);

  // Allocates host and device memory for buffer at `name`.
  void AllocateBuffer(std::string name);

  // Sets Buffer at `batch_id` to zero.
  void SetBatchZero(std::string name, int batch_id);

  // Methods to copy from host to device.
  void CopyToDevice(std::string name);
  void CopyToDeviceAsync(std::string name, cudaStream_t& stream);

  // Methods to copy from device to host.
  void CopyFromDevice(std::string name);
  void CopyFromDeviceAsync(std::string name, cudaStream_t& stream);

  // Frees all buffers.
  void Cleanup();

  // Fill input plane at binding `name`. Assumes NCHW format, and that H == W.
  template <typename T>
  void FillPlane(std::string name, int batch_id, int channel,
                 const T* grid_data, size_t grid_len) {
    CHECK(buf_map_.contains(name));
    BufferHandle& buf = buf_map_[name];
    float* host_buf = static_cast<float*>(buf.host_buf);

    for (int i = 0; i < grid_len; ++i) {
      for (int j = 0; j < grid_len; ++j) {
        int index = DimsToIndex(nvinfer1::Dims4(batch_id, channel, i, j),
                                buf.dim_sizes);
        host_buf[index] = static_cast<float>(grid_data[i * grid_len + j]);
      }
    }
  }

  // Fill single scalar value at binding `name`. `index_dims` must be the same
  // rank as the tensor we are modifying.
  template <typename T>
  void FillScalar(std::string name, nvinfer1::Dims index_dims, T val) {
    CHECK(buf_map_.contains(name));
    CHECK(buf_map_[name].dim_sizes.nbDims == index_dims.nbDims);
    BufferHandle& buf = buf_map_[name];
    float* host_buf = static_cast<float*>(buf.host_buf);
    host_buf[DimsToIndex(index_dims, buf.dim_sizes)] = val;
  }

 private:
  struct BufferHandle {
    size_t size;
    nvinfer1::Dims dims;
    nvinfer1::Dims dim_sizes;
    void* host_buf = 0;
    void* device_buf = 0;
  };

  size_t DimsToIndex(nvinfer1::Dims index_dims, nvinfer1::Dims size_dims) {
    size_t index = 0;
    for (int i = 0; i < index_dims.nbDims; ++i) {
      index += index_dims.d[i] * size_dims.d[i];
    }

    return index;
  }

  absl::node_hash_map<std::string, BufferHandle> buf_map_;
};

}  // namespace nn

#endif
