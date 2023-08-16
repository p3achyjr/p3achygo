#ifndef NN_ENGINE_TRT_CALIBRATOR_H_
#define NN_ENGINE_TRT_CALIBRATOR_H_

#include <NvInfer.h>

#include <memory>
#include <string>

namespace nn {
namespace trt {

class Int8Calibrator : public nvinfer1::IInt8EntropyCalibrator2 {
 public:
  virtual ~Int8Calibrator() = default;
  virtual int getBatchSize() const noexcept override = 0;
  virtual bool getBatch(void* bindings[], const char* names[],
                        int nbBindings) noexcept override = 0;
  virtual const void* readCalibrationCache(
      size_t& length) noexcept override = 0;
  virtual void writeCalibrationCache(const void* cache,
                                     size_t length) noexcept override = 0;

  static std::unique_ptr<Int8Calibrator> Create(size_t batch_size,
                                                std::string calib_tfrec_path,
                                                std::string calib_cache_path);
};

}  // namespace trt
}  // namespace nn

#endif
