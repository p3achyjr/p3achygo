#include "cc/nn/engine/trt_logger.h"

namespace nn {
namespace trt {

Logger& logger() {
  static Logger logger(Logger::Severity::kINFO);
  return logger;
}

}  // namespace trt
}  // namespace nn
