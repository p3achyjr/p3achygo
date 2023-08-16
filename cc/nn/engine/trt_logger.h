#ifndef NN_ENGINE_TRT_LOGGER_H_
#define NN_ENGINE_TRT_LOGGER_H_

#include <NvInfer.h>

#include <iostream>

#include "cc/nn/engine/model_arch.h"

namespace nn {
namespace trt {

/*
 * Basic Logger that pipes to std::cerr.
 */
class Logger : public nvinfer1::ILogger {
 public:
  Logger(Severity min_severity) : min_severity_(min_severity) {}

  void log(Severity severity,
           nvinfer1::AsciiChar const* msg) noexcept override {
    if (static_cast<int>(severity) > static_cast<int>(min_severity_)) {
      return;
    }

    std::cerr << "[" << SeverityToAsciiChar(severity) << "] " << msg
              << std::endl;
  }

  void log(Severity severity, std::string msg) noexcept {
    log(severity, msg.c_str());
  }

 private:
  Severity min_severity_;
  char SeverityToAsciiChar(Severity severity) {
    switch (severity) {
      case Severity::kINTERNAL_ERROR:
        return 'F';
      case Severity::kERROR:
        return 'E';
      case Severity::kWARNING:
        return 'W';
      case Severity::kINFO:
        return 'I';
      case Severity::kVERBOSE:
        return 'V';
    }

    return '?';
  }
};

Logger& logger();

}  // namespace trt
}  // namespace nn

#endif
