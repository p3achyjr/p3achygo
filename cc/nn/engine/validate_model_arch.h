#ifndef NN_ENGINE_VALIDATE_MODEL_ARCH_H_
#define NN_ENGINE_VALIDATE_MODEL_ARCH_H_

#include "cc/nn/engine/model_arch.h"

namespace nn {
namespace model_arch {

/*
 * Sanity checks model. Aborts on failure.
 */
void ValidateModelArch(Model* model);

}  // namespace model_arch
}  // namespace nn

#endif
