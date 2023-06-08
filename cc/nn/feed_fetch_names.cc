#include "cc/nn/feed_fetch_names.h"

namespace nn {
const std::vector<std::string> kInputNames = {"serving_default_args_0:0",
                                              "serving_default_args_1:0"};

const std::vector<std::string> kOutputNames = {
    "StatefulPartitionedCall:0", "StatefulPartitionedCall:1",
    "StatefulPartitionedCall:2", "StatefulPartitionedCall:3",
    "StatefulPartitionedCall:4",
};
}  // namespace nn
