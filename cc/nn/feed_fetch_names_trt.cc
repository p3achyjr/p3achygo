#include "cc/nn/feed_fetch_names.h"

namespace nn {
const std::vector<std::string> kInputNames = {"serving_default_args_0:0",
                                              "serving_default_args_1:0"};

const std::vector<std::string> kOutputNames = {
    "PartitionedCall:0", "PartitionedCall:1", "PartitionedCall:2",
    "PartitionedCall:3", "PartitionedCall:4", "PartitionedCall:5",
    "PartitionedCall:6",
};
}  // namespace nn
