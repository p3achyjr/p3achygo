#ifndef DATA_WORKER_H_
#define DATA_WORKER_H_

#include "cc/data/coordinator.h"

namespace data {

void Worker(int worker_id, Coordinator* coordinator, const std::string out_dir,
            const bool is_dry_run);

}  // namespace data

#endif