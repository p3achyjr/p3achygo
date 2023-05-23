/*
 * Main function for shuffler binary.
 */

#include <iostream>
#include <thread>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "cc/shuffler/chunk_manager.h"

ABSL_FLAG(std::string, data_path, "", "Path to self-play data.");

void WaitForSignal(shuffler::ChunkManager* chunk_manager) {
  // any line from stdin is a shutdown signal.
  std::string signal;
  std::getline(std::cin, signal);

  chunk_manager->Stop();
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();

  std::string data_path = absl::GetFlag(FLAGS_data_path);
  if (data_path == "") {
    LOG(ERROR) << "No Data Path Specified.";
    return 1;
  }

  shuffler::ChunkManager chunk_manager(data_path, 0, .01f);
  std::thread wait_thread(WaitForSignal, &chunk_manager);

  auto chunk = chunk_manager.CreateChunk();
  LOG(INFO) << "Chunk contains: " << chunk.size() << " elements.";

  if (wait_thread.joinable()) {
    wait_thread.join();
  }

  return 0;
}
