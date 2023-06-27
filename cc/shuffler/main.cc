/*
 * Main function for shuffler binary.
 */

#include <algorithm>
#include <iostream>
#include <thread>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "cc/shuffler/chunk_manager.h"

ABSL_FLAG(std::string, data_path, "", "Path to self-play data.");
ABSL_FLAG(int, gen, -1, "Generation to make chunk for.");
ABSL_FLAG(int, games_per_gen, -1, "Num new games to wait for.");
ABSL_FLAG(std::vector<std::string>, exclude_gens, {},
          "Comma-separated list of data generations to exclude.");
ABSL_FLAG(float, p, .01f,
          "Probability that any _individual_ sample is selected to include for "
          "training.");

void WaitForSignal(shuffler::ChunkManager* chunk_manager) {
  // any line from stdin is a shutdown signal.
  std::string signal;
  std::getline(std::cin, signal);

  chunk_manager->SignalStop();
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);

  std::string data_path = absl::GetFlag(FLAGS_data_path);
  if (data_path == "") {
    LOG(ERROR) << "No --data_path Specified.";
    return 1;
  }

  int gen = absl::GetFlag(FLAGS_gen);
  if (gen == -1) {
    LOG(ERROR) << "No --gen specified.";
    return 1;
  }

  int games_per_gen = absl::GetFlag(FLAGS_games_per_gen);
  if (games_per_gen == -1) {
    LOG(ERROR) << "No --games_per_gen specified.";
    return 1;
  }

  std::vector<std::string> exclude_gens_str = absl::GetFlag(FLAGS_exclude_gens);
  std::vector<int> exclude_gens;
  std::transform(exclude_gens_str.begin(), exclude_gens_str.end(),
                 std::back_inserter(exclude_gens),
                 [](const std::string& s) { return std::atoi(s.c_str()); });

  shuffler::ChunkManager chunk_manager(data_path, gen, absl::GetFlag(FLAGS_p),
                                       exclude_gens);
  std::thread wait_thread(WaitForSignal, &chunk_manager);

  // CreateChunk blocks until we receive a signal, or play `games_per_gen`
  // number of new games.
  chunk_manager.CreateChunk();
  chunk_manager.ShuffleAndFlush();

  if (wait_thread.joinable()) {
    wait_thread.join();
  }

  return 0;
}
