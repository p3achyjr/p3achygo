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
ABSL_FLAG(int, games_this_gen, -1, "Num new games to wait for.");
ABSL_FLAG(int, train_window_size, -1, "Size of training window to draw from.");
ABSL_FLAG(float, p, .01f,
          "Probability that any _individual_ sample is selected to include for "
          "training.");
ABSL_FLAG(bool, run_continuously, true,
          "Whether to run the shuffler in continuous mode.");
ABSL_FLAG(bool, is_local, false,
          "Whether self-play loop is running locally. In this case, the "
          "shuffler will ensure that data chunks are fully written out to disk "
          "before reading. It does this by checking for the presence of a "
          "`.DONE` file, written by the self-play process.");

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

  int games_this_gen = absl::GetFlag(FLAGS_games_this_gen);
  if (games_this_gen == -1) {
    LOG(ERROR) << "No --games_this_gen specified.";
    return 1;
  }

  int train_window_size = absl::GetFlag(FLAGS_train_window_size);
  if (train_window_size == -1) {
    LOG(ERROR) << "No --train_window_size specified.";
    return 1;
  }

  bool is_local = absl::GetFlag(FLAGS_is_local);

  float p = absl::GetFlag(FLAGS_p);
  LOG(INFO) << "Using Training Window of n=" << train_window_size
            << " samples and p=" << p
            << " sample draw probability. is_local=" << is_local;

  shuffler::ChunkManager chunk_manager(
      data_path, gen, p, games_this_gen, train_window_size,
      absl::GetFlag(FLAGS_run_continuously), is_local);
  std::thread wait_thread(WaitForSignal, &chunk_manager);

  // CreateChunk blocks until we receive a signal, or play `games_per_gen`
  // number of new games.
  chunk_manager.CreateChunk();
  chunk_manager.ShuffleAndFlush();

  // WaitForSignal may not have exited
  std::exit(0);
  return 0;
}
