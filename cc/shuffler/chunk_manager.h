#ifndef __SHUFFLER_CHUNK_MANAGER_H_
#define __SHUFFLER_CHUNK_MANAGER_H_

#include <deque>
#include <string>
#include <thread>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "cc/core/probability.h"
#include "cc/shuffler/filename_buffer.h"
#include "cc/shuffler/tf_record_watcher.h"
#include "tensorflow/core/platform/tstring.h"

namespace shuffler {

/*
 * Builds training chunks from a list of tfrecords, each containing a subset of
 * pre-shuffled examples.
 *
 * The chunk manager maintains a list of filenames, from which we interleave
 * reading examples from. Internally, we maintain a parameter `p`, indicating
 * the probability of selecting any element. We also maintain a parameter `B`,
 * the size of the shuffle buffer from which we randomly draw elements from when
 * full.
 *
 * `dir`: Top-level data directory.
 * `gen`: Generation this chunk is for (starts from 1).
 * `p`: Probability of choosing any single example.
 * `buf_size`: Shuffle buffer size.
 * `max_chunk_size`: Max size of chunk.
 * `cycle_len`: Number of files to interleave reading from.
 * `poll_len`: Number of examples to scan before checking for new files.
 * `exclude_gens`: Generations to exclude.
 */
class ChunkManager final {
 public:
  ChunkManager(std::string dir, int gen, float p);
  ChunkManager(std::string dir, int gen, float p,
               std::vector<int> exclude_gens);
  ChunkManager(std::string dir, int gen, float p, std::vector<int> exclude_gens,
               size_t chunk_size, int poll_interval_s);
  ~ChunkManager();

  // Disable Copy
  ChunkManager(ChunkManager const&) = delete;
  ChunkManager& operator=(ChunkManager const&) = delete;

  void CreateChunk();
  void ShuffleAndFlush();
  void SignalStop();

 private:
  void AppendToChunk(::tensorflow::tstring&& proto);
  void FsThread();  // runs in `fs_thread_`.

  std::string dir_;
  int gen_;
  float p_;
  size_t chunk_size_;
  int poll_interval_s_;
  std::vector<int> exclude_gens_;

  core::Probability probability_;
  TfRecordWatcher watcher_ ABSL_GUARDED_BY(mu_);
  FilenameBuffer fbuffer_ ABSL_GUARDED_BY(mu_);
  absl::Mutex mu_;
  absl::CondVar cv_;

  std::deque<::tensorflow::tstring> chunk_;

  // thread to scan and update file list.
  bool running_ ABSL_GUARDED_BY(mu_);
  std::thread fs_thread_;
};
}  // namespace shuffler

#endif