#include "cc/shuffler/chunk_manager.h"

#include <filesystem>  // compile with gcc 9+
#include <random>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "cc/data/filename_format.h"
#include "cc/data/tfrecord/record_reader.h"
#include "cc/data/tfrecord/record_writer.h"
#include "cc/shuffler/chunk_info.h"
#include "cc/shuffler/constants.h"

namespace shuffler {
namespace {
namespace fs = std::filesystem;
using ::data::kGoldenChunkFormat;
using ::data::kGoldenChunkSizeFormat;
using ::data::RecordReaderOptions;
using ::data::RecordWriter;
using ::data::RecordWriterOptions;
using ::data::SequentialRecordReader;

static constexpr size_t kDefaultChunkSize = 2048000;
static constexpr int kDefaultPollIntervalS = 10;

void WriteChunkToDisk(std::string filename,
                      const std::vector<std::string>& chunk) {
  RecordWriterOptions options = RecordWriterOptions::Zlib();
  options.zlib_options.compression_level = 2;
  RecordWriter writer(filename, options);
  CHECK(writer.Init().ok()) << "Failed to initialize RecordWriter";

  for (const std::string& record : chunk) {
    CHECK(writer.WriteRecord(record).ok());
  }

  CHECK(writer.Close().ok());
}
}  // namespace

ChunkManager::ChunkManager(std::string dir, int gen, float p, int games_per_gen,
                           int train_window_size, bool is_continuous,
                           bool is_local, std::optional<int> max_gen)
    : dir_(dir),
      gen_(gen),
      p_(p),
      chunk_size_(kDefaultChunkSize),
      poll_interval_s_(kDefaultPollIntervalS),
      games_per_gen_(games_per_gen),
      is_continuous_(is_continuous),
      watcher_(dir_, train_window_size, is_local, max_gen),
      fbuffer_(watcher_.GetFiles()),
      running_(true) {
  fs_thread_ = std::thread(&ChunkManager::FsThread, this);
}

ChunkManager::~ChunkManager() {
  mu_.Lock();
  running_.store(false, std::memory_order_release);
  cv_.SignalAll();
  mu_.Unlock();
  if (fs_thread_.joinable()) {
    fs_thread_.join();
  }
}

void ChunkManager::CreateChunk() {
  if (is_continuous_) {
    LOG(INFO) << "Creating Chunk " << gen_
              << " (Continuous Mode)... Dir: " << dir_;
  } else {
    LOG(INFO) << "Creating Chunk " << gen_
              << " (Finite Task Mode)... Dir: " << dir_;
  }

  int num_scanned = 0;
  auto start = std::chrono::steady_clock::now();
  while (running_.load(std::memory_order_acquire)) {
    // Pop file to read, if one exists.
    std::optional<std::string> f = PopFile();
    if (f == std::nullopt) {
      if (is_continuous_) {
        absl::MutexLock l(&mu_);
        cv_.WaitWithTimeout(&mu_, absl::Seconds(poll_interval_s_));
        continue;
      } else {
        break;
      }
    }

    // Read file into memory.
    SequentialRecordReader reader(*f, RecordReaderOptions::Zlib());
    if (!reader.Init().ok()) {
      LOG(WARNING) << "Failed to initialize reader for file: " << *f;
      continue;
    }

    while (true) {
      std::string record;
      if (!reader.ReadRecord(&record).ok()) {
        break;
      }

      if (probability_.Uniform() < p_) {
        AppendToChunk(std::move(record));
      }

      ++num_scanned;
    }
  }
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<float> elapsed = end - start;
  LOG(INFO) << "Building chunk took " << elapsed.count() << "s. Scanned "
            << num_scanned << " protos.";
}

void ChunkManager::ShuffleAndFlush() {
  // shuffle chunk
  std::vector<std::string> golden_chunk(chunk_.begin(), chunk_.end());
  auto start = std::chrono::steady_clock::now();
  std::shuffle(golden_chunk.begin(), golden_chunk.end(), probability_.prng());
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<float> elapsed = end - start;
  LOG(INFO) << "Shuffling chunk took " << elapsed.count()
            << "s. Chunk contains " << golden_chunk.size() << " elements.";

  // create directory
  std::string chunk_dir = fs::path(dir_).parent_path() / kGoldenChunkDirname;
  std::string chunk_filename =
      fs::path(chunk_dir) / absl::StrFormat(kGoldenChunkFormat, gen_);
  std::string chunk_size_filename =
      fs::path(chunk_dir) / absl::StrFormat(kGoldenChunkSizeFormat, gen_);
  fs::create_directory(chunk_dir);
  LOG(INFO) << "Writing chunk " << chunk_filename;

  // write to disk.
  start = std::chrono::steady_clock::now();
  WriteChunkToDisk(chunk_filename, golden_chunk);
  end = std::chrono::steady_clock::now();
  elapsed = end - start;
  LOG(INFO) << "Writing chunk took " << elapsed.count() << "s.";

  // write number of examples in batch.
  FILE* const file = fopen(chunk_size_filename.c_str(), "w");
  absl::FPrintF(file, "%d", golden_chunk.size());
  fclose(file);
}

void ChunkManager::SignalStop() {
  if (is_continuous_) {
    return;
  }

  absl::MutexLock l(&mu_);
  running_.store(false, std::memory_order_acquire);
  cv_.SignalAll();
}

std::optional<std::string> ChunkManager::PopFile() {
  absl::MutexLock l(&mu_);
  return fbuffer_.PopFile();
}

void ChunkManager::AppendToChunk(std::string&& proto) {
  chunk_.emplace_back(proto);
  if (chunk_.size() > chunk_size_) {
    chunk_.pop_front();
  }
}

void ChunkManager::FsThread() {
  if (!is_continuous_) {
    return;
  }

  while (true) {
    absl::MutexLock l(&mu_);
    cv_.WaitWithTimeout(&mu_, absl::Seconds(poll_interval_s_));
    if (!running_.load(std::memory_order_acquire)) {
      break;
    }

    std::vector<std::string> new_files = watcher_.UpdateAndGetNew();
    LOG_IF(INFO, new_files.size() > 0)
        << "Found " << new_files.size() << " new files. "
        << watcher_.NumGamesSinceInit() << " new games played since init.";

    // If we have received enough new files, flush.
    if (watcher_.NumGamesSinceInit() >= games_per_gen_) {
      running_.store(false, std::memory_order_release);
      break;
    }

    fbuffer_.AddNewFiles(new_files);
  }
}
}  // namespace shuffler
