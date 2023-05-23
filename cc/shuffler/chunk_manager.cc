#include "cc/shuffler/chunk_manager.h"

#include <random>
#include <string>
#include <vector>

#include "absl/time/time.h"
#include "tensorflow/core/lib/io/compression.h"
#include "tensorflow/core/lib/io/record_reader.h"

namespace shuffler {
namespace {
static constexpr size_t kDefaultChunkSize = 2048000;
static constexpr int kDefaultPollIntervalS = 300;
static constexpr int kLoggingInterval = 1000000;
static constexpr int kEmptySleepIntervalS = 30;
}  // namespace

using ::tensorflow::tstring;
using ::tensorflow::io::RecordReaderOptions;
using ::tensorflow::io::SequentialRecordReader;
using ::tensorflow::io::compression::kZlib;

ChunkManager::ChunkManager(std::string dir, int gen, float p)
    : ChunkManager(dir, gen, p, {} /* exclude_gens */) {}

ChunkManager::ChunkManager(std::string dir, int gen, float p,
                           std::vector<int> exclude_gens)
    : ChunkManager(dir, gen, p, exclude_gens, kDefaultChunkSize,
                   kDefaultPollIntervalS) {}

ChunkManager::ChunkManager(std::string dir, int gen, float p,
                           std::vector<int> exclude_gens, size_t chunk_size,
                           int poll_interval_s)
    : dir_(dir),
      gen_(gen),
      p_(p),
      chunk_size_(chunk_size),
      poll_interval_s_(poll_interval_s),
      exclude_gens_(exclude_gens),
      probability_(static_cast<uint64_t>(std::time(nullptr))),
      watcher_(dir_, exclude_gens_),
      fbuffer_(watcher_.GetFiles()),
      running_(true) {
  fs_thread_ = std::move(std::thread(&ChunkManager::FsThread, this));
}

ChunkManager::~ChunkManager() {
  mu_.Lock();
  running_ = false;
  cv_.SignalAll();
  mu_.Unlock();
  if (fs_thread_.joinable()) {
    fs_thread_.join();
  }
}

std::vector<::tensorflow::tstring> ChunkManager::CreateChunk() {
  LOG(INFO) << "Creating Chunk...";

  int num_scanned = 0;
  auto start = std::chrono::steady_clock::now();
  while (true) {
    std::optional<std::string> f;
    {
      absl::MutexLock l(&mu_);
      if (!running_) {
        break;
      }

      LOG_EVERY_N_SEC(INFO, 30) << fbuffer_.size() << " files in buffer.";
      f = fbuffer_.PopFile();
    }

    if (f == std::nullopt) {
      LOG(INFO) << "No files remaining. Sleeping for " << kEmptySleepIntervalS
                << "s.";

      absl::MutexLock l(&mu_);
      cv_.WaitWithTimeout(&mu_, absl::Seconds(kEmptySleepIntervalS));
      if (!running_) {
        break;
      }
      continue;
    }

    std::unique_ptr<tensorflow::RandomAccessFile> file;
    TF_CHECK_OK(tensorflow::Env::Default()->NewRandomAccessFile(*f, &file));

    SequentialRecordReader reader(
        file.get(), RecordReaderOptions::CreateRecordReaderOptions(kZlib));

    while (true) {
      tstring record;
      if (!reader.ReadRecord(&record).ok()) {
        break;
      }

      if (probability_.Uniform() < p_) {
        AppendToChunk(std::move(record));
      }

      ++num_scanned;
      if (num_scanned % kLoggingInterval == 0) {
        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<float> elapsed = now - start;
        LOG(INFO) << "Time so far: " << elapsed.count()
                  << "s. Num scanned so far: " << num_scanned
                  << ". Chunk size: " << chunk_.size() << ".";
      }
    }
  }
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<float> elapsed = end - start;
  LOG(INFO) << "Building chunk took " << elapsed.count() << "s. Scanned "
            << num_scanned << " protos.";

  std::vector<tstring> chunk(chunk_.begin(), chunk_.end());
  start = std::chrono::steady_clock::now();
  std::shuffle(chunk.begin(), chunk.end(), probability_.prng());
  end = std::chrono::steady_clock::now();
  elapsed = end - start;
  LOG(INFO) << "Shuffling chunk took " << elapsed.count() << "s.";

  return chunk;
}

void ChunkManager::Stop() {
  absl::MutexLock l(&mu_);
  running_ = false;
  cv_.SignalAll();
}

void ChunkManager::AppendToChunk(tstring&& proto) {
  chunk_.emplace_back(proto);
  if (chunk_.size() > chunk_size_) {
    chunk_.pop_front();
  }
}

void ChunkManager::FsThread() {
  while (true) {
    absl::MutexLock l(&mu_);
    cv_.WaitWithTimeout(&mu_, absl::Seconds(poll_interval_s_));
    if (!running_) {
      break;
    }

    std::vector<std::string> new_files = watcher_.UpdateAndGetNew();
    LOG(INFO) << "Found " << new_files.size() << " new files.";
    fbuffer_.AddNewFiles(new_files);
  }
}
}  // namespace shuffler
