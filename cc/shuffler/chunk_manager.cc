#include "cc/shuffler/chunk_manager.h"

#include <filesystem>  // compile with gcc 9+
#include <random>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "cc/shuffler/constants.h"
#include "tensorflow/core/lib/io/compression.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "tensorflow/core/lib/io/record_writer.h"

namespace shuffler {
namespace {
namespace fs = std::filesystem;
using ::tensorflow::tstring;
using ::tensorflow::io::RecordReaderOptions;
using ::tensorflow::io::RecordWriter;
using ::tensorflow::io::RecordWriterOptions;
using ::tensorflow::io::SequentialRecordReader;
using ::tensorflow::io::compression::kZlib;

static constexpr size_t kDefaultChunkSize = 2048000;
static constexpr int kDefaultPollIntervalS = 30;
static constexpr int kLoggingInterval = 1000000;

// keep in sync with python/gcs_utils.py
static constexpr char kChunkFormat[] = "chunk_%d.tfrecord.zz";

void WriteChunkToDisk(std::string filename, const std::vector<tstring>& chunk) {
  std::unique_ptr<tensorflow::WritableFile> file;
  TF_CHECK_OK(tensorflow::Env::Default()->NewWritableFile(filename, &file));

  RecordWriterOptions options;
  options.compression_type = RecordWriterOptions::ZLIB_COMPRESSION;
  options.zlib_options.compression_level = 2;
  RecordWriter writer(file.get(), options);

  for (const tstring& record : chunk) {
    TF_CHECK_OK(writer.WriteRecord(record));
  }

  TF_CHECK_OK(writer.Close());
  TF_CHECK_OK(file->Close());
}
}  // namespace

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
  fs_thread_ = std::thread(&ChunkManager::FsThread, this);
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

void ChunkManager::CreateChunk() {
  LOG(INFO) << "Creating Chunk...";

  int num_scanned = 0;
  auto start = std::chrono::steady_clock::now();
  while (true) {
    // Pop file to read, if one exists.
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
      LOG(INFO) << "No files remaining. Sleeping for " << poll_interval_s_
                << "s.";

      absl::MutexLock l(&mu_);
      cv_.WaitWithTimeout(&mu_, absl::Seconds(poll_interval_s_));
      if (!running_) {
        break;
      }
      continue;
    }

    // Read file into memory.
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
      LOG_IF(INFO, num_scanned % kLoggingInterval == 0)
          << "Time so far: "
          << std::chrono::duration<float>(std::chrono::steady_clock::now() -
                                          start)
                 .count()
          << "s. Num scanned so far: " << num_scanned
          << ". Chunk size: " << chunk_.size() << ".";
    }
  }
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<float> elapsed = end - start;
  LOG(INFO) << "Building chunk took " << elapsed.count() << "s. Scanned "
            << num_scanned << " protos.";
}

void ChunkManager::ShuffleAndFlush() {
  // shuffle chunk
  std::vector<tstring> golden_chunk(chunk_.begin(), chunk_.end());
  auto start = std::chrono::steady_clock::now();
  std::shuffle(golden_chunk.begin(), golden_chunk.end(), probability_.prng());
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<float> elapsed = end - start;
  LOG(INFO) << "Shuffling chunk took " << elapsed.count()
            << "s. Chunk contains " << golden_chunk.size() << " elements.";

  // create directory
  std::string chunk_dir = fs::path(dir_) / kGoldenChunkDirname;
  std::string chunk_filename =
      fs::path(chunk_dir) / absl::StrFormat(kChunkFormat, gen_);
  fs::create_directory(chunk_dir);

  // write to disk
  WriteChunkToDisk(chunk_filename, golden_chunk);
}

void ChunkManager::SignalStop() {
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
