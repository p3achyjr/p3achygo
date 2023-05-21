#include "cc/shuffler/chunk_manager.h"

#include <chrono>
#include <random>
#include <string>
#include <vector>

#include "tensorflow/core/lib/io/compression.h"
#include "tensorflow/core/lib/io/record_reader.h"

namespace shuffler {
namespace {
static constexpr size_t kDefaultChunkSize = 2048000;
static constexpr int kDefaultPollInterval = 10000000;
static constexpr int kLoggingInterval = 1000000;
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
                   kDefaultPollInterval) {}

ChunkManager::ChunkManager(std::string dir, int gen, float p,
                           std::vector<int> exclude_gens, size_t chunk_size,
                           int poll_interval)
    : dir_(dir),
      gen_(gen),
      p_(p),
      chunk_size_(chunk_size),
      poll_interval_(poll_interval),
      exclude_gens_(exclude_gens),
      probability_(static_cast<uint64_t>(std::time(nullptr))),
      watcher_(dir_, exclude_gens_),
      fbuffer_(watcher_.GetFiles()) {}

std::vector<::tensorflow::tstring> ChunkManager::CreateChunk() {
  LOG(INFO) << "Creating Chunk...";
  LOG(INFO) << "Scanning " << watcher_.GetFiles().size() << " files.";

  int num_scanned = 0;
  auto start = std::chrono::steady_clock::now();
  while (!fbuffer_.empty()) {
    auto f = fbuffer_.PopFile();
    DCHECK(f != std::nullopt);

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
      if (num_scanned % poll_interval_ == 0) {
        // scan for new files.
        fbuffer_.AddNewFiles(watcher_.UpdateAndGetNew());
      }

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

void ChunkManager::AppendToChunk(tstring&& proto) {
  chunk_.emplace_back(proto);
  if (chunk_.size() > chunk_size_) {
    chunk_.pop_front();
  }
}
}  // namespace shuffler
