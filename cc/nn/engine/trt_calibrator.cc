#include "cc/nn/engine/trt_calibrator.h"

#include <NvInfer.h>

#include <deque>
#include <fstream>

#include "cc/constants/constants.h"
#include "cc/game/board.h"
#include "cc/game/color.h"
#include "cc/nn/engine/trt_names.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature_util.h"
#include "tensorflow/core/lib/io/compression.h"
#include "tensorflow/core/lib/io/record_reader.h"

namespace nn {
namespace trt {
namespace {

inline std::ostream& operator<<(std::ostream& os,
                                const game::Board::BoardData& board) {
  auto is_star_point = [](int i, int j) {
    return (i == 3 || i == 9 || i == 15) && (j == 3 || j == 9 || j == 15);
  };

  for (auto i = 0; i < BOARD_LEN; i++) {
    if (i < 10)
      os << i << "  ";
    else
      os << i << " ";
    for (auto j = 0; j < BOARD_LEN; j++) {
      if (board[i * BOARD_LEN + j] == EMPTY && is_star_point(i, j)) {
        os << "+ ";
      } else if (board[i * BOARD_LEN + j] == EMPTY) {
        os << "⋅ ";
      } else if (board[i * BOARD_LEN + j] == BLACK) {
        os << "○ ";
      } else if (board[i * BOARD_LEN + j] == WHITE) {
        os << "● ";
      }
    }

    os << "\n";
  }

  os << "   "
     << "A B C D E F G H I J K L M N O P Q R S";

  return os;
}

using ::tensorflow::GetFeatureValues;
using ::tensorflow::tstring;
using ::tensorflow::io::RecordReaderOptions;
using ::tensorflow::io::SequentialRecordReader;
using ::tensorflow::io::compression::kZlib;

template <typename T>
T ParseScalar(const std::string& s) {
  T val;
  memcpy(&val, s.data(), sizeof(T));
  return val;
}

template <typename T, size_t N>
std::array<T, N> ParseSequence(const std::string& s) {
  std::array<T, N> arr;
  memcpy(arr.data(), s.data(), sizeof(T) * N);
  return arr;
}

struct GoFeatures {
  int bsize;
  game::Color color;
  std::array<game::Color, constants::kNumBoardLocs> board;
  std::array<int16_t, constants::kNumLastMoves> last_moves;
  std::array<game::Color, constants::kNumBoardLocs> stones_atari;
  std::array<game::Color, constants::kNumBoardLocs> stones_two_liberties;
  std::array<game::Color, constants::kNumBoardLocs> stones_three_liberties;
};

class Int8CalibratorImpl : public Int8Calibrator {
 public:
  Int8CalibratorImpl(size_t batch_size, std::string calib_tfrec_path,
                     std::string calib_cache_path);
  ~Int8CalibratorImpl();
  void initialize();
  int getBatchSize() const noexcept override { return batch_size_; }
  bool getBatch(void* bindings[], const char* names[],
                int nbBindings) noexcept override;
  const void* readCalibrationCache(size_t& length) noexcept override;
  void writeCalibrationCache(const void* cache,
                             size_t length) noexcept override;

 private:
  static constexpr int kNumCalibBatches = 10;
  size_t batch_size_;
  std::string calib_tfrec_path_;
  int batch_counter_;
  int num_batches_;
  std::deque<GoFeatures> examples_;
  std::string calib_cache_path_;
  std::vector<char> calibration_cache_;
  size_t nbytes_planes_;
  void* host_input_planes_;
  void* device_input_planes_;
  size_t nbytes_features_;
  void* host_input_features_;
  void* device_input_features_;
};

Int8CalibratorImpl::Int8CalibratorImpl(size_t batch_size,
                                       std::string calib_tfrec_path,
                                       std::string calib_cache_path)
    : batch_size_(batch_size),
      calib_tfrec_path_(calib_tfrec_path),
      batch_counter_(0),
      calib_cache_path_(calib_cache_path) {
  initialize();
}

Int8CalibratorImpl::~Int8CalibratorImpl() {
  cudaFree(device_input_planes_);
  cudaFreeHost(host_input_planes_);
  cudaFree(device_input_features_);
  cudaFreeHost(host_input_features_);
}

void Int8CalibratorImpl::initialize() {
  std::cerr << "<<axlui>> INITIALIZING" << std::endl;
  std::unique_ptr<tensorflow::RandomAccessFile> file;
  TF_CHECK_OK(tensorflow::Env::Default()->NewRandomAccessFile(calib_tfrec_path_,
                                                              &file));

  SequentialRecordReader reader(
      file.get(), RecordReaderOptions::CreateRecordReaderOptions(kZlib));

  int total_num_examples = 0;
  for (int i = 0; i < batch_size_ * kNumCalibBatches; ++i) {
    tstring record;
    auto status = reader.ReadRecord(&record);
    if (status.code() == tensorflow::error::OUT_OF_RANGE) {
      break;  // EOF reached
    } else if (!status.ok()) {
      LOG(ERROR) << "Error reading TFRecord " << i;
      continue;
    }

    tensorflow::Example example;
    if (!example.ParseFromString(std::string(record))) {
      LOG(ERROR) << "Error parsing TFRecord" << i;
      continue;
    }

    // Read features as binary strings.
    auto& bsize_feat = GetFeatureValues<std::string>("bsize", example).Get(0);
    auto& board_feat = GetFeatureValues<std::string>("board", example).Get(0);
    auto& last_moves_feat =
        GetFeatureValues<std::string>("last_moves", example).Get(0);
    auto& stones_atari_feat =
        GetFeatureValues<std::string>("stones_atari", example).Get(0);
    auto& stones_two_liberties_feat =
        GetFeatureValues<std::string>("stones_two_liberties", example).Get(0);
    auto& stones_three_liberties_feat =
        GetFeatureValues<std::string>("stones_three_liberties", example).Get(0);
    auto& color_feat = GetFeatureValues<std::string>("color", example).Get(0);

    // Parse strings into desired format.
    int bsize = static_cast<int>(ParseScalar<uint8_t>(bsize_feat));
    CHECK(bsize == BOARD_LEN && bsize * bsize == constants::kNumBoardLocs);

    GoFeatures go_features;
    go_features.bsize = bsize;
    go_features.color = ParseScalar<game::Color>(color_feat);
    go_features.board =
        ParseSequence<game::Color, constants::kNumBoardLocs>(board_feat);
    go_features.last_moves =
        ParseSequence<int16_t, constants::kNumLastMoves>(last_moves_feat);
    go_features.stones_atari =
        ParseSequence<game::Color, constants::kNumBoardLocs>(stones_atari_feat);
    go_features.stones_two_liberties =
        ParseSequence<game::Color, constants::kNumBoardLocs>(
            stones_two_liberties_feat);
    go_features.stones_three_liberties =
        ParseSequence<game::Color, constants::kNumBoardLocs>(
            stones_three_liberties_feat);

    examples_.emplace_back(go_features);
    ++total_num_examples;
  }

  num_batches_ = total_num_examples / batch_size_;
  nbytes_planes_ = sizeof(float) * batch_size_ *
                   constants::kNumInputFeaturePlanes * BOARD_LEN * BOARD_LEN;
  nbytes_features_ =
      sizeof(float) * batch_size_ * constants::kNumInputFeatureScalars;
  cudaMalloc(&device_input_planes_, nbytes_planes_);
  cudaMallocHost(&host_input_planes_, nbytes_planes_);
  cudaMalloc(&device_input_features_, nbytes_features_);
  cudaMallocHost(&host_input_features_, nbytes_features_);
  std::cerr << "<<axlui>> INITIALIZED. NUM BATCHES: " << num_batches_
            << ", NUM EXAMPLES: " << examples_.size() << std::endl;
}

bool Int8CalibratorImpl::getBatch(void* bindings[], const char* names[],
                                  int nbBindings) noexcept {
  static constexpr int kBatchOffsetPlanes =
      BOARD_LEN * BOARD_LEN * constants::kNumInputFeaturePlanes;
  static constexpr int kBatchOffsetScalars = constants::kNumInputFeatureScalars;
  static constexpr int kChannelOffset = BOARD_LEN * BOARD_LEN;
  auto fill_plane_pair =
      [](float* buf, int batch_id, int our_channel, int opp_channel,
         std::array<game::Color, constants::kNumBoardLocs> board_data) {
        int batch_offset = kBatchOffsetPlanes * batch_id;
        int our_offset = batch_offset + our_channel * kChannelOffset;
        int opp_offset = batch_offset + opp_channel * kChannelOffset;
        for (int i = 0; i < BOARD_LEN; ++i) {
          for (int j = 0; j < BOARD_LEN; ++j) {
            int grid_offset = i * BOARD_LEN + j;
            auto color = board_data[grid_offset];
            if (color == BLACK) {
              buf[our_offset + grid_offset] = 1.0f;
            } else if (color == WHITE) {
              buf[opp_offset + grid_offset] = 1.0f;
            }
          }
        }
      };

  std::cerr << "<<axlui>> GETTING BATCH. NUM BINDINGS: " << nbBindings
            << std::endl;
  if (batch_counter_ >= num_batches_) {
    std::cerr << "<<axlui>> No More Batches.";
    return false;
  }

  std::vector<GoFeatures> batch_examples;
  for (int _ = 0; _ < batch_size_; ++_) {
    batch_examples.emplace_back(examples_.front());
    examples_.pop_front();
  }

  std::cerr << "Board:\n" << batch_examples[batch_size_ - 1].board;

  for (int i = 0; i < nbBindings; ++i) {
    const char* name = names[i];
    void* host_binding;
    void* device_binding;
    size_t nbytes;
    if (std::string(name) == input::kPlanesName ||
        std::string(name) == "args_0") {
      host_binding = host_input_planes_;
      device_binding = device_input_planes_;
      nbytes = nbytes_planes_;
    } else if (std::string(name) == input::kFeaturesName ||
               std::string(name) == "args_1") {
      host_binding = host_input_features_;
      device_binding = device_input_features_;
      nbytes = nbytes_features_;
    } else {
      LOG(FATAL) << "Unknown Binding Name: " << name;
    }

    memset(host_binding, 0, nbytes);
    std::cerr << "<<axlui>> BINDING NAME: " << std::string(name)
              << ", NBYTES: " << nbytes << std::endl;

    // Fill binding.
    for (int batch_id = 0; batch_id < batch_size_; ++batch_id) {
      const GoFeatures& example = batch_examples[i];
      if (std::string(name) == input::kPlanesName ||
          std::string(name) == "args_0") {
        float* host_float_buf = static_cast<float*>(host_binding);
        fill_plane_pair(host_float_buf, batch_id, 0, 1, example.board);
        fill_plane_pair(host_float_buf, batch_id, 7, 8, example.stones_atari);
        fill_plane_pair(host_float_buf, batch_id, 9, 10,
                        example.stones_two_liberties);
        fill_plane_pair(host_float_buf, batch_id, 11, 12,
                        example.stones_three_liberties);
        for (int i = 0; i < constants::kNumLastMoves; ++i) {
          int16_t last_move = example.last_moves[i];
          if (last_move < 0 || last_move == constants::kPassMoveEncoding) {
            continue;
          }

          int offset = batch_id * kBatchOffsetPlanes +
                       (i + 2) * kChannelOffset + last_move;
          host_float_buf[offset] = 1.0f;
        }
      } else if (std::string(name) == input::kFeaturesName ||
                 std::string(name) == "args_1") {
        float* host_float_buf = static_cast<float*>(host_binding);
        host_float_buf[example.color == BLACK ? 0 : 1] = 1.0f;
        for (int i = 0; i < constants::kNumLastMoves; ++i) {
          int16_t last_move = example.last_moves[i];
          if (last_move != constants::kPassMoveEncoding) {
            continue;
          }

          int offset = batch_id * kBatchOffsetScalars + (i + 2);
          host_float_buf[offset] = 1.0f;
        }
      }
    }

    // Copy to device.
    cudaMemcpy(device_binding, host_binding, nbytes, cudaMemcpyHostToDevice);
    bindings[i] = device_binding;
  }

  std::cerr << "<<axlui>> GOT BATCH. REMAINING EXAMPLES: " << examples_.size()
            << std::endl;

  ++batch_counter_;
  return true;
}

const void* Int8CalibratorImpl::readCalibrationCache(size_t& length) noexcept {
  calibration_cache_.clear();
  std::ifstream input(calib_cache_path_, std::ios::binary);
  input >> std::noskipws;
  if (input.good()) {
    std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
              std::back_inserter(calibration_cache_));
  }
  length = calibration_cache_.size();
  return length ? calibration_cache_.data() : nullptr;
}

void Int8CalibratorImpl::writeCalibrationCache(const void* cache,
                                               size_t length) noexcept {
  std::ofstream output(calib_cache_path_, std::ios::binary);
  output.write(reinterpret_cast<const char*>(cache), length);
}

}  // namespace

/* static */ std::unique_ptr<Int8Calibrator> Int8Calibrator::Create(
    size_t batch_size, std::string calib_tfrec_path,
    std::string calib_cache_path) {
  std::unique_ptr<Int8Calibrator> calibrator =
      std::make_unique<Int8CalibratorImpl>(batch_size, calib_tfrec_path,
                                           calib_cache_path);

  return calibrator;
}

}  // namespace trt
}  // namespace nn
