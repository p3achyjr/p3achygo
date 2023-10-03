#include "cc/nn/engine/go_dataset.h"

#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature_util.h"
#include "tensorflow/core/lib/io/compression.h"
#include "tensorflow/core/lib/io/record_reader.h"

namespace nn {
namespace {
using ::tensorflow::Example;
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
}  // namespace

GoDataset::GoDataset(size_t batch_size, std::string ds_path)
    : index_(0), batch_size_(batch_size) {
  std::unique_ptr<tensorflow::RandomAccessFile> file;
  TF_CHECK_OK(tensorflow::Env::Default()->NewRandomAccessFile(ds_path, &file));
  SequentialRecordReader reader(
      file.get(), RecordReaderOptions::CreateRecordReaderOptions(kZlib));

  int num_examples = 0;
  while (true) {
    std::vector<Row> batch;
    batch.resize(batch_size_);
    bool is_last_batch = false;
    for (int i = 0; i < batch_size_; ++i) {
      tstring record;
      auto status = reader.ReadRecord(&record);
      if (status.code() == tensorflow::error::OUT_OF_RANGE) {
        is_last_batch = true;
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
          GetFeatureValues<std::string>("stones_three_liberties", example)
              .Get(0);
      auto& color_feat = GetFeatureValues<std::string>("color", example).Get(0);
      auto& policy = GetFeatureValues<std::string>("pi", example).Get(0);
      auto& score_margin =
          GetFeatureValues<float>("score_margin", example).Get(0);

      // Parse strings into desired format.
      int bsize = static_cast<int>(ParseScalar<uint8_t>(bsize_feat));
      CHECK(bsize == BOARD_LEN && bsize * bsize == constants::kNumBoardLocs);

      std::array<game::Loc, constants::kNumLastMoves> last_moves;
      std::array<int16_t, constants::kNumLastMoves> last_move_encodings =
          ParseSequence<int16_t, constants::kNumLastMoves>(last_moves_feat);
      std::transform(last_move_encodings.begin(), last_move_encodings.end(),
                     last_moves.begin(),
                     [](int16_t mv) { return game::AsLoc(mv); });

      GoFeatures go_features;
      go_features.bsize = bsize;
      go_features.color = ParseScalar<game::Color>(color_feat);
      go_features.board =
          ParseSequence<game::Color, constants::kNumBoardLocs>(board_feat);
      go_features.last_moves = last_moves;
      go_features.stones_atari =
          ParseSequence<game::Color, constants::kNumBoardLocs>(
              stones_atari_feat);
      go_features.stones_two_liberties =
          ParseSequence<game::Color, constants::kNumBoardLocs>(
              stones_two_liberties_feat);
      go_features.stones_three_liberties =
          ParseSequence<game::Color, constants::kNumBoardLocs>(
              stones_three_liberties_feat);

      GoLabels go_labels;
      go_labels.policy =
          ParseSequence<float, constants::kMaxMovesPerPosition>(policy);
      go_labels.score_margin = score_margin;
      go_labels.did_win = go_labels.score_margin >= 0;
      ++num_examples;

      batch[i] = Row{go_features, go_labels};
    }

    batches_.emplace_back(batch);
    if (is_last_batch) break;
  }
}

}  // namespace nn
