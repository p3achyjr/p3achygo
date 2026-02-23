#include "cc/nn/engine/go_dataset.h"

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "cc/constants/constants.h"
#include "cc/data/tfrecord/record_reader.h"
#include "cc/proto/feature_util.h"
#include "example.pb.h"

namespace nn {
namespace {
using ::data::RecordReaderOptions;
using ::data::SequentialRecordReader;
using ::tensorflow::Example;
using ::tensorflow::GetFeatureValues;

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
  SequentialRecordReader reader(ds_path, RecordReaderOptions::Zlib());
  CHECK(reader.Init().ok()) << "Failed to initialize reader for: " << ds_path;

  int num_examples = 0;
  while (true) {
    std::vector<Row> batch;
    batch.resize(batch_size_);
    bool is_last_batch = false;
    for (int i = 0; i < batch_size_; ++i) {
      std::string record;
      auto status = reader.ReadRecord(&record);
      if (absl::IsOutOfRange(status)) {
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
      auto& stones_in_ladder_feat =
          GetFeatureValues<std::string>("stones_in_ladder", example).Get(0);
      auto& color_feat = GetFeatureValues<std::string>("color", example).Get(0);
      auto& policy = GetFeatureValues<std::string>("pi", example).Get(0);
      auto& score_margin =
          GetFeatureValues<float>("score_margin", example).Get(0);
      auto& komi_feat = GetFeatureValues<float>("komi", example).Get(0);

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
      go_features.stones_laddered =
          ParseSequence<game::Color, constants::kNumBoardLocs>(
              stones_in_ladder_feat);
      go_features.komi = komi_feat;

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
