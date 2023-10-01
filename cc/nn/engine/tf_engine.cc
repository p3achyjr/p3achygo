#include "cc/nn/engine/tf_engine.h"

#include <vector>

#include "cc/game/color.h"
#include "cc/game/loc.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/public/session.h"

namespace nn {
namespace {
using namespace ::game;
using namespace ::tensorflow;

// Keep in sync with //python/model.py:call
static constexpr int kPiLogitsIndex = 0;
static constexpr int kQ30Index = 1;
static constexpr int kQ100Index = 2;
static constexpr int kQ200Index = 3;
static constexpr int kPiProbsIndex = 4;
static constexpr int kOutcomeLogitsIndex = 5;
static constexpr int kOutcomeIndex = 6;
static constexpr int kOwnIndex = 7;
static constexpr int kScoreLogitsIndex = 8;
static constexpr int kScoreProbsIndex = 9;
static constexpr int kGammaIndex = 10;
static constexpr int kPiLogitsAuxIndex = 11;

static constexpr char kSavedModelTagServe[] = "serve";

const std::vector<std::string> kInputNames = {"serving_default_args_0:0",
                                              "serving_default_args_1:0"};

const std::vector<std::string> kTfOutputNames = {
    "StatefulPartitionedCall:0",  "StatefulPartitionedCall:1",
    "StatefulPartitionedCall:2",  "StatefulPartitionedCall:3",
    "StatefulPartitionedCall:4",  "StatefulPartitionedCall:5",
    "StatefulPartitionedCall:6",  "StatefulPartitionedCall:7",
    "StatefulPartitionedCall:8",  "StatefulPartitionedCall:9",
    "StatefulPartitionedCall:10", "StatefulPartitionedCall:11",
};

const std::vector<std::string> kTrtOutputNames = {
    "PartitionedCall:0", "PartitionedCall:1",  "PartitionedCall:2",
    "PartitionedCall:3", "PartitionedCall:4",  "PartitionedCall:5",
    "PartitionedCall:6", "PartitionedCall:7",  "PartitionedCall:8",
    "PartitionedCall:9", "PartitionedCall:10", "PartitionedCall:11",
};

inline ::tensorflow::TensorShape CreateTensorShape(
    std::initializer_list<int64_t> dims) {
  ::tensorflow::TensorShape shape;
  for (const auto& dim : dims) {
    shape.AddDim(dim);
  }

  return shape;
}

/*
 * Wrapper around inference for TF-TRT engines.
 */
class TFEngineImpl : public TFEngine {
 public:
  TFEngineImpl(std::string path, Kind kind, int batch_size);
  ~TFEngineImpl() = default;

  Engine::Kind kind() override {
    return kind_ == TFEngine::Kind::kTF ? Engine::Kind::kTF
                                        : Engine::Kind::kTFTrt;
  }
  std::string path() override { return path_; }
  void LoadBatch(int batch_id, const GoFeatures& features) override;
  void RunInference() override;
  void GetBatch(int batch_id, NNInferResult& result) override;
  void GetOwnership(
      int batch_id,
      std::array<float, constants::kMaxMovesPerPosition>& own) override;

 private:
  std::vector<tensorflow::Tensor> nn_input_buf_;
  std::vector<tensorflow::Tensor> nn_output_buf_;
  tensorflow::SavedModelBundleLite model_bundle_;
  tensorflow::SessionOptions session_options_;
  tensorflow::RunOptions run_options_;
  const Kind kind_;
  const int batch_size_;
  const std::string path_;
};

TFEngineImpl::TFEngineImpl(std::string path, Kind kind, int batch_size)
    : session_options_(SessionOptions()),
      run_options_(RunOptions()),
      kind_(kind),
      batch_size_(batch_size),
      path_(path) {
  nn_input_buf_ = {
      Tensor(DataType::DT_FLOAT,
             CreateTensorShape({batch_size_, BOARD_LEN, BOARD_LEN,
                                constants::kNumInputFeaturePlanes})),
      Tensor(DataType::DT_FLOAT,
             CreateTensorShape(
                 {batch_size_, constants::kNumInputFeatureScalars}))};

  nn_input_buf_[0].flat<float>().setZero();
  nn_input_buf_[1].flat<float>().setZero();

  nn_output_buf_ = {
      // move logits
      Tensor(DataType::DT_FLOAT,
             CreateTensorShape({batch_size_, constants::kMaxMovesPerPosition})),
      // q30
      Tensor(DataType::DT_FLOAT, CreateTensorShape({batch_size_, 1})),
      // q100
      Tensor(DataType::DT_FLOAT, CreateTensorShape({batch_size_, 1})),
      // q200
      Tensor(DataType::DT_FLOAT, CreateTensorShape({batch_size_, 1})),
      // move softmax
      Tensor(DataType::DT_FLOAT,
             CreateTensorShape({batch_size_, constants::kMaxMovesPerPosition})),
      // win logits
      Tensor(DataType::DT_FLOAT,
             CreateTensorShape({batch_size_, constants::kNumValueLogits})),
      // win percent
      Tensor(DataType::DT_FLOAT,
             CreateTensorShape({batch_size_, constants::kNumValueLogits})),
      // ownership
      Tensor(DataType::DT_FLOAT,
             CreateTensorShape({batch_size_, BOARD_LEN, BOARD_LEN, 1})),
      // score logits
      Tensor(DataType::DT_FLOAT,
             CreateTensorShape({batch_size_, constants::kNumScoreLogits})),
      // score probabilities
      Tensor(DataType::DT_FLOAT,
             CreateTensorShape({batch_size_, constants::kNumScoreLogits})),
      // gamma, just ignore
      Tensor(DataType::DT_FLOAT, CreateTensorShape({batch_size_, 1})),
      // auxiliary move logits
      Tensor(
          DataType::DT_FLOAT,
          CreateTensorShape({batch_size_, constants::kMaxMovesPerPosition}))};

  TF_CHECK_OK(LoadSavedModel(session_options_, run_options_, path,
                             {kSavedModelTagServe}, &model_bundle_));
}

void TFEngineImpl::LoadBatch(int batch_id, const GoFeatures& features) {
  nn_input_buf_[0].SubSlice(batch_id).unaligned_flat<float>().setZero();
  nn_input_buf_[1].SubSlice(batch_id).unaligned_flat<float>().setZero();

  auto raw = nn_input_buf_[0].shaped<float, 4>(
      {batch_size_, BOARD_LEN, BOARD_LEN, constants::kNumInputFeaturePlanes});
  auto color = features.color;
  auto fill_plane_pair =
      [batch_id, color, &raw](
          const std::array<game::Color, constants::kNumBoardLocs>& grid,
          int our_index, int opp_index) {
        for (auto i = 0; i < BOARD_LEN; ++i) {
          for (auto j = 0; j < BOARD_LEN; ++j) {
            if (grid[i * BOARD_LEN + j] == color) {
              raw(batch_id, i, j, our_index) = 1;
            } else if (grid[i * BOARD_LEN + j] == OppositeColor(color)) {
              raw(batch_id, i, j, opp_index) = 1;
            }
          }
        }
      };

  // fill board state
  fill_plane_pair(features.board, 0, 1);

  // fill moves
  auto mv_offset = 2;
  for (auto i = 0; i < constants::kNumLastMoves; ++i) {
    Loc loc = AsLoc(features.last_moves[i]);
    if (loc == kNoopLoc) continue;
    if (loc == kPassLoc) continue;

    raw(batch_id, loc.i, loc.j, i + mv_offset) = 1;
  }

  // liberties
  fill_plane_pair(features.stones_atari, 7, 8);
  fill_plane_pair(features.stones_two_liberties, 9, 10);
  fill_plane_pair(features.stones_three_liberties, 11, 12);

  // fill input features.
  nn_input_buf_[1].matrix<float>()(batch_id, 0) = color == BLACK ? 1 : 0;
  nn_input_buf_[1].matrix<float>()(batch_id, 1) = color == WHITE ? 1 : 0;
  for (auto i = 0; i < constants::kNumLastMoves; ++i) {
    Loc loc = AsLoc(features.last_moves[i]);
    if (loc == kPassLoc) {
      nn_input_buf_[1].matrix<float>()(batch_id, i + mv_offset) = 1;
    }
  }
}

void TFEngineImpl::RunInference() {
  const std::vector<std::string>& output_names =
      kind_ == Kind::kTF ? kTfOutputNames : kTrtOutputNames;
  std::vector<std::pair<std::string, Tensor>> nn_input = {
      {kInputNames[0], nn_input_buf_[0]}, {kInputNames[1], nn_input_buf_[1]}};
  TF_CHECK_OK(model_bundle_.GetSession()->Run(nn_input, output_names, {},
                                              &nn_output_buf_));
}

void TFEngineImpl::GetBatch(int batch_id, NNInferResult& result) {
  const auto move_logits =
      nn_output_buf_[kPiLogitsIndex].SubSlice(batch_id).unaligned_flat<float>();
  const auto move_probs =
      nn_output_buf_[kPiProbsIndex].SubSlice(batch_id).unaligned_flat<float>();
  const auto value_probs =
      nn_output_buf_[kOutcomeIndex].SubSlice(batch_id).unaligned_flat<float>();
  const auto score_probs = nn_output_buf_[kScoreProbsIndex]
                               .SubSlice(batch_id)
                               .unaligned_flat<float>();
  for (int i = 0; i < constants::kMaxMovesPerPosition; ++i) {
    result.move_logits[i] = move_logits(i);
  }
  for (int i = 0; i < constants::kMaxMovesPerPosition; ++i) {
    result.move_probs[i] = move_probs(i);
  }
  for (int i = 0; i < constants::kNumValueLogits; ++i) {
    result.value_probs[i] = value_probs(i);
  }
  for (int i = 0; i < constants::kNumScoreLogits; ++i) {
    result.score_probs[i] = score_probs(i);
  }
}

void TFEngineImpl::GetOwnership(
    int batch_id, std::array<float, constants::kMaxMovesPerPosition>& own) {
  const auto own_slice =
      nn_output_buf_[kOwnIndex].SubSlice(batch_id).unaligned_flat<float>();
  for (int i = 0; i < constants::kMaxMovesPerPosition; ++i) {
    own[i] = own_slice(i);
  }
}

}  // namespace

/* static */ std::unique_ptr<TFEngine> TFEngine::Create(std::string path,
                                                        TFEngine::Kind kind,
                                                        int batch_size) {
  return std::make_unique<TFEngineImpl>(path, kind, batch_size);
}

}  // namespace nn
