/*
 * Main function for self-play binary.
 */

#include <sys/stat.h>

#include <thread>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "cc/constants/constants.h"
#include "cc/core/filepath.h"
#include "cc/nn/engine/engine_factory.h"
#include "cc/nn/nn_interface.h"
#include "cc/recorder/dir.h"
#include "cc/recorder/game_recorder.h"
#include "cc/selfplay/reuse_buffer.h"
#include "cc/selfplay/self_play_thread.h"

ABSL_FLAG(std::string, model_path, "", "Path to model.");
ABSL_FLAG(int, num_threads, 1, "Number of threads to use.");
ABSL_FLAG(std::string, recorder_path, "",
          "Path to write SGF files and TF examples. 'sgf' and 'chunks' are "
          "appended to the path.");
ABSL_FLAG(int, flush_interval, 128, "Number of games to buffer before flush.");
ABSL_FLAG(int, max_moves, 600, "Maximum number of moves per game.");
ABSL_FLAG(int, gen, 0, "Model generation we are generating games from.");
ABSL_FLAG(std::string, id, "", "Worker ID.");

// MCTS knobs.
ABSL_FLAG(int, gumbel_selected_k, 8,
          "Number of High Playout Cap Randomization moves to sample.");
ABSL_FLAG(int, gumbel_selected_n, 128,
          "Number of High Playout Cap Randomization visits.");
ABSL_FLAG(int, gumbel_default_k, 5,
          "Number of Low Playout Cap Randomization moves to sample.");
ABSL_FLAG(int, gumbel_default_n, 32,
          "Number of Low Playout Cap Randomization visits.");
ABSL_FLAG(std::string, reuse_buffer_type, "goexploit",
          "Reuse buffer type: 'goexploit', 'regret', or 'composite'.");
ABSL_FLAG(
    float, use_seen_state_prob, 0.5f,
    "Probability of drawing the initial game state from the reuse buffer.");
ABSL_FLAG(float, sel_mult_base, 0.0f,
          "Base multiplier for training position selection probability. "
          "Enables signal-based sampling (|NN-MCTS| and top1/2 Q gap). "
          "0 disables (falls back to flat probability).");
ABSL_FLAG(float, sel_mult_prob, 1.0f,
          "Fraction of moves [0,1] where signal-based sel_mult is applied. "
          "Remaining fraction falls back to sel_mult=1.0.");
ABSL_FLAG(float, bias_cache_lambda, 0.0f,
          "Bias cache lambda (adjustment scale). 0 disables bias cache.");
ABSL_FLAG(float, bias_cache_alpha, 0.8f,
          "Bias cache alpha (visit count attenuation exponent).");
ABSL_FLAG(int, nonroot_var_scale_prior_visits, 10,
          "Prior visits for nonroot PUCT var-scaling. -1 disables.");

void WaitForSignal() {
  // any line from stdin is a shutdown signal.
  std::string signal;
  std::getline(std::cin, signal);

  LOG(INFO) << "Self-Play received shutdown signal. Shutting down...";
  selfplay::SignalStop();
}

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);

  std::string model_path = absl::GetFlag(FLAGS_model_path);
  if (model_path == "") {
    LOG(ERROR) << "No Model Path (--model_path) specified.";
    return 1;
  }

  core::FilePath recorder_path(absl::GetFlag(FLAGS_recorder_path));
  if (recorder_path == "") {
    LOG(WARNING) << "No Recorder Path Specified. SGF and TF files will be "
                    "written to /tmp/";
    recorder_path = "/tmp/";
  }

  std::string worker_id = absl::GetFlag(FLAGS_id);
  if (worker_id == "") {
    LOG(ERROR) << "No Worker ID (--id) specified.";
    return 1;
  }

  int perms =
      S_IRUSR | S_IWUSR | S_IXUSR | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH;
  mkdir((recorder_path / recorder::kSgfDir).c_str(), perms);
  mkdir((recorder_path / recorder::kChunkDir).c_str(), perms);

  int num_threads = absl::GetFlag(FLAGS_num_threads);
  if (num_threads > constants::kMaxNumThreads) {
    LOG(ERROR) << "Requesting " << num_threads << " threads, but only up to "
               << constants::kMaxNumThreads << " is supported.";
    return 1;
  }

  LOG(INFO) << "Max Hardware Concurrency: "
            << std::thread::hardware_concurrency() << ". Using " << num_threads
            << " threads.";

  // initialize NN evaluator.
  std::unique_ptr<nn::Engine> engine =
      nn::CreateEngine(nn::KindFromEnginePath(model_path), model_path,
                       num_threads, nn::GetVersionFromModelPath(model_path));
  std::unique_ptr<nn::NNInterface> nn_interface =
      std::make_unique<nn::NNInterface>(num_threads, std::move(engine));
  nn_interface->SetNumCacheLastMoves(1);

  // initialize serialization objects.
  std::unique_ptr<recorder::GameRecorder> game_recorder =
      recorder::GameRecorder::Create(recorder_path, num_threads,
                                     absl::GetFlag(FLAGS_flush_interval),
                                     absl::GetFlag(FLAGS_gen), worker_id);

  // initialize reuse buffer (uniform GoExploit or regret-guided).
  std::unique_ptr<selfplay::ReuseBuffer> reuse_buffer;
  std::string buffer_type = absl::GetFlag(FLAGS_reuse_buffer_type);
  if (buffer_type == "regret") {
    reuse_buffer.reset(static_cast<selfplay::ReuseBuffer*>(
        new selfplay::RegretGuidedBuffer()));
  } else if (buffer_type == "composite") {
    reuse_buffer.reset(static_cast<selfplay::ReuseBuffer*>(
        new selfplay::CompositeReuseBuffer()));
  } else if (buffer_type == "goexploit") {
    reuse_buffer.reset(static_cast<selfplay::ReuseBuffer*>(
        new selfplay::GoExploitReuseBuffer()));
  } else {
    LOG(ERROR) << "Unknown --reuse_buffer_type '" << buffer_type
               << "'. Must be one of: goexploit, regret, composite.";
    return 1;
  }
  LOG(INFO) << "Reuse Buffer=" << reuse_buffer->Name()
            << "  use_seen_state_prob="
            << absl::GetFlag(FLAGS_use_seen_state_prob);

  LOG(INFO) << "========== Self-Play Config ==========";
  LOG(INFO) << "  gumbel_selected: n=" << absl::GetFlag(FLAGS_gumbel_selected_n)
            << "  k=" << absl::GetFlag(FLAGS_gumbel_selected_k);
  LOG(INFO) << "  gumbel_default:  n=" << absl::GetFlag(FLAGS_gumbel_default_n)
            << "  k=" << absl::GetFlag(FLAGS_gumbel_default_k);
  LOG(INFO) << "  sel_mult_base=" << absl::GetFlag(FLAGS_sel_mult_base)
            << "  sel_mult_prob=" << absl::GetFlag(FLAGS_sel_mult_prob);
  LOG(INFO) << "  bias_cache_lambda=" << absl::GetFlag(FLAGS_bias_cache_lambda)
            << "  bias_cache_alpha=" << absl::GetFlag(FLAGS_bias_cache_alpha);
  LOG(INFO) << "  nonroot_var_scale_prior_visits="
            << absl::GetFlag(FLAGS_nonroot_var_scale_prior_visits);
  LOG(INFO) << "=======================================";

  std::vector<std::string> sink_names;
  for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
    sink_names.push_back(
        absl::StrFormat("/tmp/thread%d_%s_log.txt", thread_id, worker_id));
  }
  std::vector<std::thread> threads;
  for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
    size_t seed = absl::HashOf(worker_id, thread_id);
    std::thread thread(
        selfplay::Run, seed, thread_id, nn_interface.get(), game_recorder.get(),
        reuse_buffer.get(), sink_names[thread_id],
        selfplay::SPConfig{
            absl::GetFlag(FLAGS_max_moves),
            selfplay::GumbelParams{absl::GetFlag(FLAGS_gumbel_selected_n),
                                   absl::GetFlag(FLAGS_gumbel_selected_k)},
            selfplay::GumbelParams{absl::GetFlag(FLAGS_gumbel_default_n),
                                   absl::GetFlag(FLAGS_gumbel_default_k)},
            absl::GetFlag(FLAGS_use_seen_state_prob),
            absl::GetFlag(FLAGS_sel_mult_base),
            absl::GetFlag(FLAGS_sel_mult_prob),
            absl::GetFlag(FLAGS_bias_cache_lambda),
            absl::GetFlag(FLAGS_bias_cache_alpha),
            absl::GetFlag(FLAGS_nonroot_var_scale_prior_visits)});
    threads.emplace_back(std::move(thread));
  }

  LOG(INFO) << "Spawned " << num_threads << " threads.";

  // Block until we receive signal from stdin.
  WaitForSignal();

  for (auto& thread : threads) {
    thread.join();
  }

  // Delete logfiles
  for (const auto& sink_name : sink_names) {
    std::remove(sink_name.c_str());
  }

  LOG(INFO) << "Self-Play Done!";
  return 0;
}
