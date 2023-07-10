/*
 * Main function for dataset creation.
 */
#include <filesystem>
#include <thread>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "cc/data/coordinator.h"

namespace {
namespace fs = std::filesystem;
}

ABSL_FLAG(std::string, sgf_dir, "", "Path to SGFs");
ABSL_FLAG(bool, dry_run, false, "Whether to perform a dry run");
ABSL_FLAG(std::string, out_dir, "", "Path to write dataset to.");

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverity::kInfo);

  std::string sgf_dir = absl::GetFlag(FLAGS_sgf_dir);
  if (sgf_dir == "") {
    LOG(ERROR) << "No --sgf_dir specified.";
  }

  if (!absl::GetFlag(FLAGS_dry_run) && absl::GetFlag(FLAGS_out_dir) == "") {
    LOG(ERROR) << "No --out_dir specified.";
  }

  std::string out_dir = fs::path(absl::GetFlag(FLAGS_out_dir));
  LOG(INFO) << "Out Dir: " << out_dir;

  int num_workers = std::thread::hardware_concurrency();
  LOG(INFO) << "Using " << num_workers << " workers.";

  data::Coordinator coordinator(num_workers, sgf_dir, out_dir,
                                absl::GetFlag(FLAGS_dry_run));
  coordinator.Run();

  return 0;
}
