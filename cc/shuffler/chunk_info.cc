#include "cc/shuffler/chunk_info.h"

#include <iostream>
#include <regex>

#include "absl/log/check.h"

namespace shuffler {
namespace {
// Keep in sync with //cc/recorder/tf_recorder.cc
static constexpr char kChunkRegex[] =
    "gen(\\d+)_b(\\d+)_g(\\d+)_n(\\d+).tfrecord.zz";
}  // namespace

std::optional<ChunkInfo> ParseChunkFilename(std::string chunk_filename) {
  static const std::regex re(kChunkRegex);

  std::smatch match;
  if (!std::regex_match(chunk_filename, match, re)) {
    return std::nullopt;
  }

  CHECK(match.size() == 5);
  return ChunkInfo{
      std::atoi(match[1].str().c_str()),
      std::atoi(match[2].str().c_str()),
      std::atoi(match[3].str().c_str()),
      std::atoi(match[4].str().c_str()),
  };
}

}  // namespace shuffler
