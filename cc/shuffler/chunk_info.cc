#include "cc/shuffler/chunk_info.h"

#include <iostream>
#include <regex>

#include "absl/log/check.h"
#include "cc/data/filename_format.h"

namespace shuffler {

std::optional<ChunkInfo> ParseChunkFilename(std::string chunk_filename) {
  static const std::regex re(data::kChunkRegex);

  std::smatch match;
  if (!std::regex_match(chunk_filename, match, re)) {
    return std::nullopt;
  }

  CHECK(match.size() == 7);
  return ChunkInfo{
      std::atoi(match[1].str().c_str()), std::atoi(match[2].str().c_str()),
      std::atoi(match[3].str().c_str()), std::atoi(match[4].str().c_str()),
      std::atoi(match[5].str().c_str()),
  };
}

}  // namespace shuffler
