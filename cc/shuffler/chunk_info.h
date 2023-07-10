#ifndef __SHUFFLER_CHUNK_INFO_H_
#define __SHUFFLER_CHUNK_INFO_H_

#include <optional>
#include <string>

namespace shuffler {

struct ChunkInfo {
  int gen;
  int batch;
  int games;
  int examples;
};

std::optional<ChunkInfo> ParseChunkFilename(std::string chunk_filename);

}  // namespace shuffler

#endif
