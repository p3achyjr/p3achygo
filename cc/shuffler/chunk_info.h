#ifndef SHUFFLER_CHUNK_INFO_H_
#define SHUFFLER_CHUNK_INFO_H_

#include <optional>
#include <string>

namespace shuffler {

struct ChunkInfo {
  int gen;
  int batch;
  int num_games;
  int num_examples;
  int timestamp;
  std::string worker_id;
};

std::optional<ChunkInfo> ParseChunkFilename(std::string chunk_filename);

}  // namespace shuffler

#endif
