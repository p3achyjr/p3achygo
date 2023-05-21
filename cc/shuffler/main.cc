/*
 * Main function for shuffler binary.
 */

#include <iostream>

#include "cc/shuffler/chunk_manager.h"

int main(int argc, char** argv) {
  shuffler::ChunkManager chunk_manager("/tmp/p3achygo_data/tf", 0, .01f);

  auto chunk = chunk_manager.CreateChunk();
  std::cerr << "Chunk contains: " << chunk.size() << " elements.\n";

  return chunk.size();
}
