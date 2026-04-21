#ifndef DATA_FILENAME_FORMATS_H_
#define DATA_FILENAME_FORMATS_H_

namespace data {

/*
 * Centralized location for file formats.
 */

// Format for selfplay chunk.
static constexpr char kChunkFormat[] =
    "gen%04d_b%03d_g%03d_n%05d_t%d_%s.tfrecord.zz";

// Format for lock-file, written once selfplay writing is finished.
static constexpr char kChunkDoneFormat[] =
    "gen%04d_b%03d_g%03d_n%05d_t%d_%s.done";

// Regex for parsing selfplay chunk files.
static constexpr char kChunkRegex[] =
    "gen(\\d+)_b(\\d+)_g(\\d+)_n(\\d+)_t(\\d+)_(.*)\\.tfrecord\\.zz";

// Regex for parsing selfplay chunk .DONE files.
static constexpr char kChunkDoneRegex[] =
    "gen(\\d+)_b(\\d+)_g(\\d+)_n(\\d+)_t(\\d+)_(.*)\\.done";

// Format for SGFs
static constexpr char kSgfFormat[] = "gen%04d_b%03d_g%03d_%s.sgf";

// Format for SGF lock-file.
static constexpr char kSgfDoneFormat[] = "gen%04d_b%03d_g%03d_%s.done";

// Format for SGFs with full game trees.
static constexpr char kSgfFullFormat[] = "FULL_gen%04d_b%03d_g%03d_%s.sgf";

// Format for game-tree SGF lock-file.
static constexpr char kSgfFullDoneFormat[] = "FULL_gen%04d_b%03d_g%03d_%s.done";

// Format for per-batch visit counts file.
static constexpr char kVisitCountFormat[] =
    "gen%04d_b%03d_g%03d_n%05d_t%d_%s.visit_count";

// Format for per-batch search stats file.
static constexpr char kStatsFormat[] = "gen%04d_b%03d_g%03d_n%05d_t%d_%s.stats";

// keep in sync with python/gcs_utils.py
static constexpr char kGoldenChunkFormat[] = "chunk_%04d.tfrecord.zz";

// keep in sync with python/gcs_utils.py
static constexpr char kGoldenChunkSizeFormat[] = "chunk_%04d.size";

}  // namespace data

#endif
