// Picks the root value histograms (mcts_value_dist) with the heaviest far
// tails and the clearest two-hump shape from a set of golden chunks, and dumps
// the picked rows (board, last moves, pi top 5, outcome fields) as JSON.
// C++ port of claude/vdist_20261122/pick_rows2.py + extract.py (same criteria).
//
// Rows are deduped on (board, color, komi, last_moves, mcts_value_dist).
// Heavy tail: share of leaves > 0.5 from the median and > 3 robust sd
//   (IQR/1.349, floored at one bucket width) from it; rows with N >= 150 and
//   >= 60% of leaves within 3 robust sd.
// Two hump: 2-component normal mixture (80 EM iterations, init at the
//   quartiles) on rows with N >= 200 and sd > 0.12; kept if each weight >= 0.3,
//   |mu| < 0.85, sigmas in [0.05, 0.2], |mu1 - mu2| in [0.3, 0.8], and the
//   [1,2,1]-smoothed histogram has >= 2 modes (each >= 15% of leaves within
//   +-2 buckets, valley <= 0.5 x smaller peak). Ranked by
//   D = sqrt(2)|mu1 - mu2| / sqrt(s1^2 + s2^2).

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "absl/strings/str_split.h"
#include "cc/data/tfrecord/record_reader.h"
#include "cc/proto/feature_util.h"
#include "example.pb.h"

ABSL_FLAG(std::string, chunks, "", "Comma-separated golden chunk paths.");
ABSL_FLAG(std::string, ids, "", "Comma-separated golden ids (same order).");
ABSL_FLAG(std::string, out_path, "", "Output JSON.");
ABSL_FLAG(int, top, 10, "Rows per category.");
ABSL_FLAG(std::string, mode, "top",
          "top: heaviest tails / clearest two humps (rules above). bands: "
          "analysis-rule heavy tail (>5% of leaves beyond median +- 3 robust "
          "sd in atanh space) and two peaks (modes at 10% mass, valley <= "
          "0.5); random rows from the middle 5% and the 5% nearest the "
          "cutoff of each strength, plus shares of moves.");
ABSL_FLAG(uint64_t, seed, 20261125, "Sampling seed (bands mode).");

namespace {

using ::tensorflow::Example;
using ::tensorflow::GetFeatureValues;

constexpr int kNB = 51;
constexpr double kW = 2.0 / kNB;
constexpr int kKeep = 200;  // per-thread candidates kept before dedupe

double Center(int i) { return -1 + (i + 0.5) * kW; }

uint64_t Fnv(uint64_t h, const std::string& s) {
  for (unsigned char c : s) h = (h ^ c) * 1099511628211ULL;
  return h;
}

struct Pick {
  double metric;
  uint64_t hash;
  int golden;
  int row;
  double w[2], mu[2], s[2];
  std::string record;
};

struct Stats {
  uint64_t hash;
  bool heavy_gt20, two_hump, n200, heavy_elig;
};

// Index of the first bucket whose cumulative share reaches p.
int QIdx(const std::array<double, kNB>& cs, double p) {
  for (int i = 0; i < kNB; ++i)
    if (cs[i] >= p) return i;
  return kNB - 1;
}

double HeavyTail(const std::array<double, kNB>& P, int N) {
  std::array<double, kNB> cs;
  double acc = 0;
  for (int i = 0; i < kNB; ++i) cs[i] = acc += P[i];
  const double med = Center(QIdx(cs, .5));
  const double rsd =
      std::max((Center(QIdx(cs, .75)) - Center(QIdx(cs, .25))) / 1.349, kW);
  double core = 0, tail = 0;
  for (int i = 0; i < kNB; ++i) {
    const double d = std::abs(Center(i) - med);
    if (d <= 3 * rsd) core += P[i];
    if (d > 3 * rsd && d > 0.5) tail += P[i];
  }
  return (N >= 150 && core >= 0.6) ? tail : -1;
}

void Gmm2(const std::array<double, kNB>& P, double w[2], double mu[2],
          double s[2]) {
  std::array<double, kNB> cs;
  double acc = 0;
  for (int i = 0; i < kNB; ++i) cs[i] = acc += P[i];
  mu[0] = Center(QIdx(cs, .25)) - 1e-3;
  mu[1] = Center(QIdx(cs, .75)) + 1e-3;
  s[0] = s[1] = 0.1;
  w[0] = w[1] = 0.5;
  for (int it = 0; it < 80; ++it) {
    for (int k = 0; k < 2; ++k) s[k] = std::max(s[k], kW / 2);
    double nk[2] = {0, 0}, mk[2] = {0, 0};
    std::array<double, kNB> r0, r1;
    for (int i = 0; i < kNB; ++i) {
      double l[2];
      for (int k = 0; k < 2; ++k) {
        const double z = (Center(i) - mu[k]) / s[k];
        l[k] = w[k] * std::exp(-0.5 * z * z) / s[k];
      }
      const double tot = std::max(l[0] + l[1], 1e-300);
      r0[i] = l[0] / tot * P[i];
      r1[i] = l[1] / tot * P[i];
      nk[0] += r0[i];
      nk[1] += r1[i];
      mk[0] += r0[i] * Center(i);
      mk[1] += r1[i] * Center(i);
    }
    for (int k = 0; k < 2; ++k) {
      nk[k] += 1e-12;
      w[k] = nk[k];
      mu[k] = mk[k] / nk[k];
    }
    double v[2] = {0, 0};
    for (int i = 0; i < kNB; ++i) {
      v[0] += r0[i] * (Center(i) - mu[0]) * (Center(i) - mu[0]);
      v[1] += r1[i] * (Center(i) - mu[1]) * (Center(i) - mu[1]);
    }
    for (int k = 0; k < 2; ++k) s[k] = std::sqrt(v[k] / nk[k] + kW * kW / 12);
  }
}

// *ratio (if given): min over adjacent kept modes of valley / smaller peak.
int NModes(const std::array<uint32_t, kNB>& H, int N, double frac, double dip,
           double* ratio = nullptr) {
  std::array<double, kNB> S;
  for (int i = 0; i < kNB; ++i) {
    const double l = i > 0 ? H[i - 1] : 0, r = i < kNB - 1 ? H[i + 1] : 0;
    S[i] = (l + 2.0 * H[i] + r) / 4;
  }
  std::vector<int> peaks;
  for (int i = 0; i < kNB; ++i) {
    const double l = i > 0 ? S[i - 1] : -1, r = i < kNB - 1 ? S[i + 1] : -1;
    double win = 0;
    for (int j = std::max(0, i - 2); j < std::min(kNB, i + 3); ++j) win += H[j];
    if (S[i] > l && S[i] >= r && S[i] > 0 && win >= frac * N)
      peaks.push_back(i);
  }
  std::vector<int> kept;
  for (int p : peaks) {
    if (kept.empty()) {
      kept.push_back(p);
      continue;
    }
    const int q = kept.back();
    const double valley = *std::min_element(S.begin() + q, S.begin() + p + 1);
    if (valley <= dip * std::min(S[q], S[p])) {
      kept.push_back(p);
    } else if (S[p] > S[q]) {
      kept.back() = p;
    }
  }
  if (ratio != nullptr) {
    *ratio = 1e9;
    for (size_t k = 1; k < kept.size(); ++k) {
      const int q = kept[k - 1], p = kept[k];
      const double valley = *std::min_element(S.begin() + q, S.begin() + p + 1);
      *ratio = std::min(*ratio, valley / std::min(S[q], S[p]));
    }
  }
  return kept.size();
}

void Insert(std::vector<Pick>& v, Pick p) {
  v.push_back(std::move(p));
  if (v.size() > 2 * kKeep) {
    std::nth_element(
        v.begin(), v.begin() + kKeep, v.end(),
        [](const Pick& a, const Pick& b) { return a.metric > b.metric; });
    v.resize(kKeep);
  }
}

void Scan(const std::string& path, int golden, std::vector<Pick>* heavy,
          std::vector<Pick>* bimodal, std::vector<Stats>* stats) {
  data::SequentialRecordReader reader(path, data::RecordReaderOptions::Zlib());
  CHECK(reader.Init().ok()) << path;
  std::string record;
  for (int row = 0; reader.ReadRecord(&record).ok(); ++row) {
    Example ex;
    CHECK(ex.ParseFromString(record));
    const std::string& d =
        GetFeatureValues<std::string>("mcts_value_dist", ex).Get(0);
    CHECK_EQ(d.size(), kNB * sizeof(uint32_t));
    std::array<uint32_t, kNB> H;
    std::memcpy(H.data(), d.data(), d.size());
    int N = 0;
    for (uint32_t x : H) N += x;
    uint64_t h = 1469598103934665603ULL;
    h = Fnv(h, GetFeatureValues<std::string>("board", ex).Get(0));
    h = Fnv(h, GetFeatureValues<std::string>("color", ex).Get(0));
    const float komi = GetFeatureValues<float>("komi", ex).Get(0);
    h = Fnv(h, std::string(reinterpret_cast<const char*>(&komi), 4));
    h = Fnv(h, GetFeatureValues<std::string>("last_moves", ex).Get(0));
    h = Fnv(h, d);
    Stats st{h, false, false, N >= 200, false};
    if (N == 0) {
      stats->push_back(st);
      continue;
    }
    std::array<double, kNB> P;
    double m = 0;
    for (int i = 0; i < kNB; ++i) m += (P[i] = double(H[i]) / N) * Center(i);
    double var = 0;
    for (int i = 0; i < kNB; ++i)
      var += P[i] * (Center(i) - m) * (Center(i) - m);

    const double ht = HeavyTail(P, N);
    st.heavy_elig = ht >= 0;
    st.heavy_gt20 = ht > 0.2;
    if (ht >= 0) Insert(*heavy, Pick{ht, h, golden, row, {}, {}, {}, record});

    if (N >= 200 && std::sqrt(var) > 0.12) {
      Pick p{0, h, golden, row, {}, {}, {}, {}};
      Gmm2(P, p.w, p.mu, p.s);
      const double gap = std::abs(p.mu[0] - p.mu[1]);
      if (std::min(p.w[0], p.w[1]) >= 0.3 &&
          std::max(std::abs(p.mu[0]), std::abs(p.mu[1])) < 0.85 &&
          std::min(p.s[0], p.s[1]) >= 0.05 && std::max(p.s[0], p.s[1]) <= 0.2 &&
          gap >= 0.3 && gap <= 0.8 && NModes(H, N, 0.15, 0.5) >= 2) {
        st.two_hump = true;
        p.metric =
            std::sqrt(2.0) * gap / std::sqrt(p.s[0] * p.s[0] + p.s[1] * p.s[1]);
        p.record = record;
        Insert(*bimodal, std::move(p));
      }
    }
    stats->push_back(st);
  }
  LOG(INFO) << path << ": " << stats->size() << " rows";
}

std::string Gtp(int idx) {
  static const char* kCols = "ABCDEFGHJKLMNOPQRST";
  if (idx < 0) return "null";
  if (idx == 361) return "\"pass\"";
  return "\"" + std::string(1, kCols[idx % 19]) +
         std::to_string(19 - idx / 19) + "\"";
}

std::string RowJson(const Pick& p, const std::string& cat, int rank) {
  Example ex;
  CHECK(ex.ParseFromString(p.record));
  auto bytes = [&](const char* k) {
    return GetFeatureValues<std::string>(k, ex).Get(0);
  };
  auto fl = [&](const char* k) {
    return GetFeatureValues<float>(k, ex).Get(0);
  };
  const std::string board = bytes("board");
  const std::string lm = bytes("last_moves");
  const std::string pis = bytes("pi");
  const std::string dist = bytes("mcts_value_dist");
  const std::string aux = bytes("pi_aux");
  std::vector<float> pi(pis.size() / 4);
  std::memcpy(pi.data(), pis.data(), pis.size());
  std::vector<int> order(pi.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::partial_sort(order.begin(), order.begin() + 5, order.end(),
                    [&](int a, int b) { return pi[a] > pi[b]; });
  std::array<uint32_t, kNB> H;
  std::memcpy(H.data(), dist.data(), dist.size());
  int N = 0;
  double m = 0;
  for (int i = 0; i < kNB; ++i) N += H[i];
  for (int i = 0; i < kNB; ++i) m += double(H[i]) / N * Center(i);
  int16_t next;
  std::memcpy(&next, aux.data(), 2);
  char buf[512];
  std::string j = "{";
  std::snprintf(
      buf, sizeof buf,
      "\"id\":\"%d#%d\",\"golden\":%d,\"row\":%d,\"cat\":\"%s\",\"rank\":%d,"
      "\"metric\":%.4f,",
      p.golden, p.row, p.golden, p.row, cat.c_str(), rank, p.metric);
  j += buf;
  if (cat == "bimodal") {
    std::snprintf(
        buf, sizeof buf,
        "\"gmm\":{\"w\":[%.2f,%.2f],\"mu\":[%.2f,%.2f],\"s\":[%.2f,%.2f]},",
        p.w[0], p.w[1], p.mu[0], p.mu[1], p.s[0], p.s[1]);
    j += buf;
  } else {
    j += "\"gmm\":null,";
  }
  j += "\"board\":[";
  int stones = 0;
  for (size_t i = 0; i < board.size(); ++i) {
    j += (i ? "," : "") + std::to_string(int(int8_t(board[i])));
    stones += board[i] != 0;
  }
  j += "],\"last_moves\":[";
  for (size_t i = 0; i < lm.size() / 2; ++i) {
    int16_t mv;
    std::memcpy(&mv, lm.data() + 2 * i, 2);
    j += (i ? "," : "") + Gtp(mv);
  }
  j += "],\"dist\":[";
  for (int i = 0; i < kNB; ++i) j += (i ? "," : "") + std::to_string(H[i]);
  j += "],\"top5\":[";
  for (int k = 0; k < 5; ++k) {
    std::snprintf(buf, sizeof buf, "%s{\"move\":%s,\"idx\":%d,\"p\":%.4f}",
                  k ? "," : "", Gtp(order[k]).c_str(), order[k], pi[order[k]]);
    j += buf;
  }
  std::snprintf(buf, sizeof buf,
                "],\"to_move\":\"%s\",\"komi\":%.1f,\"stones\":%d,\"root_"
                "value\":%.3f,\"n\":%d,"
                "\"score_margin\":%.1f,\"q6\":%.3f,\"q16\":%.3f,\"q50\":%.3f,"
                "\"q6_score\":%.2f,"
                "\"q50_score\":%.2f,\"next_move\":%s}",
                int8_t(bytes("color")[0]) == 1 ? "B" : "W", fl("komi"), stones,
                m, N, fl("score_margin"), fl("q6"), fl("q16"), fl("q50"),
                fl("q6_score"), fl("q50_score"), Gtp(next).c_str());
  return j + buf;
}

// ---- bands mode ----

struct BandRow {
  uint64_t hash;
  int golden;
  int row;
  float far_z;  // share of leaves beyond median +- 3 robust sd, atanh space
  float ratio;  // valley / smaller peak (two-peak rows only)
  bool n50;
  bool two;
};

double FarZ(const std::array<uint32_t, kNB>& H, int N) {
  static const std::array<double, kNB> kZ = [] {
    std::array<double, kNB> z;
    for (int i = 0; i < kNB; ++i) z[i] = std::atanh(Center(i));
    return z;
  }();
  double min_gap = 1e9;
  for (int i = 1; i < kNB; ++i) min_gap = std::min(min_gap, kZ[i] - kZ[i - 1]);
  std::array<double, kNB> cs;
  double acc = 0;
  for (int i = 0; i < kNB; ++i) cs[i] = acc += double(H[i]) / N;
  const double med = kZ[QIdx(cs, .5)];
  const double rsd =
      std::max((kZ[QIdx(cs, .75)] - kZ[QIdx(cs, .25)]) / 1.349, min_gap);
  double far = 0;
  for (int i = 0; i < kNB; ++i)
    if (std::abs(kZ[i] - med) > 3 * rsd) far += H[i];
  return far / N;
}

void BandScan(const std::string& path, int golden, std::vector<BandRow>* out) {
  data::SequentialRecordReader reader(path, data::RecordReaderOptions::Zlib());
  CHECK(reader.Init().ok()) << path;
  std::string record;
  for (int row = 0; reader.ReadRecord(&record).ok(); ++row) {
    Example ex;
    CHECK(ex.ParseFromString(record));
    const std::string& d =
        GetFeatureValues<std::string>("mcts_value_dist", ex).Get(0);
    std::array<uint32_t, kNB> H;
    std::memcpy(H.data(), d.data(), d.size());
    int N = 0;
    for (uint32_t x : H) N += x;
    uint64_t h = 1469598103934665603ULL;
    h = Fnv(h, GetFeatureValues<std::string>("board", ex).Get(0));
    h = Fnv(h, GetFeatureValues<std::string>("color", ex).Get(0));
    const float komi = GetFeatureValues<float>("komi", ex).Get(0);
    h = Fnv(h, std::string(reinterpret_cast<const char*>(&komi), 4));
    h = Fnv(h, GetFeatureValues<std::string>("last_moves", ex).Get(0));
    h = Fnv(h, d);
    BandRow r{h, golden, row, 0, 0, N >= 50, false};
    if (r.n50) {
      r.far_z = FarZ(H, N);
      double ratio;
      r.two = NModes(H, N, 0.10, 0.5, &ratio) >= 2;
      if (r.two) r.ratio = ratio;
    }
    out->push_back(r);
  }
}

void FetchRecords(const std::string& path, int golden,
                  std::vector<Pick*> picks) {
  data::SequentialRecordReader reader(path, data::RecordReaderOptions::Zlib());
  CHECK(reader.Init().ok());
  std::string record;
  int max_row = 0;
  for (Pick* p : picks) max_row = std::max(max_row, p->row);
  for (int row = 0; row <= max_row && reader.ReadRecord(&record).ok(); ++row) {
    for (Pick* p : picks)
      if (p->golden == golden && p->row == row) p->record = record;
  }
}

int RunBands(const std::vector<std::string>& chunks,
             const std::vector<std::string>& ids) {
  const int n = chunks.size();
  std::vector<std::vector<BandRow>> per(n);
  std::vector<std::thread> threads;
  for (int g = 0; g < n; ++g)
    threads.emplace_back(BandScan, chunks[g], std::stoi(ids[g]), &per[g]);
  for (auto& t : threads) t.join();
  threads.clear();

  absl::flat_hash_set<uint64_t> seen;
  std::vector<BandRow> rows;
  int64_t total = 0;
  for (const auto& v : per) {
    for (const BandRow& r : v) {
      ++total;
      if (seen.insert(r.hash).second && r.n50) rows.push_back(r);
    }
  }
  int64_t n_heavy = 0, n_two = 0, n_either = 0, n_both = 0;
  std::vector<const BandRow*> heavy, two;
  for (const BandRow& r : rows) {
    const bool hv = r.far_z > 0.05;
    n_heavy += hv;
    n_two += r.two;
    n_either += hv || r.two;
    n_both += hv && r.two;
    if (hv) heavy.push_back(&r);
    if (r.two) two.push_back(&r);
  }
  // Strength order: heavy by far_z descending; two-peak by ratio ascending
  // (lower valley = clearer). "slight" = the 5% of qualifiers nearest the
  // cutoff; "median" = the middle 5% (47.5-52.5th percentile).
  std::stable_sort(heavy.begin(), heavy.end(),
                   [](auto a, auto b) { return a->far_z > b->far_z; });
  std::stable_sort(two.begin(), two.end(),
                   [](auto a, auto b) { return a->ratio < b->ratio; });
  std::mt19937_64 rng(absl::GetFlag(FLAGS_seed));
  const int top = absl::GetFlag(FLAGS_top);
  std::vector<std::pair<std::string, std::vector<Pick>>> cats;
  std::string windows;
  char buf[512];
  for (auto [name, v, is_heavy] : {std::tuple{"heavy", &heavy, true},
                                   std::tuple{"bimodal", &two, false}}) {
    const size_t m = v->size();
    for (auto [suffix, lo, hi] :
         {std::tuple{"_med", size_t(m * 0.475), size_t(m * 0.525)},
          std::tuple{"_slight", size_t(m * 0.95), m}}) {
      std::vector<const BandRow*> win(v->begin() + lo, v->begin() + hi);
      std::shuffle(win.begin(), win.end(), rng);
      win.resize(std::min<size_t>(top, win.size()));
      std::vector<Pick> picks;
      for (const BandRow* r : win) {
        Pick p{is_heavy ? r->far_z : r->ratio,
               r->hash,
               r->golden,
               r->row,
               {},
               {},
               {},
               {}};
        picks.push_back(p);
      }
      std::sort(picks.begin(), picks.end(), [&](auto& a, auto& b) {
        return is_heavy ? a.metric > b.metric : a.metric < b.metric;
      });
      auto val = [&](size_t i) {
        return is_heavy ? (*v)[i]->far_z : (*v)[i]->ratio;
      };
      std::snprintf(buf, sizeof buf,
                    "%s\"%s%s\":{\"from\":%.4f,\"to\":%.4f,\"n\":%zu}",
                    windows.empty() ? "" : ",", name, suffix, val(lo),
                    val(hi - 1), hi - lo);
      windows += buf;
      cats.push_back({std::string(name) + suffix, std::move(picks)});
    }
  }
  for (int g = 0; g < n; ++g) {
    std::vector<Pick*> want;
    for (auto& [c, ps] : cats)
      for (Pick& p : ps)
        if (p.golden == std::stoi(ids[g])) want.push_back(&p);
    if (!want.empty())
      threads.emplace_back(FetchRecords, chunks[g], std::stoi(ids[g]), want);
  }
  for (auto& t : threads) t.join();

  std::string out = "{\"rows\":[";
  bool first = true;
  for (auto& [c, ps] : cats) {
    int rank = 0;
    for (const Pick& p : ps) {
      CHECK(!p.record.empty());
      out += (first ? "" : ",") + RowJson(p, c, ++rank);
      first = false;
    }
  }
  std::snprintf(buf, sizeof buf,
                "],\"total\":%lld,\"rows_n50\":%zu,\"n_heavy\":%lld,"
                "\"n_two\":%lld,\"n_either\":%lld,\"n_both\":%lld,"
                "\"windows\":{",
                (long long)total, rows.size(), (long long)n_heavy,
                (long long)n_two, (long long)n_either, (long long)n_both);
  out += buf + windows + "}}";
  std::FILE* f = std::fopen(absl::GetFlag(FLAGS_out_path).c_str(), "w");
  CHECK(f);
  std::fputs(out.c_str(), f);
  std::fclose(f);
  std::printf("%s windows {%s}\n", buf + 2, windows.c_str());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  const std::vector<std::string> chunks =
      absl::StrSplit(absl::GetFlag(FLAGS_chunks), ',');
  const std::vector<std::string> ids =
      absl::StrSplit(absl::GetFlag(FLAGS_ids), ',');
  CHECK_EQ(chunks.size(), ids.size());
  if (absl::GetFlag(FLAGS_mode) == "bands") return RunBands(chunks, ids);
  const int n = chunks.size();
  std::vector<std::vector<Pick>> heavy(n), bimodal(n);
  std::vector<std::vector<Stats>> stats(n);
  std::vector<std::thread> threads;
  for (int g = 0; g < n; ++g) {
    threads.emplace_back(Scan, chunks[g], std::stoi(ids[g]), &heavy[g],
                         &bimodal[g], &stats[g]);
  }
  for (auto& t : threads) t.join();

  // Dedupe counts across goldens (first occurrence in golden order wins).
  absl::flat_hash_set<uint64_t> seen;
  int64_t total = 0, uniq = 0, n_gt20 = 0, n_elig = 0, n_two = 0, n200 = 0;
  for (const auto& v : stats) {
    for (const Stats& s : v) {
      ++total;
      if (!seen.insert(s.hash).second) continue;
      ++uniq;
      n_gt20 += s.heavy_gt20;
      n_elig += s.heavy_elig;
      n_two += s.two_hump;
      n200 += s.n200;
    }
  }
  const int top = absl::GetFlag(FLAGS_top);
  std::string out = "{\"rows\":[";
  bool first = true;
  for (auto [cat, src] :
       {std::pair{"heavy", &heavy}, std::pair{"bimodal", &bimodal}}) {
    std::vector<Pick> all;
    for (auto& v : *src) all.insert(all.end(), v.begin(), v.end());
    std::sort(all.begin(), all.end(),
              [](const Pick& a, const Pick& b) { return a.metric > b.metric; });
    absl::flat_hash_set<uint64_t> used;
    int rank = 0;
    for (const Pick& p : all) {
      if (rank == top) break;
      if (!used.insert(p.hash).second) continue;
      out += (first ? "" : ",") + RowJson(p, cat, ++rank);
      first = false;
      LOG(INFO) << cat << " #" << rank << " golden " << p.golden << " row "
                << p.row << " metric " << p.metric;
    }
  }
  char buf[256];
  std::snprintf(buf, sizeof buf,
                "],\"total\":%lld,\"unique\":%lld,\"n_heavy_gt20\":%lld,"
                "\"n_heavy_elig\":%lld,\"n_two_hump\":%lld,\"n_rows200\":%lld}",
                (long long)total, (long long)uniq, (long long)n_gt20,
                (long long)n_elig, (long long)n_two, (long long)n200);
  out += buf;
  std::FILE* f = std::fopen(absl::GetFlag(FLAGS_out_path).c_str(), "w");
  CHECK(f);
  std::fputs(out.c_str(), f);
  std::fclose(f);
  std::printf("%s\n", buf + 2);
  return 0;
}
