#ifndef GTP_SERVICE_H_
#define GTP_SERVICE_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "cc/analysis/analysis.h"
#include "cc/game/loc.h"
#include "cc/game/move.h"
#include "cc/gtp/command.h"

namespace gtp {

class Service {
 public:
  virtual ~Service() = default;

  virtual Response<int> GtpProtocolVersion(std::optional<int> id) = 0;
  virtual Response<std::string> GtpName(std::optional<int> id) = 0;
  virtual Response<std::string> GtpVersion(std::optional<int> id) = 0;
  virtual Response<bool> GtpKnownCommand(std::optional<int> id,
                                         std::string cmd_name) = 0;
  virtual Response<std::string> GtpListCommands(std::optional<int> id) = 0;
  virtual Response<> GtpBoardSize(std::optional<int> id, int board_size) = 0;
  virtual Response<> GtpClearBoard(std::optional<int> id) = 0;
  virtual Response<> GtpKomi(std::optional<int> id, float komi) = 0;
  virtual Response<> GtpPlay(std::optional<int> id, game::Move move) = 0;
  virtual Response<game::Loc> GtpGenMove(std::optional<int> id,
                                         game::Color color) = 0;
  virtual Response<std::string> GtpPrintBoard(std::optional<int> id) = 0;
  virtual Response<game::Scores> GtpFinalScore(std::optional<int> id) = 0;

  // Analysis methods. Call these from separate threads.
  virtual Response<> GtpStartAnalysis(std::optional<int> id,
                                      game::Color color) = 0;
  virtual analysis::AnalysisSnapshot GtpAnalysisSnapshot(game::Color color) = 0;
  virtual void GtpStopAnalysis() = 0;
  virtual game::Loc GtpGenMoveAnalyze(game::Color color) = 0;

  // Private Commands.
  virtual Response<std::string> GtpPlayDbg(std::optional<int> id,
                                           game::Move move) = 0;
  virtual Response<std::string> GtpGenMoveDbg(std::optional<int> id,
                                              game::Color color) = 0;
  virtual Response<std::array<float, BOARD_LEN * BOARD_LEN>> GtpOwnership(
      std::optional<int> id) = 0;

  static absl::StatusOr<std::unique_ptr<Service>> CreateService(
      std::string model_path, int n, int k);

 protected:
  Service() = default;
};

}  // namespace gtp

#endif
