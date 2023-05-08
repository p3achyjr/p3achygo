#ifndef __RECORDER_SGF_TREE_H__
#define __RECORDER_SGF_TREE_H__

#include <string>

#include "cc/game/game.h"
#include "cc/game/loc.h"
#include "cc/recorder/sgf_visitor.h"

namespace recorder {

/*
 * Classes defining an SGF Tree.
 */
class SgfVisitable {
 public:
  virtual ~SgfVisitable() = default;
  virtual void Accept(SgfVisitor* visitor) const = 0;

 protected:
  SgfVisitable() = default;
};

class SgfProp : public SgfVisitable {
 public:
  ~SgfProp() = default;
};

class SgfKomiProp final : public SgfProp {
 public:
  SgfKomiProp(float komi) : komi_(komi) {}
  void Accept(SgfVisitor* visitor) const override { visitor->Visit(this); }
  std::string tag() const { return kTag; }
  float komi() const { return komi_; }

 private:
  static constexpr char kTag[] = "KM";
  float komi_;
};

class SgfResultProp final : public SgfProp {
 public:
  SgfResultProp(game::Game::Result result) : result_(result) {}
  void Accept(SgfVisitor* visitor) const override { visitor->Visit(this); }
  std::string tag() const { return kTag; }
  game::Game::Result result() const { return result_; }

 private:
  static constexpr char kTag[] = "RE";
  game::Game::Result result_;
};

class SgfBPlayerProp final : public SgfProp {
 public:
  SgfBPlayerProp(std::string player) : player_(player) {}
  void Accept(SgfVisitor* visitor) const override { visitor->Visit(this); }
  std::string tag() const { return kTag; }
  std::string player() const { return player_; }

 private:
  static constexpr char kTag[] = "PB";
  std::string player_;
};

class SgfWPlayerProp final : public SgfProp {
 public:
  SgfWPlayerProp(std::string player) : player_(player) {}
  void Accept(SgfVisitor* visitor) const override { visitor->Visit(this); }
  std::string tag() const { return kTag; }
  std::string player() const { return player_; }

 private:
  static constexpr char kTag[] = "PW";
  std::string player_;
};

class SgfBMoveProp final : public SgfProp {
 public:
  SgfBMoveProp(game::Loc move) : move_(move) {}
  void Accept(SgfVisitor* visitor) const override { visitor->Visit(this); }
  std::string tag() const { return kTag; }
  game::Loc move() const { return move_; }

 private:
  static constexpr char kTag[] = "B";
  game::Loc move_;
};

class SgfWMoveProp final : public SgfProp {
 public:
  SgfWMoveProp(game::Loc move) : move_(move) {}
  void Accept(SgfVisitor* visitor) const override { visitor->Visit(this); }
  std::string tag() const { return kTag; }
  game::Loc move() const { return move_; }

 private:
  static constexpr char kTag[] = "W";
  game::Loc move_;
};

class SgfNode final : public SgfVisitable {
 public:
  void Accept(SgfVisitor* visitor) const override { visitor->Visit(this); }
  void AddProperty(std::unique_ptr<SgfProp> prop) {
    properties_.emplace_back(std::move(prop));
  }
  void AddChild(std::unique_ptr<SgfNode> child) {
    children_.emplace_back(std::move(child));
  }

  const std::vector<std::unique_ptr<SgfProp>>& properties() const {
    return properties_;
  }

  const std::vector<std::unique_ptr<SgfNode>>& children() const {
    return children_;
  }

  const std::unique_ptr<SgfProp>& property(int i) const {
    return properties_[i];
  }

  const std::unique_ptr<SgfNode>& child(int i) const { return children_[i]; }

 private:
  std::vector<std::unique_ptr<SgfProp>> properties_;
  std::vector<std::unique_ptr<SgfNode>> children_;
};

}  // namespace recorder

#endif
