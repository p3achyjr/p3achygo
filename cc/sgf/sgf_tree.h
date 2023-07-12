#ifndef __SGF_SGF_TREE_H__
#define __SGF_SGF_TREE_H__

#include <string>

#include "cc/game/game.h"
#include "cc/game/loc.h"
#include "cc/sgf/sgf_visitor.h"

namespace sgf {

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

class SgfSizeProp final : public SgfProp {
 public:
  SgfSizeProp(int size) : size_(size) {}
  void Accept(SgfVisitor* visitor) const override { visitor->Visit(this); }
  std::string tag() const { return kTag; }
  int size() const { return size_; }

  static constexpr char kTag[] = "SZ";

 private:
  int size_;
};

class SgfKomiProp final : public SgfProp {
 public:
  SgfKomiProp(float komi) : komi_(komi) {}
  void Accept(SgfVisitor* visitor) const override { visitor->Visit(this); }
  std::string tag() const { return kTag; }
  float komi() const { return komi_; }

  static constexpr char kTag[] = "KM";

 private:
  float komi_;
};

class SgfResultProp final : public SgfProp {
 public:
  SgfResultProp(game::Game::Result result) : result_(result) {}
  void Accept(SgfVisitor* visitor) const override { visitor->Visit(this); }
  std::string tag() const { return kTag; }
  game::Game::Result result() const { return result_; }

  static constexpr char kTag[] = "RE";

 private:
  game::Game::Result result_;
};

class SgfHandicapProp final : public SgfProp {
 public:
  SgfHandicapProp(int handicap) : handicap_(handicap) {}
  void Accept(SgfVisitor* visitor) const override { visitor->Visit(this); }
  std::string tag() const { return kTag; }
  int handicap() const { return handicap_; }

  static constexpr char kTag[] = "HA";

 private:
  int handicap_;
};

class SgfCommentProp final : public SgfProp {
 public:
  SgfCommentProp(std::string comment) : comment_(comment) {}
  void Accept(SgfVisitor* visitor) const override { visitor->Visit(this); }
  std::string tag() const { return kTag; }
  std::string comment() const { return comment_; }

  static constexpr char kTag[] = "C";

 private:
  std::string comment_;
};

class SgfBPlayerProp final : public SgfProp {
 public:
  SgfBPlayerProp(std::string player) : player_(player) {}
  void Accept(SgfVisitor* visitor) const override { visitor->Visit(this); }
  std::string tag() const { return kTag; }
  std::string player() const { return player_; }

  static constexpr char kTag[] = "PB";

 private:
  std::string player_;
};

class SgfWPlayerProp final : public SgfProp {
 public:
  SgfWPlayerProp(std::string player) : player_(player) {}
  void Accept(SgfVisitor* visitor) const override { visitor->Visit(this); }
  std::string tag() const { return kTag; }
  std::string player() const { return player_; }

  static constexpr char kTag[] = "PW";

 private:
  std::string player_;
};

class SgfBMoveProp final : public SgfProp {
 public:
  SgfBMoveProp(game::Loc move) : move_(move) {}
  void Accept(SgfVisitor* visitor) const override { visitor->Visit(this); }
  std::string tag() const { return kTag; }
  game::Loc move() const { return move_; }

  static constexpr char kTag[] = "B";

 private:
  game::Loc move_;
};

class SgfWMoveProp final : public SgfProp {
 public:
  SgfWMoveProp(game::Loc move) : move_(move) {}
  void Accept(SgfVisitor* visitor) const override { visitor->Visit(this); }
  std::string tag() const { return kTag; }
  game::Loc move() const { return move_; }

  static constexpr char kTag[] = "W";

 private:
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

}  // namespace sgf

#endif
