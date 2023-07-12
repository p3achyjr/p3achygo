#ifndef __SGF_SGF_SERIALIZER_H_
#define __SGF_SGF_SERIALIZER_H_

#include "cc/sgf/sgf_tree.h"
#include "cc/sgf/sgf_visitor.h"

namespace sgf {

/*
 * Class for serializing an SGF Tree.
 */
class SgfSerializer final : public sgf::SgfVisitor {
 public:
  SgfSerializer() = default;
  ~SgfSerializer() = default;

  // Disable Copy and Move.
  SgfSerializer(SgfSerializer const&) = delete;
  SgfSerializer& operator=(SgfSerializer const&) = delete;
  SgfSerializer(SgfSerializer&&) = delete;
  SgfSerializer& operator=(SgfSerializer&&) = delete;

  std::string Serialize(const sgf::SgfNode* node);

  // sgf::SgfVisitor Impl.
  void Visit(const sgf::SgfSizeProp* property) override;
  void Visit(const sgf::SgfKomiProp* property) override;
  void Visit(const sgf::SgfResultProp* property) override;
  void Visit(const sgf::SgfHandicapProp* property) override;
  void Visit(const sgf::SgfCommentProp* prop) override;
  void Visit(const sgf::SgfBPlayerProp* property) override;
  void Visit(const sgf::SgfWPlayerProp* property) override;
  void Visit(const sgf::SgfBMoveProp* property) override;
  void Visit(const sgf::SgfWMoveProp* property) override;
  void Visit(const sgf::SgfNode* node) override;

 private:
  void VisitNode(const sgf::SgfNode* node, bool is_root);
  std::string sgf_;
};
}  // namespace sgf

#endif
