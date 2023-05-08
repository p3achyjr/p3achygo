#ifndef __RECORDER_SGF_SERIALIZER_H_
#define __RECORDER_SGF_SERIALIZER_H_

#include "cc/recorder/sgf_tree.h"
#include "cc/recorder/sgf_visitor.h"

namespace recorder {

class SgfSerializer final : public SgfVisitor {
 public:
  SgfSerializer() = default;
  ~SgfSerializer() = default;

  // Disable Copy and Move.
  SgfSerializer(SgfSerializer const&) = delete;
  SgfSerializer& operator=(SgfSerializer const&) = delete;
  SgfSerializer(SgfSerializer&&) = delete;
  SgfSerializer& operator=(SgfSerializer&&) = delete;

  std::string Serialize(const SgfNode* node);

  // SgfVisitor Impl.
  void Visit(const SgfKomiProp* property) override;
  void Visit(const SgfResultProp* property) override;
  void Visit(const SgfBPlayerProp* property) override;
  void Visit(const SgfWPlayerProp* property) override;
  void Visit(const SgfBMoveProp* property) override;
  void Visit(const SgfWMoveProp* property) override;
  void Visit(const SgfNode* property) override;

 private:
  void VisitNode(const SgfNode* node, bool is_root);
  std::string sgf_;
};
}  // namespace recorder

#endif
