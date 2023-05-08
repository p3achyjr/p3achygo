#ifndef __RECORDER_SGF_VISITOR_H_
#define __RECORDER_SGF_VISITOR_H_

namespace recorder {
class SgfKomiProp;
class SgfResultProp;
class SgfBPlayerProp;
class SgfWPlayerProp;
class SgfBMoveProp;
class SgfWMoveProp;
class SgfNode;

class SgfVisitor {
 public:
  virtual void Visit(const SgfKomiProp* prop) = 0;
  virtual void Visit(const SgfResultProp* prop) = 0;
  virtual void Visit(const SgfBPlayerProp* prop) = 0;
  virtual void Visit(const SgfWPlayerProp* prop) = 0;
  virtual void Visit(const SgfBMoveProp* prop) = 0;
  virtual void Visit(const SgfWMoveProp* prop) = 0;
  virtual void Visit(const SgfNode* prop) = 0;
};
}  // namespace recorder

#endif
