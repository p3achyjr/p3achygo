#ifndef __SGF_SGF_VISITOR_H_
#define __SGF_SGF_VISITOR_H_

namespace sgf {
class SgfSizeProp;
class SgfKomiProp;
class SgfResultProp;
class SgfHandicapProp;
class SgfCommentProp;
class SgfBPlayerProp;
class SgfWPlayerProp;
class SgfBMoveProp;
class SgfWMoveProp;
class SgfNode;

/*
 * Abstract base class for SGF visitors.
 */
class SgfVisitor {
 public:
  virtual void Visit(const SgfSizeProp* prop) = 0;
  virtual void Visit(const SgfKomiProp* prop) = 0;
  virtual void Visit(const SgfResultProp* prop) = 0;
  virtual void Visit(const SgfHandicapProp* prop) = 0;
  virtual void Visit(const SgfCommentProp* prop) = 0;
  virtual void Visit(const SgfBPlayerProp* prop) = 0;
  virtual void Visit(const SgfWPlayerProp* prop) = 0;
  virtual void Visit(const SgfBMoveProp* prop) = 0;
  virtual void Visit(const SgfWMoveProp* prop) = 0;
  virtual void Visit(const SgfNode* node) = 0;
};
}  // namespace sgf

#endif
