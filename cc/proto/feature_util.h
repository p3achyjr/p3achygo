#ifndef CC_PROTO_FEATURE_UTIL_H_
#define CC_PROTO_FEATURE_UTIL_H_

// Simplified feature utility functions for accessing TensorFlow Example protos.
// This is a minimal port of tensorflow/core/example/feature_util.h.

#include <string>

#include "example.pb.h"
#include "feature.pb.h"

namespace tensorflow {

// Returns the Features from an Example.
inline const Features& GetFeatures(const Example& example) {
  return example.features();
}

inline Features* GetFeatures(Example* example) {
  return example->mutable_features();
}

// Traits to determine the correct container type for feature values.
template <typename T>
struct FeatureValueTraits {
  using container_type = google::protobuf::RepeatedField<T>;
};

template <>
struct FeatureValueTraits<std::string> {
  using container_type = google::protobuf::RepeatedPtrField<std::string>;
};

// Template declaration for getting feature values from a Feature.
template <typename FeatureType>
const typename FeatureValueTraits<FeatureType>::container_type& GetFeatureValues(
    const Feature& feature);

// Specialization for int64_t
template <>
inline const google::protobuf::RepeatedField<int64_t>& GetFeatureValues<int64_t>(
    const Feature& feature) {
  return feature.int64_list().value();
}

// Specialization for float
template <>
inline const google::protobuf::RepeatedField<float>& GetFeatureValues<float>(
    const Feature& feature) {
  return feature.float_list().value();
}

// Specialization for std::string (bytes_list)
template <>
inline const google::protobuf::RepeatedPtrField<std::string>&
GetFeatureValues<std::string>(const Feature& feature) {
  return feature.bytes_list().value();
}

// Returns a read-only repeated field corresponding to a feature with the
// specified name and FeatureType.
template <typename FeatureType>
const auto& GetFeatureValues(const std::string& key, const Example& example) {
  return GetFeatureValues<FeatureType>(
      GetFeatures(example).feature().at(key));
}

}  // namespace tensorflow

#endif  // CC_PROTO_FEATURE_UTIL_H_
