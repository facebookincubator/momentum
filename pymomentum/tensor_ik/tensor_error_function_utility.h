// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <Eigen/Core>

namespace pymomentum {

template <typename T>
T extractScalar(Eigen::Ref<const Eigen::VectorX<T>> vec, Eigen::Index index) {
  assert(vec.size() != 0);
  return vec(index);
}

template <typename T>
T extractScalar(
    Eigen::Ref<const Eigen::VectorX<T>> vec,
    Eigen::Index index,
    const T& defaultValue) {
  if (vec.size() == 0) {
    return defaultValue;
  }

  return extractScalar<T>(vec, index);
}

// Extract a sub-vector of size `Size` from an Eigen vector-like. The vector is
// made by several sub-vectors of the same size concatenating together.
template <typename T, Eigen::Index Size>
Eigen::Matrix<T, Size, 1> extractVector(
    Eigen::Ref<const Eigen::VectorX<T>> vec,
    Eigen::Index index) {
  assert(vec.size() != 0);
  return vec.template segment<Size>(index * Size);
}

// Extract a sub-vector of size `Size` from an Eigen vector-like. Return a
// default value if the vector is empty. The vector is made by several
// sub-vectors of the same size concatenating together.
template <typename T, Eigen::Index Size, typename Derived>
Eigen::Matrix<T, Size, 1> extractVector(
    Eigen::Ref<const Eigen::VectorX<T>> vec,
    Eigen::Index index,
    const Eigen::EigenBase<Derived>& defaultValue) {
  if (vec.size() == 0) {
    return defaultValue;
  }

  return extractVector<T, Size>(vec, index);
}

template <typename T>
Eigen::Quaternion<T> extractQuaternion(
    Eigen::Ref<const Eigen::VectorX<T>> vec,
    Eigen::Index index) {
  if (vec.size() == 0) {
    return Eigen::Quaternion<T>::Identity();
  }

  return Eigen::Quaternion<T>(vec.template segment<4>(4 * index)).normalized();
}

} // namespace pymomentum
