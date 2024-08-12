/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/math/types.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace momentum {

template <typename T>
struct TransformT {
  Quaternion<T> rotation;
  Vector3<T> translation;
  T scale;

  [[nodiscard]] static TransformT<T> makeRotation(const Quaternion<T>& rotation_in);
  [[nodiscard]] static TransformT<T> makeTranslation(const Vector3<T>& translation_in);
  [[nodiscard]] static TransformT<T> makeScale(const T& scale_in);

  TransformT() : rotation(Quaternion<T>::Identity()), translation(Vector3<T>::Zero()), scale(1) {
    // Empty
  }

  explicit TransformT(
      const Vector3<T>& translation_in,
      const Quaternion<T>& rotation_in,
      const T& scale_in)
      : rotation(rotation_in), translation(translation_in), scale(scale_in) {
    // Empty
  }

  [[nodiscard]] Affine3<T> matrix() const;

  [[nodiscard]] Vector3<T> transformPoint(const Vector3<T>& pt) const;
  [[nodiscard]] Vector3<T> rotate(const Vector3<T>& vec) const;

  [[nodiscard]] TransformT<T> inverse() const;

  template <typename T2>
  [[nodiscard]] TransformT<T2> cast() const {
    return TransformT<T2>(
        this->translation.template cast<T2>(),
        this->rotation.template cast<T2>(),
        static_cast<T2>(this->scale));
  }
};

template <typename T>
TransformT<T> operator*(const TransformT<T>& lhs, const TransformT<T>& rhs);

template <typename T>
using TransformListT =
    std::vector<TransformT<T>>; // structure describing a the state of all joints in a skeleton

using Transform = TransformT<float>;
using TransformList = TransformListT<float>;

} // namespace momentum
