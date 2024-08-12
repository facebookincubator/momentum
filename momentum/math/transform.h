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
  Eigen::Quaternion<T> rotation;
  Eigen::Vector3<T> translation;
  T scale;

  using Affine3 = Eigen::Transform<T, 3, Eigen::Affine>;

  TransformT()
      : rotation(Eigen::Quaternion<T>::Identity()),
        translation(Eigen::Vector3<T>::Zero()),
        scale(1) {}

  explicit TransformT(
      const Eigen::Vector3<T>& translation_in,
      const Eigen::Quaternion<T>& rotation_in,
      const T scale_in) // NOLINT(facebook-hte-ConstantArgumentPassByValue)
      : rotation(rotation_in), translation(translation_in), scale(scale_in) {}

  static TransformT<T> makeRotation(const Eigen::Quaternion<T>& rotation_in);
  static TransformT<T> makeTranslation(const Eigen::Vector3<T>& translation_in);
  // NOLINTNEXTLINE(facebook-hte-ConstantArgumentPassByValue)
  static TransformT<T> makeScale(T scale_in);

  Affine3 matrix() const;

  Eigen::Vector3<T> transformPoint(const Eigen::Vector3<T>& pt) const;
  Eigen::Vector3<T> rotate(const Eigen::Vector3<T>& vec) const;

  TransformT<T> inverse() const;

  template <typename T2>
  TransformT<T2> cast() const {
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
