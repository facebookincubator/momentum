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

  template <typename T2>
  explicit TransformT(const TransformT<T2>& other)
      : rotation(other.rotation.template cast<T>()),
        translation(other.translation.template cast<T>()),
        scale(other.scale) {
    // Empty
  }

  explicit TransformT(
      const Vector3<T>& translation_in,
      const Quaternion<T>& rotation_in = Quaternion<T>::Identity(),
      const T& scale_in = T(1))
      : rotation(rotation_in), translation(translation_in), scale(scale_in) {
    // Empty
  }

  explicit TransformT(
      Vector3<T>&& translation_in,
      Quaternion<T>&& rotation_in = Quaternion<T>::Identity(),
      T&& scale_in = T(1))
      : rotation(std::move(rotation_in)),
        translation(std::move(translation_in)),
        scale(std::move(scale_in)) {
    // Empty
  }

  explicit TransformT(const Matrix4<T>& other)
      : rotation(other.template topLeftCorner<3, 3>()),
        translation(other.template topRightCorner<3, 1>()) {
    scale = rotation.norm();
    rotation.coeffs() /= scale;
  }

  TransformT<T>& operator=(const Affine3<T>& other) {
    translation = other.translation();
    rotation = other.linear();
    scale = rotation.norm();
    rotation.coeffs() /= scale;
    return *this;
  }

  TransformT<T>& operator=(const Matrix4<T>& other) {
    translation = other.template topRightCorner<3, 1>();
    rotation = other.template topLeftCorner<3, 3>();
    scale = rotation.norm();
    rotation.coeffs() /= scale;
    return *this;
  }

  [[nodiscard]] TransformT<T> operator*(const TransformT<T>& other) const {
    // [ s_1*R_1 t_1 ] * [ s_2*R_2 t_2 ] = [ s_1*s_2*R_1*R_2  s_1*R_1*t_2 + t_1 ]
    // [     0    1  ]   [     0    1  ]   [        0                     1     ]
    const Vector3<T> trans = translation + rotation * (scale * other.translation);
    const Quaternion<T> rot = rotation * other.rotation;
    const T newScale = scale * other.scale;
    return TransformT<T>(std::move(trans), std::move(rot), std::move(newScale));
  }

  [[nodiscard]] Affine3<T> operator*(const Affine3<T>& other) const {
    Affine3<T> out = Affine3<T>::Identity();
    out.linear().noalias() = toLinear() * other.linear();
    out.translation().noalias() = rotation * (scale * other.translation()) + translation;
    return out;
  }

  [[nodiscard]] Vector3<T> operator*(const Vector3<T>& other) const {
    return transformPoint(other);
  }

  explicit operator Affine3<T>() const {
    return toAffine3();
  }

  [[nodiscard]] Affine3<T> toAffine3() const;

  /// Returns 4x4 matrix representation of the transform.
  [[nodiscard]] Matrix4<T> toMatrix() const {
    Matrix4<T> out = Matrix4<T>::Zero();
    out.template topLeftCorner<3, 3>().noalias() = rotation.toRotationMatrix() * scale;
    out.template topRightCorner<3, 1>() = translation;
    out(3, 3) = T(1);
    return out;
  }

  /// Returns 3x3 rotation matrix representation of the transform.
  [[nodiscard]] Matrix3<T> toRotationMatrix() const {
    return rotation.toRotationMatrix();
  }

  /// Returns the linear part (rotation * scale) of the transform.
  [[nodiscard]] Matrix3<T> toLinear() const {
    return toRotationMatrix() * scale;
  }

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
using TransformListT =
    std::vector<TransformT<T>>; // structure describing a the state of all joints in a skeleton

using Transform = TransformT<float>;
using TransformList = TransformListT<float>;

} // namespace momentum
