/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/joint.h>
#include <momentum/character/types.h>
#include <momentum/math/transform.h>
#include <momentum/math/types.h>

namespace momentum {

/// JointState provides all sorts of transformations computed on a joint.
/// Joint transform is :
///     WorldTransform = ParentWorldTransform * Tz * Ty * Tx * Rpre * R * S,
/// with R depending on rotation order (R = rz * ry * rx).
/// Each joint thus has 7 parameters: 3 translation, 3 rotation, and 1 scale.
template <typename T>
struct JointStateT {
  using Affine3 = Eigen::Transform<T, 3, Eigen::Affine>;

  /// Relative transformation from the parent joint to this joint, which is defined by the
  /// joint parameters
  TransformT<T> localTransform;

  /// Global transformation of this joint
  TransformT<T> transform;

  /// Columns contain the three translation axes for this joint in global space
  Eigen::Matrix3<T> translationAxis;

  /// Columns contain the three rotation axes for this joint in global space
  Eigen::Matrix3<T> rotationAxis;

  /// Indicate whether translationAxis and rotationAxis are up-to-date with other transformations
  bool derivDirty = true;

  /// Recursively update all the transformations from the root to leaves
  void set(
      const JointT<T>& joint,
      const JointVectorT<T>& parameters,
      const JointStateT<T>* parentState = nullptr,
      bool computeDeriv = true) noexcept;

  // get derivatives of joints with respect to a given reference point. This needs to be as fast as
  // possible as this is part of many inner loops.
  /// The derivative of a global vector ref wrt the rotation parameters of the global
  /// transformation.
  [[nodiscard]] Eigen::Vector3<T> getRotationDerivative(size_t index, const Eigen::Vector3<T>& ref)
      const;

  /// The derivative of any global vector wrt the translation parameters of the global
  /// transformation.
  [[nodiscard]] Eigen::Vector3<T> getTranslationDerivative(size_t index) const;

  /// The derivative of a global vector ref wrt the scaling parameter of the global transformation.
  [[nodiscard]] Eigen::Vector3<T> getScaleDerivative(const Eigen::Vector3<T>& ref) const noexcept;

  template <typename T2>
  void set(const JointStateT<T2>& rhs);

  /// Local rotation matrix
  [[nodiscard]] const Quaternion<T>& localRotation() const {
    return localTransform.rotation;
  }

  Quaternion<T>& localRotation() {
    return localTransform.rotation;
  }

  /// Local translation offset
  [[nodiscard]] const Vector3<T>& localTranslation() const {
    return localTransform.translation;
  }

  Vector3<T>& localTranslation() {
    return localTransform.translation;
  }

  /// Local scaling (affects descendants)
  [[nodiscard]] const T& localScale() const {
    return localTransform.scale;
  }

  T& localScale() {
    return localTransform.scale;
  }

  /// Rotation matrix from local to global space
  [[nodiscard]] const Quaternion<T>& rotation() const {
    return transform.rotation;
  }

  Quaternion<T>& rotation() {
    return transform.rotation;
  }

  /// Translation offset from local to global space
  [[nodiscard]] const Vector3<T>& translation() const {
    return transform.translation;
  }

  Vector3<T>& translation() {
    return transform.translation;
  }

  /// Return the X component of the global transform
  [[nodiscard]] const T& x() const {
    return transform.translation.x();
  }

  T& x() {
    return transform.translation.x();
  }

  /// Return the Y component of the global transform
  [[nodiscard]] const T& y() const {
    return transform.translation.y();
  }

  T& y() {
    return transform.translation.y();
  }

  /// Return the Z component of the global transform
  [[nodiscard]] const T& z() const {
    return transform.translation.z();
  }

  T& z() {
    return transform.translation.z();
  }

  [[nodiscard]] const T& quatW() const {
    return transform.rotation.w();
  }

  T& quatW() {
    return transform.rotation.w();
  }

  [[nodiscard]] const T& quatX() const {
    return transform.rotation.x();
  }

  T& quatX() {
    return transform.rotation.x();
  }

  [[nodiscard]] const T& quatY() const {
    return transform.rotation.y();
  }

  T& quatY() {
    return transform.rotation.y();
  }

  [[nodiscard]] const T& quatZ() const {
    return transform.rotation.z();
  }

  T& quatZ() {
    return transform.rotation.z();
  }

  /// Scaling from local to global space
  [[nodiscard]] const T& scale() const {
    return transform.scale;
  }

  T& scale() {
    return transform.scale;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// Structure describing the state of all joints in a skeleton
template <typename T>
using JointStateListT = std::vector<JointStateT<T>>;

using JointState = JointStateT<float>;
using JointStateList = JointStateListT<float>;

} // namespace momentum
