/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/joint.h>
#include <momentum/character/types.h>
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

  // local joint transformation as defined by parameters
  /// Local rotation matrix
  Eigen::Quaternion<T> localRotation;
  /// Local translation offset
  Eigen::Vector3<T> localTranslation;
  /// Local scaling (affects descendants)
  T localScale;

  // joint transformation from local to global space
  /// Rotation matrix from local to global space
  Eigen::Quaternion<T> rotation;
  /// Translation offset from local to global space
  Eigen::Vector3<T> translation;
  /// Scaling from local to global space
  T scale;

  /// Local to global complete matrix
  Affine3 transformation;

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

  /// An Affine3 transformation from the local space to the parent space
  [[nodiscard]] AffineTransform3<T> localToParentXF() const {
    return createAffineTransform3(localTranslation, localRotation, localScale);
  }

  /// An Affine3 transformation from the local space to the world space
  [[nodiscard]] AffineTransform3<T> localToWorldXF() const {
    return createAffineTransform3(translation, rotation, scale);
  }

  template <typename T2>
  void set(const JointStateT<T2>& rhs);

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// Structure describing the state of all joints in a skeleton
template <typename T>
using JointStateListT = std::vector<JointStateT<T>>;

using JointState = JointStateT<float>;
using JointStateList = JointStateListT<float>;

} // namespace momentum
