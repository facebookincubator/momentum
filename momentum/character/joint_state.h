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

  // local joint transformation as defined by parameters
  /// Local rotation matrix
  Eigen::Quaternion<T> localRotation; // TODO: Remove
  /// Local translation offset
  Eigen::Vector3<T> localTranslation; // TODO: Remove
  /// Local scaling (affects descendants)
  T localScale; // TODO: Remove

  // joint transformation from local to global space
  /// Rotation matrix from local to global space
  Eigen::Quaternion<T> rotation; // TODO: Remove
  /// Translation offset from local to global space
  Eigen::Vector3<T> translation; // TODO: Remove
  /// Scaling from local to global space
  T scale; // TODO: Remove

  /// Local to global complete matrix
  Affine3 transformation; // TODO: Remove

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

  // TODO: Remove
  explicit JointStateT(
      const Eigen::Quaternion<T>& localRotation = Eigen::Quaternion<T>(),
      const Eigen::Vector3<T>& localTranslation = Eigen::Vector3<T>(),
      const T& localScale = T(),
      const Eigen::Quaternion<T>& rotation = Eigen::Quaternion<T>(),
      const Eigen::Vector3<T>& translation = Eigen::Vector3<T>(),
      const T& scale = T(),
      const Affine3& transformation = Affine3(),
      const Eigen::Matrix3<T>& translationAxis = Eigen::Matrix3<T>(),
      const Eigen::Matrix3<T>& rotationAxis = Eigen::Matrix3<T>())
      : localRotation(localRotation),
        localTranslation(localTranslation),
        localScale(localScale),
        rotation(rotation),
        translation(translation),
        scale(scale),
        transformation(transformation),
        translationAxis(translationAxis),
        rotationAxis(rotationAxis) {
    localTransform.rotation = localRotation;
    localTransform.translation = localTranslation;
    localTransform.scale = localScale;
    transform.rotation = rotation;
    transform.translation = translation;
    transform.scale = scale;
  }

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

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// Structure describing the state of all joints in a skeleton
template <typename T>
using JointStateListT = std::vector<JointStateT<T>>;

using JointState = JointStateT<float>;
using JointStateList = JointStateListT<float>;

} // namespace momentum
