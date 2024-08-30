/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

#include <momentum/character/joint.h>
#include <momentum/character/parameter_transform.h>
#include <momentum/character/types.h>
#include <momentum/common/exception.h>

namespace momentum {

/// The skeletal structure of a momentum Character.
template <typename T>
struct SkeletonT {
  /// The list of joints in this skeleton.
  JointList joints;

  explicit SkeletonT(JointList joints);
  SkeletonT() = default;

  /// Look up the index of a joint from its name.
  [[nodiscard]] size_t getJointIdByName(const std::string_view name) const {
    for (size_t i = 0; i < joints.size(); i++)
      if (joints[i].name == name)
        return i;
    return kInvalidIndex;
  }

  /// Get the names of all the joints in this skeleton.
  [[nodiscard]] std::vector<std::string> getJointNames() const {
    std::vector<std::string> result;
    result.reserve(joints.size());
    for (const auto& j : joints)
      result.emplace_back(j.name);
    return result;
  }

  /// Get the list of indices of the direct children of a joint or all its descendants.
  [[nodiscard]] std::vector<size_t> getChildrenJoints(
      const size_t jointId,
      const bool recursive = true) const {
    MT_THROW_IF_T(
        jointId >= joints.size(),
        std::out_of_range,
        "Out of bounds getChildrenJoints query. Requested index: {}. Number of joints: {}",
        jointId,
        joints.size());

    std::vector<size_t> childrenJoints;
    std::vector<int> jointDistance(joints.size(), -1);
    jointDistance[jointId] = 0;

    // traversal assuming parentJoint < childJoint
    for (size_t jointIter = jointId + 1; jointIter < joints.size(); ++jointIter) {
      const auto jointParent = joints[jointIter].parent;
      const int distParent = (jointParent == kInvalidIndex) ? -1 : jointDistance[jointParent];
      if ((recursive && distParent >= 0) || (!recursive && distParent == 0)) {
        jointDistance[jointIter] = distParent + 1;
        childrenJoints.push_back(jointIter);
      }
    }

    return childrenJoints;
  }

  /// Check whether the two input joints lie on the same branch of the hierarchy.
  /// Returns true if ancestorJointId is an ancestor of jointId; that is,
  /// if jointId is in the tree rooted at ancestorJointId.
  /// Note that a joint is considered to be its own ancestor; that is,
  /// isAncestor(id, id) returns true.
  [[nodiscard]] bool isAncestor(size_t jointId, size_t ancestorJointId) const;

  [[nodiscard]] size_t commonAncestor(size_t joint1, size_t joint2) const;

  /// Casts the current skeleton to another scalar type.
  template <typename U>
  [[nodiscard]] SkeletonT<U> cast() const {
    if constexpr (std::is_same_v<T, U>) {
      return *this;
    } else {
      SkeletonT<U> newSkeleton;
      newSkeleton.joints = ::momentum::cast<U>(joints);
      return newSkeleton;
    }
  }
};

} // namespace momentum
