/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character/skeleton.h"

#include "momentum/character/skeleton_state.h"
#include "momentum/character/types.h"
#include "momentum/common/checks.h"

namespace momentum {

template <typename T>
SkeletonT<T>::SkeletonT(JointList joints_in) : joints(std::move(joints_in)) {
  // Ensure some invariants of the skeleton:
  for (size_t iJoint = 0; iJoint < joints.size(); ++iJoint) {
    MT_CHECK(joints[iJoint].parent == kInvalidIndex || joints[iJoint].parent < iJoint);
  }
}

template <typename T>
bool SkeletonT<T>::isAncestor(size_t jointId, size_t ancestorJointId) const {
  MT_CHECK(jointId < joints.size(), "{} vs {}", jointId, joints.size());
  MT_CHECK(ancestorJointId < joints.size(), "{} vs {}", ancestorJointId, joints.size());
  while (jointId != kInvalidIndex) {
    if (jointId == ancestorJointId) {
      return true;
    }
    jointId = joints[jointId].parent;
  }

  return false;
}

template <typename T>
size_t SkeletonT<T>::commonAncestor(size_t joint1, size_t joint2) const {
  MT_CHECK(joint1 == kInvalidIndex || joint1 < joints.size());
  MT_CHECK(joint2 == kInvalidIndex || joint2 < joints.size());

  while (joint1 != kInvalidIndex && joint2 != kInvalidIndex && joint1 != joint2) {
    if (joint1 < joint2) {
      joint2 = joints[joint2].parent;
    } else {
      joint1 = joints[joint1].parent;
    }
  }

  return joint1;
}

template struct SkeletonT<float>;
template struct SkeletonT<double>;

} // namespace momentum
