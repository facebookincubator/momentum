/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/math/types.h>

namespace momentum {

// base skinning class
struct PoseShape {
  size_t baseJoint;
  Quaternionf baseRot;
  std::vector<size_t> jointMap;
  VectorXf baseShape;
  MatrixXf shapeVectors;

  std::vector<Vector3f> compute(const SkeletonState& state) const;

  inline bool isApprox(const PoseShape& poseShape) const {
    return (
        (baseJoint == poseShape.baseJoint) && baseRot.isApprox(poseShape.baseRot) &&
        (jointMap == poseShape.jointMap) && baseShape.isApprox(poseShape.baseShape) &&
        shapeVectors.isApprox(poseShape.shapeVectors));
  }
};

} // namespace momentum
