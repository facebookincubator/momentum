/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/common/memory.h>

namespace momentum {

MOMENTUM_FWD_DECLARE_STRUCT(BlendShape);
MOMENTUM_FWD_DECLARE_STRUCT(BlendShapeBase);
MOMENTUM_FWD_DECLARE_STRUCT(PoseShape);
MOMENTUM_FWD_DECLARE_STRUCT(SkinWeights);

MOMENTUM_FWD_DECLARE_TEMPLATE_STRUCT(Character);
MOMENTUM_FWD_DECLARE_TEMPLATE_STRUCT(CharacterState);
MOMENTUM_FWD_DECLARE_TEMPLATE_STRUCT(CollisionGeometryState);
MOMENTUM_FWD_DECLARE_TEMPLATE_STRUCT(Joint);
MOMENTUM_FWD_DECLARE_TEMPLATE_STRUCT(JointState);
MOMENTUM_FWD_DECLARE_TEMPLATE_STRUCT(ParameterTransform);
MOMENTUM_FWD_DECLARE_TEMPLATE_STRUCT(Skeleton);
MOMENTUM_FWD_DECLARE_TEMPLATE_STRUCT(SkeletonState);
MOMENTUM_FWD_DECLARE_TEMPLATE_STRUCT(TaperedCapsule);

} // namespace momentum
