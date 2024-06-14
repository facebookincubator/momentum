/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/common/memory.h>

namespace momentum {

MOMENTUM_FWD_DECLARE_TEMPLATE_STRUCT(ErrorFunctionDerivatives);
MOMENTUM_FWD_DECLARE_TEMPLATE_STRUCT(OrientationConstraint);
MOMENTUM_FWD_DECLARE_TEMPLATE_STRUCT(OrientationConstraintState);
MOMENTUM_FWD_DECLARE_TEMPLATE_STRUCT(PositionConstraint);
MOMENTUM_FWD_DECLARE_TEMPLATE_STRUCT(PositionConstraintState);

MOMENTUM_FWD_DECLARE_TEMPLATE_CLASS(FullyDifferentiableMotionErrorFunction);
MOMENTUM_FWD_DECLARE_TEMPLATE_CLASS(FullyDifferentiableOrientationErrorFunction);
MOMENTUM_FWD_DECLARE_TEMPLATE_CLASS(FullyDifferentiablePosePriorErrorFunction);
MOMENTUM_FWD_DECLARE_TEMPLATE_CLASS(FullyDifferentiablePositionErrorFunction);
MOMENTUM_FWD_DECLARE_TEMPLATE_CLASS(FullyDifferentiableSkeletonErrorFunction);
MOMENTUM_FWD_DECLARE_TEMPLATE_CLASS(FullyDifferentiableStateErrorFunction);
MOMENTUM_FWD_DECLARE_TEMPLATE_CLASS(UnionErrorFunction);

} // namespace momentum
