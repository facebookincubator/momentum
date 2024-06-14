/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/common/memory.h>

namespace momentum {

MOMENTUM_FWD_DECLARE_TEMPLATE_CLASS(SolverFunction);

MOMENTUM_FWD_DECLARE_STRUCT(SolverOptions);
MOMENTUM_FWD_DECLARE_STRUCT(GaussNewtonSolverOptions);

MOMENTUM_FWD_DECLARE_TEMPLATE_CLASS(Solver);
MOMENTUM_FWD_DECLARE_TEMPLATE_CLASS(GaussNewtonSolver);
MOMENTUM_FWD_DECLARE_TEMPLATE_CLASS(GradientDescentSolver);
MOMENTUM_FWD_DECLARE_TEMPLATE_CLASS(SubsetGaussNewtonSolver);

} // namespace momentum
