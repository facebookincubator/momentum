/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/common/memory.h>

namespace momentum {

MOMENTUM_FWD_DECLARE_STRUCT(MultiposeSolverOptions);
MOMENTUM_FWD_DECLARE_STRUCT(SequenceSolverOptions);

MOMENTUM_FWD_DECLARE_TEMPLATE_CLASS(SequenceErrorFunction);
MOMENTUM_FWD_DECLARE_TEMPLATE_CLASS(ModelParametersSequenceErrorFunction);
MOMENTUM_FWD_DECLARE_TEMPLATE_CLASS(MultiposeSolver);
MOMENTUM_FWD_DECLARE_TEMPLATE_CLASS(MultiposeSolverFunction);
MOMENTUM_FWD_DECLARE_TEMPLATE_CLASS(SequenceSolver);
MOMENTUM_FWD_DECLARE_TEMPLATE_CLASS(SequenceSolverFunction);
MOMENTUM_FWD_DECLARE_TEMPLATE_CLASS(StateSequenceErrorFunction);

} // namespace momentum
