/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/common/fwd.h>

namespace momentum {

MOMENTUM_FWD_DECLARE_TEMPLATE_STRUCT(Mesh);
MOMENTUM_FWD_DECLARE_TEMPLATE_STRUCT(Mppca);
MOMENTUM_FWD_DECLARE_TEMPLATE_STRUCT(Transform);

MOMENTUM_FWD_DECLARE_TEMPLATE_CLASS(LowRankCovarianceMatrix);
MOMENTUM_FWD_DECLARE_TEMPLATE_CLASS(GeneralizedLoss);
MOMENTUM_FWD_DECLARE_TEMPLATE_CLASS(SimdGeneralizedLoss);

} // namespace momentum
