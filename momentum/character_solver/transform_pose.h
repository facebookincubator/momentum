/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character_solver/fwd.h>
#include <momentum/math/types.h>

namespace momentum {

/// Function which computes a new set of model parameters such that the character pose is a rigidly
/// transformed version of the original pose.  While there is technically a closed form solution
/// for any given skeleton, this is complicated in momentum because different
/// characters attach the rigid parameters to different joints, so a fully general solution uses IK.
/// However, getting it right requires dealing with local minima issues in Euler angles and other
/// challenges.  So centralizing this functionality in a single place is useful.
template <typename T>
std::vector<ModelParametersT<T>> transformPose(
    const Character& character,
    const std::vector<ModelParametersT<T>>& modelParameters,
    const std::vector<RigidTransform3<T>>& transforms);

} // namespace momentum
