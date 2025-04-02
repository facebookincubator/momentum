/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <Eigen/Core>

#include "axel/common/VectorizationTypes.h"

namespace axel {

template <typename S>
bool projectOnTriangle(
    const Eigen::Vector3<S>& p,
    const Eigen::Vector3<S>& a,
    const Eigen::Vector3<S>& b,
    const Eigen::Vector3<S>& c,
    Eigen::Vector3<S>& q,
    Eigen::Vector3<S>* barycentric = nullptr);

template <typename S>
WideMask<WideScalar<S>> projectOnTriangle(
    const WideVec3<S>& p,
    const WideVec3<S>& a,
    const WideVec3<S>& b,
    const WideVec3<S>& c,
    WideVec3<S>& q,
    WideVec3<S>* barycentric = nullptr);

} // namespace axel
