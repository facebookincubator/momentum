/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <Eigen/Core>

namespace axel {

/*
    Compute the distance between the closest points on two edges
    Edge 1 : p1 - q1
    Edge 2 : p2 - q2
    Result: closest points will be
    p1 + s(q1-p1) on edge 1
    p2 + t(q2-p2) on edge 2
*/
bool distanceEdgeEdge(
    const Eigen::Vector3d& p1,
    const Eigen::Vector3d& q1,
    const Eigen::Vector3d& p2,
    const Eigen::Vector3d& q2,
    double& s,
    double& t,
    double& distance);

} // namespace axel
