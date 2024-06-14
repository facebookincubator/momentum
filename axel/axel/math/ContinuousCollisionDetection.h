/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <Eigen/Core>

namespace axel {

/**
 * Performs continuous collision detection between two moving edges.
 * First edge is defined by points (x1, x2) with velocities (v1, v2).
 * Second edge is defined by points (x3, x4) with velocities (v3, v4).
 * Returns true if the two edges have an interection within the given time delta `dt`.
 * Additional check is performed when detectionFrequency > 1, which compares averaged
 * trajectory distances multiple frames in the future (with time delta = dt * detectionFrequency).
 */
bool ccdEdgeEdge(
    const Eigen::Vector3d& x1,
    const Eigen::Vector3d& x2,
    const Eigen::Vector3d& x3,
    const Eigen::Vector3d& x4,
    const Eigen::Vector3d& v1,
    const Eigen::Vector3d& v2,
    const Eigen::Vector3d& v3,
    const Eigen::Vector3d& v4,
    double distanceThreshold,
    double dt);

// Note: Vertex is represented by p4 with velocity v4, while the triangle is defined by
// (x1, x2, x3) with velocities (v1, v2, v3).
bool ccdVertexTriangle(
    const Eigen::Vector3d& x1,
    const Eigen::Vector3d& x2,
    const Eigen::Vector3d& x3,
    const Eigen::Vector3d& x4,
    const Eigen::Vector3d& v1,
    const Eigen::Vector3d& v2,
    const Eigen::Vector3d& v3,
    const Eigen::Vector3d& v4,
    double distanceThreshold,
    double dt);

} // namespace axel
