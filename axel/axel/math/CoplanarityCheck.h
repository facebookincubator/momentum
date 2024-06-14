/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <Eigen/Core>
#include <gsl/span>

namespace axel {

int solveP3(gsl::span<double, 3> x, double a, double b, double c);
int solveP2(gsl::span<double, 2> x, double a, double b, double c);

/* Find times when the four points with given velocities are coplanar in 3D space */
int timesCoplanar(
    gsl::span<double, 3> t,
    const Eigen::Vector3d& x1,
    const Eigen::Vector3d& x2,
    const Eigen::Vector3d& x3,
    const Eigen::Vector3d& x4,
    const Eigen::Vector3d& v1,
    const Eigen::Vector3d& v2,
    const Eigen::Vector3d& v3,
    const Eigen::Vector3d& v4);

} // namespace axel
