/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "axel/math/EdgeEdgeDistance.h"

namespace axel {

bool distanceEdgeEdge(
    const Eigen::Vector3d& p1,
    const Eigen::Vector3d& q1,
    const Eigen::Vector3d& p2,
    const Eigen::Vector3d& q2,
    double& s,
    double& t,
    double& distance) {
  const double epsilon = 10e-5;
  // Compute closest points C1 and C2 of S1(s) = P1 + s*(Q1 - P1) and
  // S2(t) = P2 = t*(Q2 - P2)
  const Eigen::Vector3d d1 = q1 - p1; // Direction vector segment S1
  const Eigen::Vector3d d2 = q2 - p2; // Direction vector segment S2
  const Eigen::Vector3d r = p1 - p2;
  const double a = d1.dot(d1); // Squared distance length segment S1, always nonnegative
  const double e = d2.dot(d2); // Squared distance length segment S2, always nonnegative
  const double f = d2.dot(r);
  const double c = d1.dot(r);

  // Check if either or both segments degenerate into points
  if (a <= epsilon && e <= epsilon) {
    // case 1
    s = 0.0;
    t = 0.0;
    distance = (p1 - p2).norm();
    return false;
  }

  if (a <= epsilon) {
    // case 2
    // first segment degenerates into a point
    s = 0.0;
    t = f / e;
    t = std::clamp(t, 0.0, 1.0);
  } else {
    //        const double c = d1.dot(r);
    if (e <= epsilon) {
      // case 3
      // Second segment degenerates into a point
      t = 0.0;
      s = std::clamp(-c / a, 0.0, 1.0);
    } else {
      // case 4
      // The general nondegenerate case starts here
      const double b = d1.dot(d2);
      const double denom =
          a * e - b * b; // Always nonnegative
                         // If segments not parallel, compute closest point on L1 to L2 and
                         // clamp to segment S1. Else pick arbitrary s (here 0.0)
      if (std::abs(denom) > epsilon) {
        s = std::clamp((b * f - c * e) / denom, 0.0, 1.0);
      } else {
        s = 0.0;
      }
      // Compute point on L2 closest to S1(s)
      t = (b * s + f) / e;

      // If t in [0, 1] done. Else clamp t, recompute s for the new value of t
      // and clamp s
      if (t < 0.0) {
        t = 0.0;
        s = std::clamp(-c / a, 0.0, 1.0);
      } else if (t > 1.0) {
        t = 1.0;
        s = std::clamp((b - c) / a, 0.0, 1.0);
      }
    }
  }

  const Eigen::Vector3d c1 = p1 + d1 * s;
  const Eigen::Vector3d c2 = p2 + d2 * t;
  distance = (c1 - c2).norm();

  return true;
}

} // namespace axel
