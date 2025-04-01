/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <Eigen/Geometry>

#include "axel/common/VectorizationTypes.h"

// The definitions here are split in this way to still explicitly instantiate
// common template types to reduce compilation and binary size in axel, while still
// allowing an external user to include this file and provide an explicit specialization
// for a new type.
namespace axel {

template <class S>
bool projectOnTriangle(
    const Eigen::Vector3<S>& p,
    const Eigen::Vector3<S>& a,
    const Eigen::Vector3<S>& b,
    const Eigen::Vector3<S>& c,
    Eigen::Vector3<S>& q,
    Eigen::Vector3<S>* barycentric) {
  // See book 'Real-Time Collision Detection' Ericson Section 5.1.5
  // Check if P in vertex region outside A
  const Eigen::Vector3<S> ab = b - a;
  const Eigen::Vector3<S> ac = c - a;
  const Eigen::Vector3<S> ap = p - a;
  const S d1 = ab.dot(ap);
  const S d2 = ac.dot(ap);
  constexpr S kZero(0.0);
  if (d1 <= kZero && d2 <= kZero) {
    // barycentric coordinates (1.0, 0.0, 0.0)
    q = a;
    if (barycentric) {
      *barycentric = Eigen::Vector3<S>::UnitX();
    }
    return false;
  }

  // Check if P in vertex region outside B
  const Eigen::Vector3<S> bp = p - b;
  const S d3 = ab.dot(bp);
  const S d4 = ac.dot(bp);
  if (d3 >= kZero && d4 <= d3) {
    // barycentric coordinate (0.0, 1.0, 0.0)
    q = b;
    if (barycentric) {
      *barycentric = Eigen::Vector3<S>::UnitY();
    }
    return false;
  }

  // Check if P in edge region of AB, if so return projection of P onto AB
  const S vc = d1 * d4 - d3 * d2;
  if (vc <= kZero && d1 >= kZero && d3 <= kZero) {
    // barycentric coordinate (1.0-v, v, 0.0)
    const S v = d1 / (d1 - d3);
    q = a + v * ab;
    if (barycentric) {
      *barycentric = Eigen::Vector3<S>(S(1) - v, v, S(0));
    }
    return false;
  }

  // Check if P in vertex region outside C
  const Eigen::Vector3<S> cp = p - c;
  const S d5 = ab.dot(cp);
  const S d6 = ac.dot(cp);
  if (d6 >= kZero && d5 <= d6) {
    // barycentric coordinate (0.0, 0.0, 1.0)
    q = c;
    if (barycentric) {
      *barycentric = Eigen::Vector3<S>::UnitZ();
    }
    return false;
  }

  // Check if P in edge region of AC, if so return projection of P onto AC
  const S vb = d5 * d2 - d1 * d6;
  if (vb <= kZero && d2 >= kZero && d6 <= kZero) {
    // barycentric coordinate (1.0-w, 0.0, w)
    const S w = d2 / (d2 - d6);
    q = a + w * ac;
    if (barycentric) {
      *barycentric = Eigen::Vector3<S>(S(1) - w, S(0), w);
    }
    return false;
  }

  // Check if P in edge region of BC, if so return projection of P onto BC
  const S va = d3 * d6 - d5 * d4;
  if (va <= kZero && (d4 - d3) >= kZero && (d5 - d6) >= kZero) {
    // barycentric coordinate (0.0, 1.0-w, w)
    const S w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    q = b + w * (c - b);
    if (barycentric) {
      *barycentric = Eigen::Vector3<S>(S(0), S(1) - w, w);
    }
    return false;
  }

  // P inside face region.
  // Compute Q through its barycentric coordinates (u, v, w)
  const S denom = S(1.0) / (va + vb + vc);
  const S v = vb * denom;
  const S w = vc * denom;
  q = a + ab * v + ac * w;
  if (barycentric) {
    *barycentric = Eigen::Vector3<S>(S(1) - v - w, v, w);
  }
  return true;
}

template <typename S>
WideMask<WideScalar<S>> projectOnTriangle(
    const WideVec3<S>& p,
    const WideVec3<S>& a,
    const WideVec3<S>& b,
    const WideVec3<S>& c,
    WideVec3<S>& q,
    WideVec3<S>* barycentric) {
  // Check if P in vertex region outside A
  const auto ab = b - a;
  const auto ac = c - a;
  const auto ap = p - a;
  const auto d1 = drjit::dot(ab, ap);
  const auto d2 = drjit::dot(ac, ap);
  auto isOutside = d1 <= 0.0 && d2 <= 0.0;
  q = drjit::select(isOutside, a, q);
  if (barycentric) {
    // barycentric coordinates (1.0, 0.0, 0.0)
    *barycentric = drjit::select(isOutside, WideVec3<S>(S(1), S(0), S(0)), *barycentric);
  }

  // Check if P in vertex region outside B
  const auto bp = p - b;
  const auto d3 = drjit::dot(ab, bp);
  const auto d4 = drjit::dot(ac, bp);
  const auto isOutside2 = d3 >= 0.0 && d4 <= d3;
  isOutside = isOutside || isOutside2;
  q = drjit::select(isOutside2, b, q);
  if (barycentric) {
    // barycentric coordinates (0.0, 1.0, 0.0)
    *barycentric = drjit::select(isOutside2, WideVec3<S>(S(0), S(1), S(0)), *barycentric);
  }

  // Check if P in edge region of AB, if so return projection of P onto AB
  const auto vc = d1 * d4 - d3 * d2;
  const auto isOutside3 = vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0;
  isOutside = isOutside || isOutside3;
  const auto tAB = d1 / (d1 - d3);
  q = drjit::select(isOutside3, a + tAB * ab, q);
  if (barycentric) {
    // barycentric coordinate (1.0-v, v, 0.0)
    *barycentric = drjit::select(isOutside3, WideVec3<S>(S(1) - tAB, tAB, S(0)), *barycentric);
  }

  // Check if P in vertex region outside C
  const auto cp = p - c;
  const auto d5 = drjit::dot(ab, cp);
  const auto d6 = drjit::dot(ac, cp);
  const auto isOutside4 = d6 >= 0.0 && d5 <= d6;
  isOutside = isOutside || isOutside4;
  q = drjit::select(isOutside4, c, q);
  if (barycentric) {
    // barycentric coordinate (0.0, 0.0, 1.0)
    *barycentric = drjit::select(isOutside4, WideVec3<S>(S(0), S(0), S(1)), *barycentric);
  }

  // Check if P in edge region of AC, if so return projection of P onto AC
  const auto vb = d5 * d2 - d1 * d6;
  const auto isOutside5 = vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0;
  isOutside = isOutside || isOutside5;
  const auto tAC = d2 / (d2 - d6);
  q = drjit::select(isOutside5, a + tAC * ac, q);
  if (barycentric) {
    // barycentric coordinate (1.0-w, 0.0, w)
    *barycentric = drjit::select(isOutside5, WideVec3<S>(S(1) - tAC, S(0), tAC), *barycentric);
  }

  // Check if P in edge region of BC, if so return projection of P onto BC
  const auto va = d3 * d6 - d5 * d4;
  const auto isOutside6 = va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0;
  isOutside = isOutside || isOutside6;
  const auto tBC = (d4 - d3) / ((d4 - d3) + (d5 - d6));
  q = drjit::select(isOutside6, b + tBC * (c - b), q);
  if (barycentric) {
    // barycentric coordinate (0.0, 1.0-w, w)
    *barycentric = drjit::select(isOutside6, WideVec3<S>(S(0), S(1) - tBC, tBC), *barycentric);
  }

  // P inside face region.
  // Compute Q through its barycentric coordinates (u, v, w)
  const auto denom = 1.0 / (va + vb + vc);
  const auto v = vb * denom;
  const auto w = vc * denom;
  q = drjit::select(isOutside, q, a + ab * v + ac * w);
  if (barycentric) {
    *barycentric = drjit::select(isOutside, *barycentric, WideVec3<S>(S(1) - v - w, v, w));
  }
  return !isOutside;
}

} // namespace axel
