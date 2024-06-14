/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/joint_state.h>
#include <momentum/character/types.h>
#include <momentum/math/types.h>

namespace momentum {

struct StateSimilarity {
  VectorXf positionError;
  VectorXf orientationError;
  float positionRMSE;
  float orientationRMSE;
  float positionMax;
  float orientationMax;
};

// skeleton state class
template <typename T>
struct SkeletonStateT {
  JointParametersT<T> jointParameters;
  JointStateListT<T> jointState;

  // create empty skeleton state
  SkeletonStateT() noexcept {}

  // create skeleton state for the given parameters and skeleton
  SkeletonStateT(
      const JointParametersT<T>& parameters,
      const Skeleton& referenceSkeleton,
      bool computeDeriv = true);
  SkeletonStateT(
      JointParametersT<T>&& parameters,
      const Skeleton& referenceSkeleton,
      bool computeDeriv = true);

  template <typename T2>
  explicit SkeletonStateT(const SkeletonStateT<T2>& rhs);

  // update skeleton state for the given joint parameters and skeleton
  void set(
      const JointParametersT<T>& jointParameters,
      const Skeleton& referenceSkeleton,
      bool computeDeriv = true);

  void set(
      JointParametersT<T>&& jointParameters,
      const Skeleton& referenceSkeleton,
      bool computeDeriv = true);

  template <typename T2>
  void set(const SkeletonStateT<T2>& rhs);

  // compare to skeleton states with each other, returning all kinds of similarity measures
  static StateSimilarity compare(const SkeletonStateT<T>& state1, const SkeletonStateT<T>& state2);

  AffineTransform3ListT<T> toTransforms() const;

  template <typename T2>
  SkeletonStateT<T2> cast() const {
    SkeletonStateT<T2> result;
    result.set(*this);
    return result;
  }

 private:
  // update skeleton state for the given skeleton
  void set(const Skeleton& referenceSkeleton, bool computeDeriv);

  template <typename T2>
  void copy(const SkeletonStateT<T2>& rhs);
};

// The relative transform taking us from one local joint space to another.
// This equivalent to
//     (T_B)^{-1} T_A
// but computed in a much more numerically stable manner as it doesn't
// involve passing through world space on the way.
template <typename T>
AffineTransform3<T> transformAtoB(
    size_t jointA,
    size_t jointB,
    const Skeleton& referenceSkeleton,
    const SkeletonStateT<T>& skelState);

} // namespace momentum
