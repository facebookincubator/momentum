/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/character/types.h>
#include <momentum/math/fwd.h>
#include <momentum/math/types.h>

namespace momentum {

template <typename T>
std::vector<Vector3<T>> applySSD(
    const TransformationListT<T>& inverseBindPose,
    const SkinWeights& skin,
    typename DeduceSpanType<const Vector3<T>>::type points,
    const SkeletonStateT<T>& state);

template <typename T>
void applySSD(
    const TransformationListT<T>& inverseBindPose,
    const SkinWeights& skin,
    const MeshT<T>& mesh,
    const SkeletonStateT<T>& state,
    MeshT<T>& outputMesh);

template <typename T>
void applySSD(
    const TransformationListT<T>& inverseBindPose,
    const SkinWeights& skin,
    const MeshT<T>& mesh,
    const JointStateListT<T>& state,
    MeshT<T>& outputMesh);

Affine3f getInverseSSDTransformation(
    const TransformationList& inverseBindPose,
    const SkinWeights& skin,
    const SkeletonState& state,
    const size_t index);

std::vector<Vector3f> applyInverseSSD(
    const TransformationList& inverseBindPose,
    const SkinWeights& skin,
    gsl::span<const Vector3f> points,
    const SkeletonState& state);

void applyInverseSSD(
    const TransformationList& inverseBindPose,
    const SkinWeights& skin,
    gsl::span<const Vector3f> points,
    const SkeletonState& state,
    Mesh& mesh);

} // namespace momentum
