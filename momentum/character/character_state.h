/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>
#include <momentum/character/locator_state.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/math/fwd.h>

namespace momentum {

template <typename T>
struct CharacterStateT {
  CharacterParameters parameters;
  SkeletonState skeletonState;
  LocatorState locatorState;
  Mesh_u meshState;
  CollisionGeometryState_u collisionState;

  CharacterStateT();
  explicit CharacterStateT(const CharacterStateT& other);
  CharacterStateT(CharacterStateT&& c) noexcept;
  CharacterStateT& operator=(const CharacterStateT& rhs) = delete;
  CharacterStateT& operator=(CharacterStateT&& rhs) noexcept;
  ~CharacterStateT();

  explicit CharacterStateT(
      const Character& referenceCharacter,
      bool updateMesh = true,
      bool updateCollision = true);
  CharacterStateT(
      const CharacterParameters& parameters,
      const Character& referenceCharacter,
      bool updateMesh = true,
      bool updateCollision = true,
      bool applyLimits = true);
  void set(
      const CharacterParameters& parameters,
      const Character& referenceCharacter,
      bool updateMesh = true,
      bool updateCollision = true,
      bool applyLimits = true);
  void setBindPose(
      const Character& referenceCharacter,
      bool updateMesh = true,
      bool updateCollision = true);
};

} // namespace momentum
