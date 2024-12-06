/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>
#include <momentum/math/mppca.h>

namespace momentum {

/// Creates a character with a customizable number of joints.
///
/// @param numJoints The number of joints in the resulting character.
/// @note: The `numJoints` parameter should be equal to 3 or greater.
template <typename T = float>
[[nodiscard]] CharacterT<T> createTestCharacter(size_t numJoints = 3);
// TODO: Currently, the number of parameters is fixed to 10, affecting only the first 3 joints.
// Additional work is needed to scale the number of parameters with the number of joints.

template <typename T>
[[nodiscard]] std::shared_ptr<const MppcaT<T>> createDefaultPosePrior();

template <typename T>
[[nodiscard]] CharacterT<T> withTestBlendShapes(const CharacterT<T>& character);

template <typename T>
[[nodiscard]] CharacterT<T> withTestFaceExpressionBlendShapes(const CharacterT<T>& character);

} // namespace momentum
