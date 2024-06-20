/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/common/filesystem.h>
#include <momentum/math/types.h>

#include <gsl/span>

namespace momentum {

Character loadFbxCharacter(const filesystem::path& inputPath);

Character loadFbxCharacter(gsl::span<const std::byte> inputSpan);

void saveFbx(
    const filesystem::path& filename,
    const Character& character,
    const MatrixXf& poses = MatrixXf(),
    const VectorXf& identity = VectorXf(),
    double framerate = 120.0,
    bool saveMesh = false);

// A shorthand of saveFbx() to save both the skeleton and mesh as a model but without any animation
void saveFbxModel(const filesystem::path& filename, const Character& character);

} // namespace momentum
