/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/marker.h>
#include <momentum/character/types.h>
#include <pybind11/numpy.h>

#include <optional>
#include <string>

// Forward declarations
namespace momentum {
struct FBXCoordSystemInfo;
} // namespace momentum

namespace pymomentum {

// We need to wrap around momentum io functions because the motion matrix in
// pymomentum is a transpose. Momentum represents motion as numParameters X
// numFrames, and pymomentum represents motion as numFrames x numParameters.

// Using row-major matrices for input/output avoids some extra copies in numpy:
using RowMatrixf =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

momentum::Character loadGLTFCharacterFromFile(const std::string& path);

void saveGLTFCharacterToFile(
    const std::string& path,
    const momentum::Character& character,
    const float fps,
    std::optional<const momentum::MotionParameters> motion,
    std::optional<const momentum::IdentityParameters> offsets,
    std::optional<const std::vector<std::vector<momentum::Marker>>> markers);

void saveGLTFCharacterToFileFromSkelStates(
    const std::string& path,
    const momentum::Character& character,
    const float fps,
    const pybind11::array_t<float>& skel_states,
    std::optional<const std::vector<std::vector<momentum::Marker>>> markers);

void saveFBXCharacterToFile(
    const std::string& path,
    const momentum::Character& character,
    const float fps,
    std::optional<const Eigen::MatrixXf> motion,
    std::optional<const Eigen::VectorXf> offsets,
    std::optional<const momentum::FBXCoordSystemInfo> coordSystemInfo);

void saveFBXCharacterToFileWithJointParams(
    const std::string& path,
    const momentum::Character& character,
    const float fps,
    std::optional<const Eigen::MatrixXf> jointParams,
    std::optional<const momentum::FBXCoordSystemInfo> coordSystemInfo);

std::tuple<momentum::Character, RowMatrixf, Eigen::VectorXf, float>
loadCharacterWithMotion(const std::string& gltfFilename);

std::string toGLTF(const momentum::Character& character);

std::tuple<
    RowMatrixf,
    std::vector<std::string>,
    Eigen::VectorXf,
    std::vector<std::string>>
loadMotion(const std::string& gltfFilename);

std::vector<momentum::MarkerSequence> loadMarkersFromFile(
    const std::string& path,
    const bool mainSubjectOnly = true);

} // namespace pymomentum
