/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/marker.h>
#include <momentum/character/types.h>
#include <momentum/common/filesystem.h>
#include <momentum/io/gltf/gltf_file_format.h>
#include <momentum/math/types.h>

#include <fx/gltf.h>
#include <gsl/span>

#include <tuple>
#include <vector>

namespace momentum {

Character loadGltfCharacter(fx::gltf::Document& model);

Character loadGltfCharacter(const filesystem::path& gltfFilename);

Character loadGltfCharacter(gsl::span<const std::byte> byteSpan);

std::tuple<MotionParameters, IdentityParameters, float> loadMotion(
    const filesystem::path& gltfFilename);

/// Load a glTF character from a local file path.
///
/// This function assumes the file format of the given path is glTF without checking the extension,
/// so please ensure the file is in glTF format.
///
/// @param[in] filepath The path to the glTF character file.
/// @return A tuple containing the loaded Character object, the motion represented in model
/// parameters, the identity vector represented as joint parameters, and the fps.
std::tuple<Character, MatrixXf, VectorXf, float> loadCharacterWithMotion(
    const filesystem::path& gltfFilename);

/// Load a glTF character from a buffer.
///
/// @param[in] byteSpan The buffer containing the glTF character data.
/// @return A tuple containing the loaded Character object, the motion represented in model
/// parameters, the identity vector represented as joint parameters, and the fps.
std::tuple<Character, MatrixXf, VectorXf, float> loadCharacterWithMotion(
    gsl::span<const std::byte> byteSpan);

/// Maps the loaded motion onto the input character by matching joint names and parameter names.
///
/// This function assumes the file format of the given path is glTF without checking the extension,
/// so please ensure the file is in glTF format.
///
/// @param[in] gltfFilename The path to the glTF motion file.
/// @param[in] character The Character object to map the motion onto.
/// @return A tuple containing the motion represented in model parameters, the identity vector
/// represented as joint parameters, and the fps. The model parameters and joint parameters are
/// mapped to the input character by name matching.
std::tuple<MatrixXf, VectorXf, float> loadMotionOnCharacter(
    const filesystem::path& gltfFilename,
    const Character& character);

/// Buffer version of loadMotionOnCharacter()
///
/// @param[in] byteSpan The buffer containing the glTF motion data.
/// @param[in] character The Character object to map the motion onto.
/// @return A tuple containing the motion represented in model parameters, the identity vector
/// represented as joint parameters, and the fps. The model parameters and joint parameters are
/// mapped to the input character by name matching.
std::tuple<MatrixXf, VectorXf, float> loadMotionOnCharacter(
    gsl::span<const std::byte> byteSpan,
    const Character& character);

/// Loads a MarkerSequence from a file.
///
/// @param[in] filename Path to the file containing the MarkerSequence data.
/// @return A MarkerSequence object containing motion capture marker data.
MarkerSequence loadMarkerSequence(const filesystem::path& filename);

fx::gltf::Document makeCharacterDocument(
    const Character& character,
    float fps = 120.0f,
    const MotionParameters& motion = {},
    const IdentityParameters& offsets = {},
    const std::vector<std::vector<Marker>>& markerSequence = {},
    bool embedResource = true);

/// Saves character motion to a glb file.
///
/// @param[in] motion The model parameters representing the motion of the character (numModelParams,
/// numFrames)
/// @param[in] offsets Offset values per joint capturing the skeleton bone lengths using translation
/// and scale offset (7*numJoints, 1)
void saveCharacter(
    const filesystem::path& filename,
    const Character& Character,
    float fps = 120.0f,
    const MotionParameters& motion = {},
    const IdentityParameters& offsets = {},
    const std::vector<std::vector<Marker>>& markerSequence = {},
    GltfFileFormat fileFormat = GltfFileFormat::EXTENSION);

/// Saves character skeleton states to a glb file.
///
/// @param[in] skeletonStates The skeleton states for each frame of the motion sequence (numFrames,
/// numJoints, 8)
void saveCharacter(
    const filesystem::path& filename,
    const Character& Character,
    float fps,
    gsl::span<const SkeletonState> skeletonStates,
    const std::vector<std::vector<Marker>>& markerSequence = {},
    GltfFileFormat fileFormat = GltfFileFormat::EXTENSION);

} // namespace momentum
