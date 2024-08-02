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

// UpVector Specifies which canonical axis represents up in the system
// (typically Y or Z). Maps to fbxsdk::FbxAxisSystem::eUpVector
enum class FBXUpVector { XAxis = 1, YAxis = 2, ZAxis = 3 };

// FrontVector  Vector with origin at the screen pointing toward the camera.
// This is a subset of enum EUpVector because axis cannot be repeated.
// We use the system of "parity" to define this vector because its value (X,Y or
// Z axis) really depends on the up-vector. The EPreDefinedAxisSystem list the
// up-vector, parity and coordinate system values for the predefined systems.
// Maps to fbxsdk::FbxAxisSystem::eFrontVector
enum class FBXFrontVector { ParityEven = 1, ParityOdd = 2 };

// CoordSystem Specifies the third vector of the system.
// Maps to fbxsdk::FbxAxisSystem::eCoorSystem
enum class FBXCoordSystem { RightHanded, LeftHanded };

// A struct containing the up, front vectors and coordinate system
struct FBXCoordSystemInfo {
  // Default to the same orientations as FbxAxisSystem::eMayaYUp
  FBXUpVector upVector = FBXUpVector::YAxis;
  FBXFrontVector frontVector = FBXFrontVector::ParityOdd;
  FBXCoordSystem coordSystem = FBXCoordSystem::RightHanded;
};

Character loadFbxCharacter(const filesystem::path& inputPath);

Character loadFbxCharacter(gsl::span<const std::byte> inputSpan);

void saveFbx(
    const filesystem::path& filename,
    const Character& character,
    const MatrixXf& poses = MatrixXf(),
    const VectorXf& identity = VectorXf(),
    double framerate = 120.0,
    bool saveMesh = false,
    const FBXCoordSystemInfo& coordSystemInfo = FBXCoordSystemInfo());

void saveFbxWithJointParams(
    const filesystem::path& filename,
    const Character& character,
    const MatrixXf& jointParams = MatrixXf(),
    double framerate = 120.0,
    bool saveMesh = false,
    const FBXCoordSystemInfo& coordSystemInfo = FBXCoordSystemInfo());

// A shorthand of saveFbx() to save both the skeleton and mesh as a model but without any animation
void saveFbxModel(
    const filesystem::path& filename,
    const Character& character,
    const FBXCoordSystemInfo& coordSystemInfo = FBXCoordSystemInfo());

} // namespace momentum
