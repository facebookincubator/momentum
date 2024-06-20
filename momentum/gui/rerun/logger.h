/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>
#include <momentum/character/fwd.h>
#include <momentum/character/locator.h>
#include <momentum/character/locator_state.h>
#include <momentum/character/marker.h>

#include <rerun.hpp>

#include <map>
#include <string>
#include <vector>

namespace momentum {

void logMesh(const rerun::RecordingStream& rec, const std::string& streamName, const Mesh& mesh);

void logMarkers(
    const rerun::RecordingStream& rec,
    const std::string& streamName,
    gsl::span<const Marker> markers);

void logLocators(
    const rerun::RecordingStream& rec,
    const std::string& streamName,
    const LocatorList& locators,
    const LocatorState& locatorState);

void logMarkerLocatorCorrespondence(
    const rerun::RecordingStream& rec,
    const std::string& streamName,
    const std::map<std::string, size_t>& locatorLookup,
    const LocatorState& locatorState,
    gsl::span<const Marker> markers,
    float kPositionErrorThreshold);

void logBvh(
    const rerun::RecordingStream& rec,
    const std::string& streamName,
    const CollisionGeometry& collisionGeometry,
    const SkeletonState& skeletonState);

void logCollisionGeometry(
    const rerun::RecordingStream& rec,
    const std::string& streamName,
    const CollisionGeometry& collisionGeometry,
    const SkeletonState& skeletonState);

void logCharacter(
    const rerun::RecordingStream& rec,
    const std::string& charStreamName,
    const Character& character,
    const CharacterState& characterState);

// Separate logs for world parameters vs pose parameters because they are in different scale
void logModelParams(
    const rerun::RecordingStream& rec,
    const std::string& worldPrefix,
    const std::string& posePrefix,
    gsl::span<const std::string> names,
    const Eigen::VectorXf& params);

void logJointParams(
    const rerun::RecordingStream& rec,
    const std::string& worldPrefix,
    const std::string& posePrefix,
    gsl::span<const std::string> names,
    const Eigen::VectorXf& params);

void logModelParamNames(
    const rerun::RecordingStream& rec,
    const std::string& worldPrefix,
    const std::string& posePrefix,
    gsl::span<const std::string> names);

void logJointParamNames(
    const rerun::RecordingStream& rec,
    const std::string& worldPrefix,
    const std::string& posePrefix,
    gsl::span<const std::string> names);

/// Logs to draw a plane as a grid.
void logGround(
    const rerun::RecordingStream& rec,
    const std::string& streamName,
    float from,
    float to,
    size_t n,
    float height = 0);

} // namespace momentum
