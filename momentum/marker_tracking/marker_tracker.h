/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>
#include <momentum/character/marker.h>

namespace marker_tracking {

/// Common configuration for a tracking problem
struct BaseConfig {
  /// Minimum percentage of visible markers in a frame to consider for tracking. We will skip frames
  /// with too few visible markers as they may be better filled in through a smoothing pass.
  float minVisPercent = 0.f;
  /// Parameter to control what loss function to use. Refer to comments in GeneralizedLoss class for
  /// details. Use a smaller alpha when data is noisy; otherwise L2 is good.
  float lossAlpha = 2.0;
  /// Max number of solver iterations to run.
  size_t maxIter = 30;
  /// True to print and save debug information.
  bool debug = false;
};

/// Configuration for running body and/or locator calibration
struct CalibrationConfig : public BaseConfig {
  /// Number of frames used in calibration. It will be a uniform sample from the input.
  size_t calibFrames = 100;
  /// Number of iterations to run the main calibration loop. It could be larger if calibrating for
  /// locators only.
  size_t majorIter = 3;
  /// True to only solve for a global body scale without changing individual bone length.
  bool globalScaleOnly = false;
  /// True to calibrate only the locators and not the body.
  bool locatorsOnly = false;
};

/// Configuration for pose tracking given a calibrated body and locators
struct TrackingConfig : public BaseConfig {
  /// The weight for the smoothness error function.
  float smoothing = 0;
  /// The weight for the collision error function.
  float collisionErrorWeight = 0.0;
  /// Smoothing weights per model parameter. The size of this vector should be equal to number of
  /// model parameters and this overrides the value specific in smoothing
  Eigen::VectorXf smoothingWeights{};
};

/// Configuration for refining an already tracked motion, eg. add smoothing and/or collision
/// handling; improve residuals with extra dofs etc.
struct RefineConfig : public TrackingConfig {
  /// Minimize changes to calibration parameters when in calibration mode, by regularizing towards
  /// the input value with this regularizer weight.
  float regularizer = 0.0;
  /// Calibrate identity parameters when refining the motion.
  bool calibId = false;
  /// Calibrate locators when refining the motion.
  bool calibLocators = false;
};

/// Use multiple frames to solve for global parameters such as body proportions and/or marker
/// offsets together with the motion. It can also be used to smooth out a motion with or without
/// solving for global parameters, for example to fill gaps when there are missing markers.
///
/// @param[in] markerData Marker data.
/// @param[in] character Character definition.
/// @param[in] globalParams Bitset to indicate global parameters to solve for; could be all zeros
/// for post-process a motion.
/// @param[in] initialMotion Initial values of all parameters. It should be the same length as
/// markerData, but only frames used in solving are used. Values in unused frames do not matter.
/// Number of parameters should be the same as defined in character.
/// @param[in] config Solving options.
/// @param[in] frameStride Frame stride to select solver frames (ie. uniform sample).
///
/// @return The solved motion. It has the same length as markerData. It repeats the same solved pose
/// within a frame stride.
Eigen::MatrixXf trackSequence(
    gsl::span<const std::vector<momentum::Marker>> markerData,
    const momentum::Character& character,
    const momentum::ParameterSet& globalParams,
    const Eigen::MatrixXf& initialMotion,
    const TrackingConfig& config,
    float regularizer = 0.0,
    size_t frameStride = 1);

/// Track poses per-frame given a calibrated character.
///
/// @param[in] markerData Input marker data.
/// @param[in] character Character definition.
/// @param[in] globalParams Calibrated identity info; could be repurposed to pass in an initial pose
/// too.
/// @param[in] config Solving options.
/// @param[in] frameStride Frame stride to select solver frames (ie. uniform sample).
///
/// @return The solved motion. It has the same length as markerData. It repeats the same solved pose
/// within a frame stride.
Eigen::MatrixXf trackPosesPerframe(
    gsl::span<const std::vector<momentum::Marker>> markerData,
    const momentum::Character& character,
    const momentum::ModelParameters& globalParams,
    const TrackingConfig& config,
    size_t frameStride = 1);

/// Calibrate body proportions and locator offsets of a character from input marker data.
///
/// @param[in] markerData Input marker data.
/// @param[in] config Solving options.
/// @param[in,out] character Character definition. It provides input locators offsets which will get
/// updated in return.
/// @param[in,out] identity Initial identity parameters that get updated in return. It could also
/// hold the pose of the first frame for better initialization for tracking later.
void calibrateModel(
    gsl::span<const std::vector<momentum::Marker>> markerData,
    const CalibrationConfig& config,
    momentum::Character& character,
    momentum::ModelParameters& identity);

/// Calibrate locator offsets of a character from input identity and marker data.
///
/// @param[in] markerData Input marker data.
/// @param[in] config Solving options.
/// @param[in] identity Identity parameters of the input character.
/// @param[in,out] character Character definition. It provides input locators offsets which will get
/// updated in return. We overwrite the locators in the input character so we don't have to
/// duplicate the character object inside the function.
void calibrateLocators(
    gsl::span<const std::vector<momentum::Marker>> markerData,
    const CalibrationConfig& config,
    const momentum::ModelParameters& identity,
    momentum::Character& character);

Eigen::MatrixXf refineMotion(
    gsl::span<const std::vector<momentum::Marker>> markerData,
    const Eigen::MatrixXf& motion,
    const RefineConfig& config,
    momentum::Character& character);

} // namespace marker_tracking
