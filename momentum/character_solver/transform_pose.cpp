/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character/character.h"
#include "momentum/character/parameter_transform.h"
#include "momentum/character/skeleton.h"
#include "momentum/character_solver/orientation_error_function.h"
#include "momentum/character_solver/position_error_function.h"
#include "momentum/character_solver/skeleton_solver_function.h"
#include "momentum/common/checks.h"
#include "momentum/common/log.h"
#include "momentum/math/constants.h"
#include "momentum/math/random.h"
#include "momentum/solver/gauss_newton_solver.h"

#include <dispenso/parallel_for.h>

#include <numeric>
#include <tuple>

namespace momentum {

namespace {

momentum::Character createRigidBodySkeleton(const momentum::Character& character) {
  const auto rigidBodyParameters = character.parameterTransform.getRigidParameters();

  std::vector<bool> rigidJoints = character.parametersToActiveJoints(rigidBodyParameters);
  // Note that we need to keep _all_ the parameters that affect the rigid joints.

  // The rigidly transformed joints should appear near the top of the skeleton.
  // Rather than trying to be fancy about this, we'll just find the last affected joint and
  // keep all the joints in the list before it.  This will also allow us to much more easily
  // simplify the parameter transform since we don't need to deal with missing joints in the
  // middle.
  size_t numRigidJoints = 0;
  for (size_t i = 0; i < rigidJoints.size(); ++i) {
    if (rigidJoints[i]) {
      numRigidJoints = i + 1;
    }
  }

  if (numRigidJoints == 0) {
    throw std::runtime_error("No rigid joints found in character, unable to apply transformation.");
  }

  // Need to include all joints and parameters in the parent chain:
  std::fill(rigidJoints.begin(), rigidJoints.begin() + numRigidJoints, true);

  // Retain all parameters that affect the rigid pose, which means all parameters
  // affecting joints above the root.
  const auto parametersToKeep = character.activeJointsToParameters(rigidJoints);

  // This only changes the parameters:
  auto [intermediateTransform, intermediateLimits] =
      subsetParameterTransform(character.parameterTransform, ParameterLimits{}, parametersToKeep);

  // Construct the new skeleton by just clipping the skeleton to the rigid joints.
  momentum::Skeleton resultSkeleton;
  resultSkeleton.joints.reserve(numRigidJoints);
  for (size_t i = 0; i < numRigidJoints; ++i) {
    resultSkeleton.joints.push_back(character.skeleton.joints.at(i));
  }

  std::vector<size_t> jointMapping(character.skeleton.joints.size(), kInvalidIndex);
  std::iota(jointMapping.begin(), jointMapping.begin() + resultSkeleton.joints.size(), 0);

  return {
      resultSkeleton,
      mapParameterTransformJoints(
          intermediateTransform, resultSkeleton.joints.size(), jointMapping)};
}

} // namespace

template <typename T>
std::vector<ModelParametersT<T>> transformPose(
    const Character& character,
    const std::vector<ModelParametersT<T>>& modelParameters,
    const std::vector<TransformT<T>>& transforms) {
  MT_CHECK(modelParameters.size() == transforms.size(), "Mismatched transforms and parameters");

  // Applies a transform to a character by solving for the new set of model
  // parameters such that the new character's root matches up with the transformed
  // version of the old character's root.  This could in principle be done in a closed
  // form but such a solution would need to be hand-coded for each parametrization
  // of character, so for now we'll do it using an IK solve (which is very fast
  // in this case because we're only working with the rigid parameters).
  const auto nPoses = modelParameters.size();

  const momentum::Character characterSimplified = createRigidBodySkeleton(character);
  MT_CHECK(!characterSimplified.skeleton.joints.empty(), "No joints in simplified character");

  // Use the farthest joint since it should have all parameters applied to it:
  const auto rootJointSimplified = characterSimplified.skeleton.joints.size() - 1;
  const auto rootJointFull = rootJointSimplified;

  MT_CHECK(rootJointSimplified != momentum::kInvalidIndex);
  const auto rigidParametersSimplified =
      characterSimplified.parameterTransform.getRigidParameters();

  const momentum::ParameterTransformT<T> parameterTransformSimplified =
      characterSimplified.parameterTransform.cast<T>();

  // TODO would be nice if we can get rid of this in the float case.
  const momentum::ParameterTransformT<T> parameterTransformFull =
      character.parameterTransform.cast<T>();

  std::vector<size_t> simplifiedParamToFullParamIdx(
      parameterTransformSimplified.numAllModelParameters());
  for (Eigen::Index iSimplifiedParam = 0;
       iSimplifiedParam < characterSimplified.parameterTransform.numAllModelParameters();
       ++iSimplifiedParam) {
    const auto fullParamIdx = character.parameterTransform.getParameterIdByName(
        characterSimplified.parameterTransform.name[iSimplifiedParam]);
    MT_CHECK(fullParamIdx != momentum::kInvalidIndex);
    simplifiedParamToFullParamIdx.at(iSimplifiedParam) = fullParamIdx;
  }

  std::vector<momentum::ModelParametersT<T>> result = modelParameters;
  dispenso::parallel_for(size_t(0), nPoses, [&](size_t iFrame) {
    const auto& fullParams_init = modelParameters.at(iFrame);
    if (fullParams_init.size() == 0) {
      return;
    }

    const TransformT<T>& transform = transforms[iFrame];

    const momentum::SkeletonStateT<T> skelStateFull(
        parameterTransformFull.apply(fullParams_init), character.skeleton, false);
    const Eigen::Vector3<T> worldFromRootTranslationTarget =
        transform * skelStateFull.jointState[rootJointFull].translation;
    const Eigen::Quaternion<T> worldFromRootRotationTarget =
        transform.rotation * skelStateFull.jointState[rootJointFull].rotation;

    momentum::PositionErrorFunctionT<T> positionError(characterSimplified);
    positionError.addConstraint(PositionDataT<T>(
        Eigen::Vector3<T>::Zero(), worldFromRootTranslationTarget, rootJointSimplified, 100.0));

    momentum::OrientationErrorFunctionT<T> orientationError(characterSimplified);
    orientationError.addConstraint(OrientationDataT<T>(
        Eigen::Quaternion<T>::Identity(), worldFromRootRotationTarget, rootJointSimplified, 10.0));

    momentum::SkeletonSolverFunctionT<T> solverFunction(
        &characterSimplified.skeleton, &parameterTransformSimplified);
    solverFunction.addErrorFunction(&positionError);
    solverFunction.addErrorFunction(&orientationError);

    // Initialize with rest params
    ModelParametersT<T> solvedParametersSimplified =
        ModelParametersT<T>::Zero(solverFunction.getNumParameters());
    for (size_t iSimplifiedParam = 0; iSimplifiedParam < simplifiedParamToFullParamIdx.size();
         ++iSimplifiedParam) {
      const auto fullParamIdx = simplifiedParamToFullParamIdx[iSimplifiedParam];
      solvedParametersSimplified[iSimplifiedParam] = fullParams_init[fullParamIdx];
    }

    momentum::GaussNewtonSolverOptions solverOptions;
    solverOptions.maxIterations = 1000;
    solverOptions.minIterations = 10;
    // It shouldn't need regularization:
    solverOptions.regularization = std::numeric_limits<T>::epsilon();
    solverOptions.useBlockJtJ = true;

    // Start with a fresh RNG for each solve to ensure repeatability.
    momentum::Random<> rng;

    // The momentum solver has a lot of local minima issues related to the use
    // of Euler angles.  As a result for any given solve there is a chance if will
    // get stuck somewhere non-optimal, which would be really bad in this case because
    // we'd end up with a single pose out of the set which is flipped backward or whatever.
    // To avoid this, we'll do the solve with restarts: each solve will start with a random
    // set of rigid parameters and we'll run to convergence, then exit when the resulting
    // rotation is close enough.
    momentum::SkeletonStateT<T> skelStateSimplified;
    GaussNewtonSolverT<T> solver(solverOptions, &solverFunction);
    solver.setEnabledParameters(rigidParametersSimplified);

    for (size_t iIter = 0;; ++iIter) {
      if (iIter > 10000) {
        MT_LOGW(
            "Solver failed to converge after {} iterations when solving for rigid transform.",
            iIter);
        break;
      }

      // Randomize the starting location to deal with Euler angle local minimum issues:
      for (int k = 0; k < characterSimplified.parameterTransform.numAllModelParameters(); ++k) {
        if (rigidParametersSimplified.test(k)) {
          solvedParametersSimplified[k] = rng.uniform(-pi(), pi());
        }
      }
      solver.solve(solvedParametersSimplified.v);
      skelStateSimplified.set(
          parameterTransformSimplified.apply(solvedParametersSimplified),
          characterSimplified.skeleton,
          false);
      if ((skelStateSimplified.jointState[rootJointSimplified].rotation.toRotationMatrix() -
           worldFromRootRotationTarget.toRotationMatrix())
              .norm() < 1e-3) {
        break;
      }
    }

    ModelParametersT<T> fullParams_final = fullParams_init;
    for (size_t simplifiedParamIdx = 0; simplifiedParamIdx < simplifiedParamToFullParamIdx.size();
         ++simplifiedParamIdx) {
      if (rigidParametersSimplified.test(simplifiedParamIdx)) {
        const auto fullParamIdx = simplifiedParamToFullParamIdx[simplifiedParamIdx];
        fullParams_final.v[fullParamIdx] = solvedParametersSimplified[simplifiedParamIdx];
      }
    }

    result[iFrame] = fullParams_final;
  });

  return result;
}

template std::vector<ModelParametersT<float>> transformPose(
    const Character& characterFull,
    const std::vector<ModelParametersT<float>>& modelParametersFull,
    const std::vector<TransformT<float>>& transforms);

template std::vector<ModelParametersT<double>> transformPose(
    const Character& characterFull,
    const std::vector<ModelParametersT<double>>& modelParametersFull,
    const std::vector<TransformT<double>>& transforms);

} // namespace momentum
