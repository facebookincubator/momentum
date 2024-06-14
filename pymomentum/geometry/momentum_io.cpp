/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/geometry/momentum_io.h"

#include <momentum/character/character.h>
#include <momentum/character/joint_state.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/character/types.h>
#include <momentum/common/checks.h>
#include <momentum/io/fbx/fbx_io.h>
#include <momentum/io/gltf/gltf_io.h>
#include <momentum/io/marker/marker_io.h>

namespace pymomentum {

momentum::Character loadGLTFCharacterFromFile(const std::string& path) {
  return momentum::loadGltfCharacter(path);
}

momentum::MotionParameters transpose(
    const momentum::MotionParameters& motionParameters) {
  const auto& [parameters, poses] = motionParameters;
  return {parameters, poses.transpose()};
}

void saveGLTFCharacterToFile(
    const std::string& path,
    const momentum::Character& character,
    const float fps,
    std::optional<const momentum::MotionParameters> motion,
    std::optional<const momentum::IdentityParameters> offsets,
    std::optional<const std::vector<std::vector<momentum::Marker>>> markers) {
  if (motion.has_value()) {
    const auto& [parameters, poses] = motion.value();
    if (poses.cols() != parameters.size()) {
      std::ostringstream oss;
      oss << "Expected cols of motion parameters to be " << parameters.size()
          << " but got " << poses.cols();
      throw std::runtime_error(oss.str());
    }
  }
  momentum::saveCharacter(
      path,
      character,
      fps,
      transpose(motion.value_or(momentum::MotionParameters{})),
      offsets.value_or(momentum::IdentityParameters{}),
      markers.value_or(std::vector<std::vector<momentum::Marker>>{}));
}

void saveGLTFCharacterToFileFromSkelStates(
    const std::string& path,
    const momentum::Character& character,
    const float fps,
    const std::vector<RowMatrixf>& skel_states,
    const std::vector<RowMatrixf>& joint_params,
    std::optional<const std::vector<std::vector<momentum::Marker>>> markers) {
  if (markers.has_value()) {
    if (markers->size() != skel_states.size()) {
      std::ostringstream oss;
      oss << "The number of frames of the skeleton states array "
          << skel_states.size()
          << " does not coincide with the number of frames of the markers "
          << markers->size();
      throw std::length_error(oss.str());
    }
  }
  const int numFrames = skel_states.size();
  const int numJoints = skel_states[0].rows();
  const int numElements = skel_states[0].cols();
  MT_CHECK(
      numElements == 8,
      "Expecting size 8 (3 translation + 4 rotation + sale) for last dimension of the skel_states, but got {}",
      numElements);

  std::vector<momentum::SkeletonState> skeletonStates;

  for (int iFrame = 0; iFrame < numFrames; ++iFrame) {
    momentum::JointStateList jointStateList;
    std::vector<float> jointParamsVec;
    for (int iJoint = 0; iJoint < numJoints; ++iJoint) {
      jointParamsVec.push_back(joint_params[iFrame].coeff(iJoint, 0));
      jointParamsVec.push_back(joint_params[iFrame].coeff(iJoint, 1));
      jointParamsVec.push_back(joint_params[iFrame].coeff(iJoint, 2));
      jointParamsVec.push_back(joint_params[iFrame].coeff(iJoint, 3));
      jointParamsVec.push_back(joint_params[iFrame].coeff(iJoint, 4));
      jointParamsVec.push_back(joint_params[iFrame].coeff(iJoint, 5));
      jointParamsVec.push_back(joint_params[iFrame].coeff(iJoint, 6));

      Eigen::Quaternionf localRotation{
          joint_params[iFrame].coeff(iJoint, 3),
          joint_params[iFrame].coeff(iJoint, 4),
          joint_params[iFrame].coeff(iJoint, 5),
          joint_params[iFrame].coeff(iJoint, 6),
      };
      Eigen::Vector3f localTranslation{
          joint_params[iFrame].coeff(iJoint, 0),
          joint_params[iFrame].coeff(iJoint, 1),
          joint_params[iFrame].coeff(iJoint, 2),
      };
      float localScale = 1.0;
      Eigen::Quaternionf rotation{
          skel_states[iFrame].coeff(iJoint, 3),
          skel_states[iFrame].coeff(iJoint, 4),
          skel_states[iFrame].coeff(iJoint, 5),
          skel_states[iFrame].coeff(iJoint, 6),
      };
      Eigen::Vector3f translation{
          skel_states[iFrame].coeff(iJoint, 0),
          skel_states[iFrame].coeff(iJoint, 1),
          skel_states[iFrame].coeff(iJoint, 2),
      };
      float scale = skel_states[iFrame].coeff(iJoint, 7);
      Eigen::Transform<float, 3, Eigen::Affine> transformation =
          Eigen::Transform<float, 3, Eigen::Affine>::Identity();
      Eigen::Matrix3<float> translationAxis = Eigen::Matrix3<float>::Identity();
      Eigen::Matrix3<float> rotationAxis = Eigen::Matrix3<float>::Identity();

      momentum::JointState jointState{
          localRotation,
          localTranslation,
          localScale,
          rotation,
          translation,
          scale,
          transformation,
          translationAxis,
          rotationAxis,
      };
      jointStateList.push_back(jointState);
    }
    momentum::JointParameters jointParams = Eigen::Map<Eigen::VectorXf>(
        jointParamsVec.data(), jointParamsVec.size());
    momentum::SkeletonState skeletonState{jointParams, character.skeleton};
    skeletonStates.push_back(skeletonState);
  }
  momentum::saveCharacter(
      path,
      character,
      fps,
      skeletonStates,
      markers.value_or(std::vector<std::vector<momentum::Marker>>{}));
}

void saveFBXCharacterToFile(
    const std::string& path,
    const momentum::Character& character,
    const float fps,
    std::optional<const Eigen::MatrixXf> motion,
    std::optional<const Eigen::VectorXf> offsets) {
  if (motion.has_value() && offsets.has_value()) {
    momentum::saveFbx(
        path,
        character,
        motion.value().transpose(),
        offsets.value(),
        fps,
        true /*saveMesh*/);
  } else {
    momentum::saveFbxModel(path, character);
  }
}

std::tuple<momentum::Character, RowMatrixf, Eigen::VectorXf, float>
loadCharacterWithMotion(const std::string& gltfFilename) {
  const auto [character, motion, identity, fps] =
      momentum::loadCharacterWithMotion(gltfFilename);
  return std::make_tuple(character, motion.transpose(), identity, fps);
}

std::string toGLTF(const momentum::Character& character) {
  nlohmann::json j = momentum::makeCharacterDocument(character);
  return j.dump();
}

std::tuple<
    RowMatrixf,
    std::vector<std::string>,
    Eigen::VectorXf,
    std::vector<std::string>>
loadMotion(const std::string& gltfFilename) {
  const auto [motion, identity, fps] = momentum::loadMotion(gltfFilename);

  return {
      std::get<1>(motion).transpose(),
      std::get<0>(motion),
      std::get<1>(identity),
      std::get<0>(identity)};
}

std::vector<momentum::MarkerSequence> loadMarkersFromFile(
    const std::string& path,
    const bool mainSubjectOnly) {
  if (mainSubjectOnly) {
    const auto markerSequence = momentum::loadMarkersForMainSubject(path);
    if (markerSequence.has_value()) {
      return {markerSequence.value()};
    } else {
      return {};
    }
  } else {
    return momentum::loadMarkers(path);
  }
}

} // namespace pymomentum
