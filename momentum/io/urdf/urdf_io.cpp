/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/urdf/urdf_io.h"

#include <urdf_model/link.h>
#include <urdf_model/pose.h>
#include <urdf_parser/urdf_parser.h>

namespace momentum {

namespace {

constexpr size_t kMtoCM = 100.0;

template <typename S>
Vector3<S> toMomentumVector3(const urdf::Vector3& urdfVector3) {
  return Vector3<S>(urdfVector3.x, urdfVector3.y, urdfVector3.z);
}

template <typename S>
Quaternion<S> toMomentumQuaternion(const urdf::Rotation& urdfRotation) {
  return Quaternion<S>(urdfRotation.w, urdfRotation.x, urdfRotation.y, urdfRotation.z);
}

template <typename S>
TransformT<S> toMomentumTransform(const urdf::Pose& urdfPose) {
  return TransformT<S>(
      toMomentumVector3<S>(urdfPose.position), toMomentumQuaternion<S>(urdfPose.rotation));
}

template <typename T>
bool loadUrdfSkeletonRecursive(
    SkeletonT<T>& skeleton,
    size_t parentJointId,
    const urdf::ModelInterface* urdfModel,
    const urdf::Link* urdfLink) {
  MT_THROW_IF(urdfLink == nullptr, "URDF link is null.");

  auto* urdfJoint = urdfLink->parent_joint.get();
  MT_THROW_IF(
      parentJointId != kInvalidIndex && urdfJoint == nullptr,
      "URDF parent joint is null for a non root link ({}).",
      urdfLink->name);

  auto& joints = skeleton.joints;

  Joint joint;
  joint.name = urdfLink->name; // Use link name or joint name?
  joint.parent = parentJointId;

  // Set joint offset
  if (urdfJoint != nullptr) {
    const urdf::Pose& urdfPose = urdfJoint->parent_to_joint_origin_transform;
    joint.preRotation.setIdentity();
    joint.preRotation = toMomentumQuaternion<float>(urdfPose.rotation);
    joint.translationOffset = toMomentumVector3<float>(urdfPose.position) * kMtoCM;
  } else {
    joint.preRotation.setIdentity();
    joint.translationOffset.setZero();
  }

  const size_t jointId = joints.size();
  joints.push_back(joint);

  for (const auto& childLink : urdfLink->child_links) {
    if (!loadUrdfSkeletonRecursive(skeleton, jointId, urdfModel, childLink.get())) {
      return false;
    }
  }

  return true;
}

} // namespace

template <typename T>
SkeletonT<T> loadUrdfSkeleton(const filesystem::path& filepath) {
  urdf::ModelInterfaceSharedPtr urdfModel;

  try {
    urdfModel = urdf::parseURDFFile(filepath.string());
  } catch (const std::runtime_error& e) {
    MT_THROW("Failed to parse URDF file from: {}. Error: {}", filepath.string(), e.what());
  }

  const urdf::Link* root = urdfModel->getRoot().get();
  if (!root) {
    MT_THROW("Failed to parse URDF file from: {}. No root link found.", filepath.string());
  }

  if (root->name == "world") {
    // The URDF's specification documentation doesn't explicitly describe but the users uses "world"
    // as a reserved name for a link. If an URDF file contains a link with "world" name, then the
    // world link is regarded as a fixed body with no DOFs. Otherwise, a root link with a different
    // name, then it's regarded as a free floating (i.e., 6 DOFs) body.

    if (root->child_links.empty()) {
      MT_THROW(
          "Failed to parse URDF file from: {}. The world link should have at least one child link.",
          filepath.string());
    } else if (root->child_links.size() > 1) {
      MT_THROW(
          "Failed to parse URDF file from: {}. The world link should have only one child link.",
          filepath.string());
    }

    root = root->child_links[0].get();
  }

  SkeletonT<T> skeleton;

  if (!loadUrdfSkeletonRecursive(skeleton, kInvalidIndex, urdfModel.get(), root)) {
    MT_THROW("Failed to parse URDF file from: {}.", filepath.string());
  }

  return skeleton;
}

template SkeletonT<float> loadUrdfSkeleton(const filesystem::path& filepath);
template SkeletonT<double> loadUrdfSkeleton(const filesystem::path& filepath);

template <typename T>
CharacterT<T> loadUrdfCharacter(const filesystem::path& filepath) {
  const SkeletonT<float> skeleton = loadUrdfSkeleton<float>(filepath);

  // TODO: Parse parameter transform reflecting the URDF joint types
  const auto parameterTransform = ParameterTransform::identity(skeleton.getJointNames());

  // TODO: Parse joint limits
  // TODO: Parse collision geometries

  return CharacterT<T>(skeleton, parameterTransform);
}

template CharacterT<float> loadUrdfCharacter(const filesystem::path& filepath);
template CharacterT<double> loadUrdfCharacter(const filesystem::path& filepath);

} // namespace momentum
