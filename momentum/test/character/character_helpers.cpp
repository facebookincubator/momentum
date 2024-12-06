/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/test/character/character_helpers.h"

#include "momentum/character/blend_shape.h"
#include "momentum/character/character.h"
#include "momentum/character/collision_geometry.h"
#include "momentum/character/linear_skinning.h"
#include "momentum/character/locator.h"
#include "momentum/character/parameter_transform.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skin_weights.h"
#include "momentum/common/checks.h"
#include "momentum/math/mesh.h"
#include "momentum/math/mppca.h"
#include "momentum/math/random.h"

#include <limits>

namespace momentum {

namespace {

template <typename T>
Eigen::MatrixX<T> randomMatrix(Eigen::Index nRows, Eigen::Index nCols) {
  return normal<MatrixX<T>>(nRows, nCols, 0, 1);
}

[[nodiscard]] Skeleton createDefaultSkeleton(size_t numJoints) {
  Skeleton result;
  Joint joint;
  joint.name = "root";
  joint.parent = kInvalidIndex;
  joint.preRotation = Eigen::Quaternionf::Identity();
  joint.translationOffset = Eigen::Vector3f::Zero();
  result.joints.push_back(joint);

  for (int i = 1; i < numJoints; ++i) {
    joint.name = "joint" + std::to_string(i);
    joint.parent = i - 1;
    joint.translationOffset = Eigen::Vector3f::UnitY();
    result.joints.push_back(joint);
  }

  return result;
}

// hardcoded locators for test character
[[nodiscard]] LocatorList createDefaultLocatorList(size_t numJoints) {
  LocatorList result;
  for (int i = 0; i < numJoints; ++i) {
    Locator loc;
    loc.parent = i;
    loc.offset = Eigen::Vector3f::Random();
    loc.name = "l" + std::to_string(i);
    result.push_back(loc);
  }
  return result;
}

// Dummy collision geometry.
CollisionGeometry createDefaultCollisionGeometry(size_t numJoints) {
  CollisionGeometry result(numJoints);
  for (int i = 0; i < numJoints; ++i) {
    result[i].parent = i;
    result[i].length = 1.0f;
    result[i].radius =
        Eigen::Vector2f(1.0 + i / float(numJoints), 1.0 + (i + 1) / float(numJoints));
  }
  return result;
}

[[nodiscard]] ParameterTransform createDefaultParameterTransform(size_t numJoints) {
  ParameterTransform result;
  const auto numJointParameters = numJoints * kParametersPerJoint;
  result.name = {
      {"root_tx",
       "root_ty",
       "root_tz",
       "root_rx",
       "root_ry",
       "root_rz",
       "scale_global",
       "joint1_rx",
       "shared_rz"}};

  const size_t rxStart = result.name.size();
  for (size_t iJoint = 2; iJoint < numJoints; ++iJoint) {
    result.name.push_back("joint" + std::to_string(iJoint) + "_rx");
  }

  result.offsets = Eigen::VectorXf::Zero(numJointParameters);
  result.transform.resize(numJointParameters, static_cast<int>(result.name.size()));

  std::vector<Eigen::Triplet<float>> triplets;
  triplets.push_back(Eigen::Triplet<float>(0 * kParametersPerJoint + 0, 0, 1.0f)); // root_tx
  triplets.push_back(Eigen::Triplet<float>(0 * kParametersPerJoint + 1, 1, 1.0f)); // root_ty
  triplets.push_back(Eigen::Triplet<float>(0 * kParametersPerJoint + 2, 2, 1.0f)); // root_tz
  triplets.push_back(Eigen::Triplet<float>(0 * kParametersPerJoint + 3, 3, 1.0f)); // root_rx
  triplets.push_back(Eigen::Triplet<float>(0 * kParametersPerJoint + 4, 4, 1.0f)); // root_ry
  triplets.push_back(Eigen::Triplet<float>(0 * kParametersPerJoint + 5, 5, 1.0f)); // root_rz
  triplets.push_back(Eigen::Triplet<float>(0 * kParametersPerJoint + 6, 6, 1.0f)); // root_sc
  triplets.push_back(Eigen::Triplet<float>(1 * kParametersPerJoint + 3, 7, 1.0f)); // joint1_rx
  triplets.push_back(Eigen::Triplet<float>(1 * kParametersPerJoint + 5, 8, 0.5f)); // shared_rz
  triplets.push_back(Eigen::Triplet<float>(2 * kParametersPerJoint + 5, 8, 0.5f)); // shared_rz
  for (size_t iJoint = 2; iJoint < numJoints; ++iJoint) {
    triplets.push_back(Eigen::Triplet<float>(
        iJoint * kParametersPerJoint + 3, rxStart + iJoint - 2, 1.0f)); // joint1_rx
  }

  result.transform.setFromTriplets(triplets.begin(), triplets.end());
  result.activeJointParams = result.computeActiveJointParams();
  return result;
}

// The mesh is a made by a few vertices on the line segment from (1,0,0) to (1,1,0)
// and two dummy faces.
std::tuple<Mesh, SkinWeights> createDefaultMesh() {
  std::tuple<Mesh, SkinWeights> result;
  auto& mesh = std::get<0>(result);
  auto& skin = std::get<1>(result);

  constexpr size_t kNumSegments = 25;

  skin.index.resize(2 * kNumSegments, Eigen::NoChange);
  skin.weight.resize(2 * kNumSegments, Eigen::NoChange);
  skin.index.setZero();
  skin.weight.setZero();

  for (size_t i = 0; i < kNumSegments; i++) {
    const float fraction = static_cast<float>(i) / static_cast<float>(kNumSegments - 1);
    mesh.vertices.emplace_back(0.0f, fraction, 0.0f);
    mesh.vertices.emplace_back(1.0f, fraction, 0.0f);

    for (int k = 0; k < 2; ++k) {
      skin.index(2 * i + k, 0) = 0;
      skin.index(2 * i + k, 1) = 1;
      skin.weight(2 * i + k, 0) = 1.0f - fraction;
      skin.weight(2 * i + k, 1) = fraction;
    }
  }

  // Reorder the weights:
  for (Eigen::Index i = 0; i < skin.index.rows(); ++i) {
    if (skin.weight(i, 1) > skin.weight(i, 0)) {
      std::swap(skin.index(i, 1), skin.index(i, 0));
      std::swap(skin.weight(i, 1), skin.weight(i, 0));
    }
  }

  for (int i = 0; i < (kNumSegments - 1); ++i) {
    // 2*i+0   2*i+2
    // 2*i+1   2*i+3
    mesh.faces.emplace_back(2 * i + 0, 2 * i + 2, 2 * i + 1);
    mesh.faces.emplace_back(2 * i + 1, 2 * i + 2, 2 * i + 3);
  }

  mesh.colors.resize(2 * kNumSegments, Vector3b(0, 0, 0));
  mesh.confidence.resize(2 * kNumSegments, 1.0f);

  mesh.updateNormals();

  return result;
}

ParameterLimits createDefaultParameterLimits() {
  ParameterLimits lm(1);
  lm[0].type = MinMax;
  lm[0].weight = 1.0f;
  lm[0].data.minMax.limits = Vector2f(-0.1, 0.1);
  lm[0].data.minMax.parameterIndex = 0;
  return lm;
}

} // namespace

template <typename T>
CharacterT<T> createTestCharacter(size_t numJoints) {
  MT_CHECK(numJoints >= 3, "The number of joints '{}' should be equal to 3 or greater.", numJoints);
  auto [mesh, skin] = createDefaultMesh();
  auto collisionGeometry = createDefaultCollisionGeometry(numJoints);
  return CharacterT<T>(
      createDefaultSkeleton(numJoints),
      createDefaultParameterTransform(numJoints),
      createDefaultParameterLimits(),
      createDefaultLocatorList(numJoints),
      &mesh,
      &skin,
      &collisionGeometry);
}

template CharacterT<float> createTestCharacter(size_t numJoints);
template CharacterT<double> createTestCharacter(size_t numJoints);

template <typename T>
CharacterT<T> withTestBlendShapes(const CharacterT<T>& character) {
  MT_CHECK(character.mesh);

  const int nShapes = 5;
  auto blendShapes = std::make_shared<BlendShape>(character.mesh->vertices, nShapes);
  blendShapes->setShapeVectors(randomMatrix<float>(3 * character.mesh->vertices.size(), nShapes));
  return character.withBlendShape(blendShapes, nShapes);
}

template CharacterT<float> withTestBlendShapes(const CharacterT<float>& character);
template CharacterT<double> withTestBlendShapes(const CharacterT<double>& character);

template <typename T>
CharacterT<T> withTestFaceExpressionBlendShapes(const CharacterT<T>& character) {
  MT_CHECK(character.mesh);

  const int nShapes = 5;
  auto blendShapes = std::make_shared<BlendShapeBase>();
  blendShapes->setShapeVectors(randomMatrix<float>(3 * character.mesh->vertices.size(), nShapes));

  return character.withFaceExpressionBlendShape(blendShapes, nShapes);
}

template CharacterT<float> withTestFaceExpressionBlendShapes(const CharacterT<float>& character);
template CharacterT<double> withTestFaceExpressionBlendShapes(const CharacterT<double>& character);

template <typename T>
std::shared_ptr<const MppcaT<T>> createDefaultPosePrior() {
  std::vector<std::string> paramNames = {"joint1_rx", "shared_rz", "joint2_rx"};

  // Input dimensionality:
  const int d = paramNames.size();

  // Number of mixtures:
  const int p = 2;

  Eigen::VectorX<T> pi = randomMatrix<T>(p, 1).array().abs();
  pi /= pi.sum();
  Eigen::MatrixX<T> mu = randomMatrix<T>(p, d);
  Eigen::VectorX<T> sigma2 = randomMatrix<T>(p, 1).array().square();

  std::vector<Eigen::MatrixX<T>> W(p);
  const int q = 2;
  for (int i = 0; i < p; ++i) {
    W[i] = randomMatrix<T>(d, q);
  }

  auto result = std::make_shared<MppcaT<T>>();
  result->set(pi, mu, W, sigma2);
  result->names = paramNames;
  return result;
}

template std::shared_ptr<const MppcaT<float>> createDefaultPosePrior();
template std::shared_ptr<const MppcaT<double>> createDefaultPosePrior();

} // namespace momentum
