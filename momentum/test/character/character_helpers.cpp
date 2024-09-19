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

#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <limits>

namespace momentum {

namespace {

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

template <typename DerivedIndexMatrix, typename DerivedWeightMatrix>
void sortSkinWeightsRows(
    Eigen::MatrixBase<DerivedIndexMatrix>& indexMap,
    Eigen::MatrixBase<DerivedWeightMatrix>& weightMap) {
  using Index = typename DerivedIndexMatrix::Scalar;
  using Scalar = typename DerivedWeightMatrix::Scalar;

  for (int i = 0; i < indexMap.rows(); ++i) {
    std::vector<std::pair<Index, Scalar>> pairs;

    for (int j = 0; j < indexMap.cols(); ++j) {
      if (indexMap(i, j) == 0) { // Assuming 0 is the marker for unused elements
        break;
      }
      pairs.emplace_back(indexMap(i, j), weightMap(i, j));
    }

    // Sort pairs based on the first element of the pair (index value)
    std::sort(
        pairs.begin(),
        pairs.end(),
        [](const std::pair<Index, Scalar>& a, const std::pair<Index, Scalar>& b) {
          return a.first < b.first;
        });

    // Place sorted values back into the matrices
    for (Index k = 0; k < pairs.size(); ++k) {
      indexMap(i, k) = pairs[k].first;
      weightMap(i, k) = pairs[k].second;
    }
  }
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

namespace {

template <typename T>
Eigen::MatrixX<T> randomMatrix(Eigen::Index nRows, Eigen::Index nCols) {
  Eigen::MatrixX<T> result(nRows, nCols);
  for (Eigen::Index i = 0; i < nRows; ++i) {
    for (Eigen::Index j = 0; j < nCols; ++j) {
      result(i, j) = normal<T>(0, 1);
    }
  }
  return result;
}

MATCHER_P(FloatNearPointwise, tol, "Value mismatch") {
  for (int i = 0; i < std::get<0>(arg).size(); i++) {
    if (std::abs(std::get<0>(arg)[i] - std::get<1>(arg)[i]) > tol) {
      return false;
    }
  }
  return true;
}

MATCHER(IntExactPointwise, "Value mismatch") {
  for (int i = 0; i < std::get<0>(arg).size(); i++) {
    if (std::get<0>(arg)[i] != std::get<1>(arg)[i]) {
      return false;
    }
  }
  return true;
}

MATCHER(ElementsEq, "Elements mismatch") {
  return ::testing::get<0>(arg) == ::testing::get<1>(arg);
}

MATCHER(ElementsIsApprox, "Elements mismatch") {
  return ::testing::get<0>(arg).isApprox(::testing::get<1>(arg));
}

} // namespace

void compareMeshes(const Mesh_u& refMesh, const Mesh_u& mesh) {
  ASSERT_TRUE((refMesh && mesh));
  EXPECT_THAT(refMesh->vertices, testing::Pointwise(FloatNearPointwise(0.0001), mesh->vertices));
  EXPECT_THAT(refMesh->normals, testing::Pointwise(FloatNearPointwise(0.01), mesh->normals));
  EXPECT_THAT(refMesh->faces, testing::Pointwise(IntExactPointwise(), mesh->faces));
  EXPECT_THAT(refMesh->colors, testing::Pointwise(IntExactPointwise(), mesh->colors));
  EXPECT_THAT(refMesh->texcoords, testing::Pointwise(FloatNearPointwise(0.0001), mesh->texcoords));
  EXPECT_THAT(
      refMesh->confidence, testing::Pointwise(testing::DoubleNear(0.0001), mesh->confidence));
  EXPECT_THAT(
      refMesh->texcoord_faces, testing::Pointwise(IntExactPointwise(), mesh->texcoord_faces));
}

void compareLocators(const LocatorList& refLocators, const LocatorList& locators) {
  EXPECT_EQ(refLocators.size(), locators.size());
  auto sortedRefLocators = refLocators;
  auto sortedLocators = locators;
  auto compareLocators = [](const Locator& l1, const Locator& l2) {
    return l1.parent != l2.parent ? l1.parent < l2.parent : l1.name < l2.name;
  };
  std::sort(sortedRefLocators.begin(), sortedRefLocators.end(), compareLocators);
  std::sort(sortedLocators.begin(), sortedLocators.end(), compareLocators);
  EXPECT_THAT(sortedRefLocators, testing::Pointwise(ElementsEq(), sortedLocators));
}

void compareCollisionGeometry(
    const CollisionGeometry_u& refCollision,
    const CollisionGeometry_u& collision) {
  if (refCollision == nullptr)
    ASSERT_EQ(collision, nullptr);
  else {
    ASSERT_NE(collision, nullptr);
    EXPECT_EQ(refCollision->size(), collision->size());
    auto sortedRefCollision = *refCollision;
    auto sortedCollision = *collision;
    auto compareCollisions = [](const TaperedCapsule& l1, const TaperedCapsule& l2) {
      if (l1.parent != l2.parent)
        return l1.parent < l2.parent;
      if (l1.length != l2.length)
        return l1.length < l2.length;
      if (!l1.radius.isApprox(l2.radius))
        return l1.radius.x() != l2.radius.x() ? l1.radius.x() < l2.radius.x()
                                              : l1.radius.y() < l2.radius.y();
      const auto t1 = l1.transformation.matrix();
      const auto t2 = l2.transformation.matrix();
      EXPECT_EQ(t1.size(), t2.size());
      for (auto i = 0; i < t1.size(); i++) {
        if (std::abs(t1.data()[i] - t2.data()[i]) > std::numeric_limits<float>::epsilon()) {
          return t1.data()[i] < t2.data()[i];
        }
      }
      return false;
    };
    std::sort(sortedRefCollision.begin(), sortedRefCollision.end(), compareCollisions);
    std::sort(sortedCollision.begin(), sortedCollision.end(), compareCollisions);
    for (size_t i = 0; i < sortedCollision.size(); ++i) {
      const auto& collA = sortedRefCollision[i];
      const auto& collB = sortedCollision[i];

      EXPECT_TRUE(collA.isApprox(collB)) << "Collision geometry mismatch at index " << i << ":\n"
                                         << "- refCollision:\n"
                                         << "  - radius_0 : " << collA.radius.x() << "\n"
                                         << "  - radius_1 : " << collA.radius.y() << "\n"
                                         << "  - length   : " << collA.length << "\n"
                                         << "  - parent   : " << collA.parent << "\n"
                                         << "  - transform:\n"
                                         << collA.transformation.matrix() << "\n"
                                         << "- collision:\n"
                                         << "  - radius_0 : " << collB.radius.x() << "\n"
                                         << "  - radius_1 : " << collB.radius.y() << "\n"
                                         << "  - length   : " << collB.length << "\n"
                                         << "  - parent   : " << collB.parent << "\n"
                                         << "  - transform:\n"
                                         << collB.transformation.matrix() << std::endl;
    }
  }
}

void compareChars(const Character& refChar, const Character& character, const bool withMesh) {
  const auto& refJoints = refChar.skeleton.joints;
  const auto& joints = character.skeleton.joints;
  ASSERT_EQ(refJoints.size(), joints.size());
  for (size_t i = 0; i < refJoints.size(); ++i) {
    EXPECT_TRUE(refJoints[i].isApprox(joints[i]))
        << "Joint " << i << " is not equal:\n"
        << "- refJoint:\n"
        << "  - name: " << refJoints[i].name << "\n"
        << "  - parent: " << refJoints[i].parent << "\n"
        << "  - preRotation: " << refJoints[i].preRotation.coeffs().transpose() << "\n"
        << "  - translationOffset: " << refJoints[i].translationOffset.transpose() << "\n"
        << "- joint:\n"
        << "  - name: " << joints[i].name << "\n"
        << "  - parent: " << joints[i].parent << "\n"
        << "  - preRotation: " << joints[i].preRotation.coeffs().transpose() << "\n"
        << "  - translationOffset: " << joints[i].translationOffset.transpose() << "\n";
  }
  ASSERT_TRUE(refChar.parameterTransform.isApprox(character.parameterTransform));
  EXPECT_THAT(refChar.parameterLimits, testing::Pointwise(ElementsEq(), character.parameterLimits));
  compareLocators(refChar.locators, character.locators);
  compareCollisionGeometry(refChar.collision, character.collision);
  ASSERT_EQ(refChar.inverseBindPose.size(), character.inverseBindPose.size());
  for (size_t i = 0; i < refChar.inverseBindPose.size(); ++i) {
    EXPECT_TRUE(refChar.inverseBindPose[i].isApprox(character.inverseBindPose[i], 1e-4f))
        << "InverseBindPose " << i << " is not equal:\n"
        << "- Expected:\n"
        << refChar.inverseBindPose[i].matrix() << "\n"
        << "- Actual  :\n"
        << character.inverseBindPose[i].matrix() << std::endl;
  }
  EXPECT_EQ(refChar.jointMap, character.jointMap);

  if (withMesh) {
    auto ptrValEq = [](auto& l, auto& r) {
      if (l == nullptr && r == nullptr) {
        return true;
      }
      if (l == nullptr) {
        return (r == nullptr);
      }
      if (r == nullptr) {
        return false;
      }
      return true;
    };

    compareMeshes(refChar.mesh, character.mesh);

    auto refCharIndices = refChar.skinWeights->index;
    auto refCharWeights = refChar.skinWeights->weight;
    sortSkinWeightsRows(refCharIndices, refCharWeights);

    auto charIndices = refChar.skinWeights->index;
    auto charWeights = refChar.skinWeights->weight;
    sortSkinWeightsRows(charIndices, charWeights);

    ASSERT_EQ(refCharIndices.rows(), charIndices.rows());
    ASSERT_EQ(refCharIndices.cols(), charIndices.cols());
    for (auto i = 0u; i < refCharIndices.rows(); ++i) {
      EXPECT_EQ(refCharIndices.row(i), charIndices.row(i))
          << "SkinWeights index row " << i << " mismatch\n";
    }
    ASSERT_LT((refCharWeights - charWeights).lpNorm<Eigen::Infinity>(), 1e-4f);
    ASSERT_TRUE(ptrValEq(refChar.skinWeights, character.skinWeights));

    auto ptrValIsApprox = [](auto& l, auto& r) {
      if (l == nullptr)
        return (r == nullptr);
      if (r == nullptr)
        return false;
      return l->isApprox(*r);
    };
    ASSERT_TRUE(ptrValIsApprox(refChar.poseShapes, character.poseShapes));
    ASSERT_TRUE(ptrValIsApprox(refChar.blendShape, character.blendShape));
  }
}

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
