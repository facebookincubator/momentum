/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <momentum/character/parameter_limits.h>
#include <momentum/io/skeleton/parameter_limits_io.h>
#include "momentum/test/character/character_helpers.h"

using namespace momentum;

namespace {

Character createCharacterWithLimits() {
  auto testCharacter = createTestCharacter();

  ParameterLimits limits;

  {
    ParameterLimit limit;
    limit.type = MinMax;
    limit.weight = 1.5f;
    limit.data.minMax.limits = Vector2f(-0.2, 0.1);
    limit.data.minMax.parameterIndex = 1;
    limits.push_back(limit);
  }

  {
    ParameterLimit limit;
    limit.type = LimitType::MinMaxJoint;
    limit.weight = 2.0f;
    limit.data.minMaxJoint.limits = Vector2f(-0.3, 0.0);
    limit.data.minMaxJoint.jointIndex = 1;
    limit.data.minMaxJoint.jointParameter = 2;
    limits.push_back(limit);
  }

  {
    ParameterLimit limit;
    limit.type = LimitType::MinMaxJointPassive;
    limit.weight = 2.0f;
    limit.data.minMaxJoint.limits = Vector2f(-0.0, 0.5);
    limit.data.minMaxJoint.jointIndex = 2;
    limit.data.minMaxJoint.jointParameter = 4;
    limits.push_back(limit);
  }

  {
    ParameterLimit limit;
    limit.type = LimitType::Ellipsoid;
    limit.weight = 4.0;
    limit.data.ellipsoid.ellipsoid = limit.data.ellipsoid.ellipsoid = Affine3f::Identity();
    limit.data.ellipsoid.ellipsoid.translation() = Eigen::Vector3f(2, 3, 4);
    const Vector3f eulerXYZ = Vector3f(M_PI / 2.0, M_PI / 4.0, -M_PI / 3.0);
    limit.data.ellipsoid.ellipsoid.linear() =
        eulerXYZToRotationMatrix(eulerXYZ, EulerConvention::Extrinsic) *
        Eigen::Scaling(0.8f, 0.9f, 1.3f);
    limit.data.ellipsoid.ellipsoidInv = limit.data.ellipsoid.ellipsoid.inverse();
    limit.data.ellipsoid.offset = Eigen::Vector3f(1, 2, 3);
    limit.data.ellipsoid.ellipsoidParent = 2;
    limit.data.ellipsoid.parent = 1;
    limits.push_back(limit);
  }

  {
    ParameterLimit limit;
    limit.type = LimitType::Ellipsoid;
    limit.weight = 5.0;
    limit.data.ellipsoid.ellipsoid = limit.data.ellipsoid.ellipsoid = Affine3f::Identity();
    limit.data.ellipsoid.ellipsoid.translation() = Eigen::Vector3f(2, 3, 4);
    const Vector3f eulerXYZ = Vector3f(-M_PI / 4.0, 2.0f * M_PI / 4.0, M_PI / 3.0);
    limit.data.ellipsoid.ellipsoid.linear() =
        eulerXYZToRotationMatrix(eulerXYZ, EulerConvention::Extrinsic) *
        Eigen::Scaling(2.0f, 0.4f, 0.6f);
    limit.data.ellipsoid.ellipsoidInv = limit.data.ellipsoid.ellipsoid.inverse();
    limit.data.ellipsoid.offset = Eigen::Vector3f(2, 1, 4);
    limit.data.ellipsoid.ellipsoidParent = 0;
    limit.data.ellipsoid.parent = 2;
    limits.push_back(limit);
  }

  {
    // ellipsoid constraints are the hardest to get right so let's include two of them:
    ParameterLimit limit;
    limit.type = LimitType::Linear;
    limit.weight = 2.0;
    limit.data.linear.scale = 3.0;
    limit.data.linear.offset = 2.0;
    limit.data.linear.referenceIndex = 2;
    limit.data.linear.targetIndex = 1;
    limits.push_back(limit);
  }

  return {testCharacter.skeleton, testCharacter.parameterTransform, limits};
}

void validateParameterLimitsSame(const ParameterLimits& limits1, const ParameterLimits& limits2) {
  ASSERT_EQ(limits1.size(), limits2.size());

  for (size_t i = 0; i < limits1.size(); ++i) {
    const auto& l1 = limits1[i];
    const auto& l2 = limits2[i];

    EXPECT_NEAR(l1.weight, l2.weight, 1e-4f);
    EXPECT_EQ(l1.type, l2.type);

    switch (l1.type) {
      case LimitType::MinMax:
        EXPECT_NEAR(l1.data.minMax.limits.x(), l2.data.minMax.limits.x(), 1e-4f);
        EXPECT_NEAR(l1.data.minMax.limits.y(), l2.data.minMax.limits.y(), 1e-4f);
        EXPECT_EQ(l1.data.minMax.parameterIndex, l2.data.minMax.parameterIndex);
        break;
      case LimitType::MinMaxJoint:
      case LimitType::MinMaxJointPassive:
        EXPECT_NEAR(l1.data.minMaxJoint.limits.x(), l2.data.minMaxJoint.limits.x(), 1e-4f);
        EXPECT_NEAR(l1.data.minMaxJoint.limits.y(), l2.data.minMaxJoint.limits.y(), 1e-4f);
        EXPECT_EQ(l1.data.minMaxJoint.jointIndex, l2.data.minMaxJoint.jointIndex);
        EXPECT_EQ(l1.data.minMaxJoint.jointParameter, l2.data.minMaxJoint.jointParameter);
        break;
      case LimitType::Linear:
        EXPECT_NEAR(l1.data.linear.offset, l2.data.linear.offset, 1e-4f);
        EXPECT_NEAR(l1.data.linear.scale, l2.data.linear.scale, 1e-4f);
        EXPECT_EQ(l1.data.linear.targetIndex, l2.data.linear.targetIndex);
        EXPECT_EQ(l1.data.linear.referenceIndex, l2.data.linear.referenceIndex);
        break;
      case LimitType::Ellipsoid:
        EXPECT_LE(
            (l1.data.ellipsoid.ellipsoid.matrix() - l2.data.ellipsoid.ellipsoid.matrix())
                .lpNorm<Eigen::Infinity>(),
            1e-4f);
        EXPECT_LE(
            (l1.data.ellipsoid.ellipsoidInv.matrix() - l2.data.ellipsoid.ellipsoidInv.matrix())
                .lpNorm<Eigen::Infinity>(),
            1e-4f);
        EXPECT_EQ(l1.data.ellipsoid.ellipsoidParent, l2.data.ellipsoid.ellipsoidParent);
        EXPECT_EQ(l1.data.ellipsoid.parent, l2.data.ellipsoid.parent);
        EXPECT_LE((l1.data.ellipsoid.offset - l2.data.ellipsoid.offset).norm(), 1e-4f);
        break;
    }
  }
}

} // namespace

TEST(IoCharacterTest, ParameterLimits_RoundTrip) {
  const Character character = createCharacterWithLimits();

  const std::string limitsStr = writeParameterLimits(
      character.parameterLimits, character.skeleton, character.parameterTransform);
  std::cout << limitsStr << "\n";
  const auto limits2 =
      parseParameterLimits(limitsStr, character.skeleton, character.parameterTransform);
  validateParameterLimitsSame(character.parameterLimits, limits2);
}
