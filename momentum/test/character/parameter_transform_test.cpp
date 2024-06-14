/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "momentum/character/character.h"
#include "momentum/character/inverse_parameter_transform.h"
#include "momentum/character/joint.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/math/constants.h"
#include "momentum/test/character/character_helpers.h"

using namespace momentum;

using Types = testing::Types<float, double>;

template <typename T>
struct Momentum_ParameterTransformTest : testing::Test {
  using Type = T;
};

TYPED_TEST_SUITE(Momentum_ParameterTransformTest, Types);

TEST(Momentum_ParameterTransform, Definitions) {
  EXPECT_TRUE(kParametersPerJoint == 7);
}

TYPED_TEST(Momentum_ParameterTransformTest, Empty) {
  using T = typename TestFixture::Type;

  const size_t nJoints = 3;
  const ParameterTransformT<T> xf = ParameterTransformT<T>::empty(nJoints * kParametersPerJoint);
  const JointParametersT<T> params = xf.apply(Eigen::VectorX<T>::Zero(0));
  EXPECT_EQ(nJoints * kParametersPerJoint, params.size());
  EXPECT_TRUE(params.v.isZero());
}

TYPED_TEST(Momentum_ParameterTransformTest, Functionality) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();
  ModelParametersT<T> parameters = ModelParametersT<T>::Zero(transform.numAllModelParameters());

  JointParametersT<T> jointParameters = transform.apply(parameters);

  EXPECT_EQ(parameters.size(), transform.numAllModelParameters());
  EXPECT_EQ(jointParameters.size(), transform.numJointParameters());

  {
    SCOPED_TRACE("Checking Zero Parameters");
    EXPECT_EQ(jointParameters.v.norm(), 0);
  }

  {
    SCOPED_TRACE("Checking Other Parameters");
    parameters.v << 1.0, 1.0, 1.0, pi<T>(), 0.0, -pi<T>(), 0.1, pi<T>(), pi<T>(), -pi<T>();
    jointParameters = transform.apply(parameters);
    EXPECT_EQ(jointParameters.size(), 21);

    VectorX<T> expectedJointParameters(21);
    expectedJointParameters << 1.0, 1.0, 1.0, pi<T>(), 0.0, -pi<T>(), 0.1, 0.0, 0.0, 0.0, pi<T>(),
        0.0, pi<T>() * 0.5, 0.0, 0.0, 0.0, 0.0, -pi<T>(), 0.0, pi<T>() * 0.5, 0.0;
    EXPECT_LE((jointParameters.v - expectedJointParameters).norm(), Eps<T>(1e-7f, 1e-15));
  }
}

TYPED_TEST(Momentum_ParameterTransformTest, Inverse) {
  using T = typename TestFixture::Type;

  const ParameterTransformT<T> transform = createTestCharacter().parameterTransform.cast<T>();
  const InverseParameterTransformT<T> invTransform(transform);

  {
    ModelParametersT<T> modelParameters =
        ModelParametersT<T>::Zero(transform.numAllModelParameters());
    JointParametersT<T> jointParameters = transform.apply(modelParameters);
    CharacterParametersT<T> modelParameters2 = invTransform.apply(jointParameters);
    EXPECT_TRUE(modelParameters2.pose.v.isZero());
    EXPECT_TRUE(modelParameters2.offsets.v.isZero());

    EXPECT_EQ(modelParameters.size(), invTransform.numAllModelParameters());
    EXPECT_EQ(jointParameters.size(), invTransform.numJointParameters());
  }

  {
    ModelParametersT<T> modelParameters =
        ModelParametersT<T>::Zero(transform.numAllModelParameters());
    modelParameters.v << 1.0, 1.0, 1.0, pi<T>(), 0.0, -pi<T>(), 0.1, pi<T>(), pi<T>(), -pi<T>();
    JointParametersT<T> jointParameters = transform.apply(modelParameters);
    {
      CharacterParametersT<T> modelParameters2 = invTransform.apply(jointParameters);
      EXPECT_EQ(transform.numAllModelParameters(), modelParameters2.pose.size());
      EXPECT_EQ(jointParameters.size(), modelParameters2.offsets.size());
      for (int i = 0; i < transform.numAllModelParameters(); ++i) {
        EXPECT_NEAR(modelParameters(i), modelParameters2.pose(i), Eps<T>(1e-7f, 1e-15));
      }
    }

    // add an offset, make sure the parameter inversion preserves it:
    for (int i = 0; i < jointParameters.size(); ++i) {
      jointParameters(i) += 0.05;
    }
    {
      CharacterParametersT<T> modelParameters3 = invTransform.apply(jointParameters);
      EXPECT_EQ(transform.numAllModelParameters(), modelParameters3.pose.size());
      EXPECT_EQ(jointParameters.size(), modelParameters3.offsets.size());

      JointParametersT<T> jointParameters2 = transform.apply(modelParameters3);
      EXPECT_EQ(jointParameters.size(), jointParameters2.size());
      for (int i = 0; i < transform.numAllModelParameters(); ++i) {
        EXPECT_NEAR(jointParameters(i), jointParameters2(i), Eps<T>(1e-7f, 1e-15));
      }
    }
  }
}

TYPED_TEST(Momentum_ParameterTransformTest, InverseWithOffsets) {
  using T = typename TestFixture::Type;

  ParameterTransformT<T> transform = createTestCharacter().parameterTransform.cast<T>();
  transform.offsets(3) = 1;
  transform.offsets(5) = -1;

  const InverseParameterTransformT<T> invTransform(transform);

  {
    ModelParametersT<T> modelParameters =
        ModelParametersT<T>::Zero(transform.numAllModelParameters());
    modelParameters.v << 1.0, 1.0, 1.0, pi<T>(), 0.0, -pi<T>(), 0.1, pi<T>(), pi<T>(), -pi<T>();
    JointParametersT<T> jointParameters = transform.apply(modelParameters);
    {
      CharacterParametersT<T> modelParameters2 = invTransform.apply(jointParameters);
      EXPECT_EQ(transform.numAllModelParameters(), modelParameters2.pose.size());
      EXPECT_EQ(jointParameters.size(), modelParameters2.offsets.size());
      for (int i = 0; i < transform.numAllModelParameters(); ++i) {
        EXPECT_NEAR(modelParameters(i), modelParameters2.pose(i), Eps<T>(5e-7f, 1e-15));
      }

      for (int i = 0; i < transform.offsets.size(); ++i) {
        EXPECT_NEAR(transform.offsets(i), modelParameters2.offsets(i), Eps<T>(1e-7f, 1e-15));
      }
    }

    // add an offset, make sure the parameter inversion preserves it:
    for (int i = 0; i < jointParameters.size(); ++i) {
      jointParameters(i) += 0.05;
    }

    {
      CharacterParametersT<T> modelParameters3 = invTransform.apply(jointParameters);
      EXPECT_EQ(transform.numAllModelParameters(), modelParameters3.pose.size());
      EXPECT_EQ(jointParameters.size(), modelParameters3.offsets.size());

      JointParametersT<T> jointParameters2 = transform.apply(modelParameters3);
      EXPECT_EQ(jointParameters.size(), jointParameters2.size());
      for (int i = 0; i < transform.numAllModelParameters(); ++i) {
        EXPECT_NEAR(jointParameters(i), jointParameters2(i), Eps<T>(1e-7f, 1e-15));
      }
    }
  }
}
