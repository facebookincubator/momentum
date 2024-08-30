/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <chrono>
#include <cstring>
#include <random>

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include "momentum/character/character.h"
#include "momentum/character/linear_skinning.h"
#include "momentum/character/parameter_transform.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character/skin_weights.h"
#include "momentum/character_solver/aim_error_function.h"
#include "momentum/character_solver/collision_error_function.h"
#include "momentum/character_solver/fixed_axis_error_function.h"
#include "momentum/character_solver/fwd.h"
#include "momentum/character_solver/limit_error_function.h"
#include "momentum/character_solver/model_parameters_error_function.h"
#include "momentum/character_solver/normal_error_function.h"
#include "momentum/character_solver/orientation_error_function.h"
#include "momentum/character_solver/plane_error_function.h"
#include "momentum/character_solver/pose_prior_error_function.h"
#include "momentum/character_solver/position_error_function.h"
#include "momentum/character_solver/state_error_function.h"
#include "momentum/character_solver/vertex_error_function.h"
#include "momentum/math/constants.h"
#include "momentum/math/mesh.h"
#include "momentum/math/random.h"
#include "momentum/math/utility.h"
#include "momentum/test/character/character_helpers.h"
#include "momentum/test/character_solver/error_function_helpers.h"

using namespace momentum;

using Types = testing::Types<float, double>;

TYPED_TEST_SUITE(Momentum_ErrorFunctionsTest, Types);

TYPED_TEST(Momentum_ErrorFunctionsTest, LimitError_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  ParameterLimits lm(1);
  LimitErrorFunctionT<T> errorFunction(skeleton, character.parameterTransform);

  // TODO: None of these work right at the moment due to precision issues with numerical gradients,
  // need to fix code to use double

  {
    SCOPED_TRACE("Limit MinMax Test");
    lm[0].type = MINMAX;
    lm[0].weight = 1.0;
    lm[0].data.minMax.limits = Vector2f(-0.1, 0.1);
    lm[0].data.minMax.parameterIndex = 0;
    errorFunction.setLimits(lm);
    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        skeleton,
        transform,
        Eps<T>(1e-7f, 1e-15));
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters = VectorX<T>::Random(transform.numAllModelParameters());
      TEST_GRADIENT_AND_JACOBIAN(
          T, &errorFunction, parameters, skeleton, transform, Eps<T>(1e-2f, 1e-11));
    }
  }

  {
    SCOPED_TRACE("Limit MinMax Joint Test");
    lm[0].type = MINMAX_JOINT;
    lm[0].weight = 1.0;
    lm[0].data.minMaxJoint.limits = Vector2f(-0.1, 0.1);
    lm[0].data.minMaxJoint.jointIndex = 2;
    lm[0].data.minMaxJoint.jointParameter = 5;
    errorFunction.setLimits(lm);
    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        skeleton,
        transform,
        Eps<T>(1e-7f, 1e-15));
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters = VectorX<T>::Random(transform.numAllModelParameters());
      TEST_GRADIENT_AND_JACOBIAN(
          T, &errorFunction, parameters, skeleton, transform, Eps<T>(5e-3f, 1e-10));
    }
  }

  {
    SCOPED_TRACE("Limit LinearTest");
    lm[0].type = LINEAR;
    lm[0].weight = 1.0;
    lm[0].data.linear.referenceIndex = 0;
    lm[0].data.linear.targetIndex = 5;
    lm[0].data.linear.scale = 0.25;
    lm[0].data.linear.offset = 0.25;
    errorFunction.setLimits(lm);
    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        skeleton,
        transform,
        Eps<T>(1e-3f, 5e-12));
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters = VectorX<T>::Random(transform.numAllModelParameters());
      TEST_GRADIENT_AND_JACOBIAN(
          T, &errorFunction, parameters, skeleton, transform, Eps<T>(1e-2f, 1e-10));
    }
  }

  //{
  //    SCOPED_TRACE("Limit Ellipsoid Test");
  //    lm[0].type = ELLIPSOID;
  //    lm[0].weight = 1.0f;
  //    lm[0].data.ellipsoid.parent = 2;
  //    lm[0].data.ellipsoid.ellipsoidParent = 0;
  //    lm[0].data.ellipsoid.offset = Vector3f(0, -1, 0);
  //    lm[0].data.ellipsoid.ellipsoid = Affine3f::Identity();
  //    lm[0].data.ellipsoid.ellipsoid.translation() = Vector3f(0.5f, 0.5f, 0.5f);
  //    lm[0].data.ellipsoid.ellipsoid.linear() = (Quaternionf(Eigen::AngleAxisf(0.1f,
  //    Vector3f::UnitZ())) *
  //                                            Quaternionf(Eigen::AngleAxisf(0.2f,
  //                                            Vector3f::UnitY())) *
  //                                            Quaternionf(Eigen::AngleAxisf(0.3f,
  //                                            Vector3f::UnitX()))).toRotationMatrix() *
  //                                            Scaling(2.0f, 1.5f, 0.5f);
  //    lm[0].data.ellipsoid.ellipsoidInv = lm[0].data.ellipsoid.ellipsoid.inverse();

  //    errorFunction.setLimits(lm);
  //    parameters.setZero();
  //    testGradientAndJacobian(&errorFunction, parameters, skeleton, transform);
  //    for (size_t i = 0; i < 3; i++)
  //    {
  //        parameters = VectorXd::Random(transform.numAllModelParameters());
  //        testGradientAndJacobian(&errorFunction, parameters, skeleton, transform);
  //    }
  //}
}

TYPED_TEST(Momentum_ErrorFunctionsTest, StateError_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  StateErrorFunctionT<T> errorFunction(skeleton, character.parameterTransform);
  {
    SCOPED_TRACE("State Test");
    SkeletonStateT<T> reference(transform.bindPose(), skeleton);
    errorFunction.setTargetState(reference);
    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        skeleton,
        transform,
        Eps<T>(2e-5f, 1e-3));
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters = VectorX<T>::Random(transform.numAllModelParameters()) * 0.25;
      TEST_GRADIENT_AND_JACOBIAN(
          T, &errorFunction, parameters, skeleton, transform, Eps<T>(1e-2f, 5e-6));
    }
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, ModelParametersError_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  ModelParametersErrorFunctionT<T> errorFunction(skeleton, character.parameterTransform);
  {
    SCOPED_TRACE("Motion Test");
    SkeletonStateT<T> reference(transform.bindPose(), skeleton);
    VectorX<T> weights = VectorX<T>::Ones(transform.numAllModelParameters());
    weights(0) = 4.0;
    weights(1) = 5.0;
    weights(2) = 0.0;
    errorFunction.setTargetParameters(
        ModelParametersT<T>::Zero(transform.numAllModelParameters()), weights);
    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        skeleton,
        transform,
        Eps<T>(1e-7f, 1e-15));
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters = VectorX<T>::Random(transform.numAllModelParameters()) * 0.25;
      TEST_GRADIENT_AND_JACOBIAN(
          T, &errorFunction, parameters, skeleton, transform, Eps<T>(5e-3f, 1e-10));
    }
  }
}

// Show an example of regularizing the blend weights:
TYPED_TEST(Momentum_ErrorFunctionsTest, ModelParametersError_RegularizeBlendWeights) {
  using T = typename TestFixture::Type;

  const Character character = withTestBlendShapes(createTestCharacter());
  const ModelParametersT<T> modelParams =
      0.25 * VectorX<T>::Random(character.parameterTransform.numAllModelParameters());
  ModelParametersErrorFunctionT<T> errorFunction(
      character, character.parameterTransform.getBlendShapeParameters());
  EXPECT_GT(
      errorFunction.getError(
          modelParams,
          SkeletonStateT<T>(
              character.parameterTransform.cast<T>().apply(modelParams), character.skeleton)),
      0);
  TEST_GRADIENT_AND_JACOBIAN(
      T,
      &errorFunction,
      modelParams,
      character.skeleton,
      character.parameterTransform.cast<T>(),
      Eps<T>(1e-3f, 1e-10),
      Eps<T>(1e-7f, 1e-7),
      true);

  ModelParametersErrorFunctionT<T> errorFunction2(
      character, character.parameterTransform.getScalingParameters());
  TEST_GRADIENT_AND_JACOBIAN(
      T,
      &errorFunction2,
      modelParams,
      character.skeleton,
      character.parameterTransform.cast<T>(),
      Eps<T>(1e-3f, 5e-12),
      Eps<T>(1e-6f, 1e-7),
      true);
}

TYPED_TEST(Momentum_ErrorFunctionsTest, PosePriorError_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  PosePriorErrorFunctionT<T> errorFunction(
      skeleton, character.parameterTransform, createDefaultPosePrior<T>());

  TEST_GRADIENT_AND_JACOBIAN(
      T,
      &errorFunction,
      ModelParametersT<T>::Zero(transform.numAllModelParameters()),
      skeleton,
      transform,
      Eps<T>(1e-1f, 1e-10),
      Eps<T>(1e-5f, 5e-6));

  for (size_t i = 0; i < 3; i++) {
    ModelParametersT<T> parameters = VectorX<T>::Random(transform.numAllModelParameters());
    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        parameters,
        skeleton,
        transform,
        Eps<T>(1e-1f, 1e-9),
        Eps<T>(5e-5f, 5e-6));
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, TestSkinningErrorFunction) {
  using T = typename TestFixture::Type;

  // this unit tests checks the accuracy of our linear skinning constraint accuracy
  // against our simplified approximation that's way faster.
  // we expect our gradients to be within 10% of the true gradients of the mesh

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();
  const auto mesh = character.mesh->cast<T>();
  const auto& skin = *character.skinWeights;
  VectorX<T> parameters = VectorX<T>::Zero(transform.numAllModelParameters());

  VectorX<T> gradient = VectorX<T>::Zero(transform.numAllModelParameters());

  // create constraints
  std::vector<PositionDataT<T>> cl;
  PositionErrorFunctionT<T> errorFunction(skeleton, character.parameterTransform);
  SkeletonStateT<T> bindState(transform.apply(parameters), skeleton);
  SkeletonStateT<T> state(transform.apply(parameters), skeleton);
  TransformationListT<T> bindpose;
  for (const auto& js : bindState.jointState)
    bindpose.push_back(js.transformation.inverse());

  {
    SCOPED_TRACE("Skinning mesh constraint test");

    for (size_t vi = 0; vi < mesh.vertices.size(); vi++) {
      const Eigen::Vector3<T> target = mesh.vertices[vi];

      // add vertex to constraint list
      cl.clear();
      for (size_t si = 0; si < kMaxSkinJoints; si++) {
        if (skin.weight(vi, si) == 0.0)
          continue;
        const auto parent = skin.index(vi, si);
        cl.push_back(PositionDataT<T>(
            (target - bindState.jointState[parent].translation()),
            target,
            parent,
            skin.weight(vi, si)));
      }
      errorFunction.setConstraints(cl);

      std::vector<Vector3<T>> v = applySSD(bindpose, skin, mesh.vertices, bindState);

      // check position of skinning
      EXPECT_LE((v[vi] - target).norm(), Eps<T>(1e-7f, 1e-7));

      // check error
      gradient.setZero();
      T gradientError = errorFunction.getGradient(parameters, bindState, gradient);
      EXPECT_NEAR(gradientError, 0, Eps<T>(1e-15f, 1e-15));
      EXPECT_LE(gradient.norm(), Eps<T>(1e-7f, 1e-7));

      for (size_t i = 0; i < 10; i++) {
        parameters = VectorX<T>::Random(transform.numAllModelParameters());
        state.set(transform.apply(parameters), skeleton);
        v = applySSD(bindpose, skin, mesh.vertices, state);

        cl.clear();
        for (size_t si = 0; si < kMaxSkinJoints; si++) {
          if (skin.weight(vi, si) == 0.0)
            continue;
          const auto parent = skin.index(vi, si);
          const Vector3<T> offset = state.jointState[parent].transformation.inverse() * v[vi];
          cl.push_back(PositionDataT<T>(offset, target, parent, skin.weight(vi, si)));
        }
        errorFunction.setConstraints(cl);

        gradient.setZero();
        gradientError = errorFunction.getGradient(parameters, state, gradient);
        auto numError = (v[vi] - target).squaredNorm();
        EXPECT_NEAR(gradientError, numError, Eps<T>(1e-5f, 1e-5));

        // calculate numerical gradient
        constexpr T kStepSize = 1e-5;
        VectorX<T> numGradient = VectorX<T>::Zero(transform.numAllModelParameters());
        for (auto p = 0; p < transform.numAllModelParameters(); p++) {
          // perform higher-order finite differences for accuracy
          VectorX<T> params = parameters;
          params(p) = parameters(p) - kStepSize;
          state.set(transform.apply(params), skeleton);
          v = applySSD(bindpose, skin, mesh.vertices, state);
          const T h_1 = (v[vi] - target).squaredNorm();

          params(p) = parameters(p) + kStepSize;
          state.set(transform.apply(params), skeleton);
          v = applySSD(bindpose, skin, mesh.vertices, state);
          const T h1 = (v[vi] - target).squaredNorm();

          numGradient(p) = (h1 - h_1) / (2.0 * kStepSize);
        }

        // check the gradients are similar
        {
          SCOPED_TRACE("Checking Numerical Gradient");
          if ((numGradient + gradient).norm() != 0.0) {
            EXPECT_LE(
                (numGradient - gradient).norm() / (numGradient + gradient).norm(),
                Eps<T>(1e-1f, 1e-1));
          }
        }
      }
    }
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, VertexErrorFunction) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values

  const size_t nConstraints = 10;

  // Test WITHOUT blend shapes:
  {
    const Character character_orig = createTestCharacter();
    const ModelParametersT<T> modelParams =
        0.25 * VectorX<T>::Random(character_orig.parameterTransform.numAllModelParameters());

    for (VertexConstraintType type :
         {VertexConstraintType::POSITION,
          VertexConstraintType::PLANE,
          VertexConstraintType::NORMAL,
          VertexConstraintType::SYMMETRIC_NORMAL}) {
      const T errorTol = [&]() {
        switch (type) {
          case VertexConstraintType::POSITION:
          case VertexConstraintType::PLANE:
            return Eps<T>(5e-2f, 1e-5);

          // TODO NORMAL constraints have a much higher epsilon than I'd prefer to see;
          // it would be good to dig into this.
          case VertexConstraintType::NORMAL:
          case VertexConstraintType::SYMMETRIC_NORMAL:
            return Eps<T>(5e-2f, 5e-2);

          default:
            // Shouldn't reach here
            return T(0);
        }
      }();

      VertexErrorFunctionT<T> errorFunction(character_orig, type);
      for (size_t iCons = 0; iCons < nConstraints; ++iCons) {
        errorFunction.addConstraint(
            uniform<int>(0, character_orig.mesh->vertices.size() - 1),
            uniform<float>(0, 1),
            uniform<Vector3<T>>(0, 1),
            uniform<Vector3<T>>(0, 1).normalized());
      }

      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          modelParams,
          character_orig.skeleton,
          character_orig.parameterTransform.cast<T>(),
          errorTol,
          Eps<T>(1e-6f, 1e-15),
          true,
          false);
    }
  }

  // Test WITH blend shapes:
  {
    const Character character_blend = withTestBlendShapes(createTestCharacter());
    const ModelParametersT<T> modelParams =
        0.25 * VectorX<T>::Random(character_blend.parameterTransform.numAllModelParameters());
    // It's trickier to test NORMAL and SYMMETRIC_NORMAL constraints in the blend shape case because
    // the mesh normals are recomputed after blend shapes are applied (this is the only sensible
    // thing to do since the blend shapes can drastically change the shape) and thus the normals
    // depend on the blend shapes in a very complicated way that we aren't currently trying to
    // model.
    for (VertexConstraintType type :
         {VertexConstraintType::POSITION, VertexConstraintType::PLANE}) {
      VertexErrorFunctionT<T> errorFunction(character_blend, type);
      for (size_t iCons = 0; iCons < nConstraints; ++iCons) {
        errorFunction.addConstraint(
            uniform<int>(0, character_blend.mesh->vertices.size() - 1),
            uniform<float>(0, 1),
            uniform<Vector3<T>>(0, 1),
            uniform<Vector3<T>>(0, 1).normalized());
      }

      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          modelParams,
          character_blend.skeleton,
          character_blend.parameterTransform.cast<T>(),
          Eps<T>(1e-2f, 1e-5),
          Eps<T>(1e-6f, 5e-16),
          true,
          false);
    }
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, VertexPositionErrorFunctionFaceParameters) {
  using T = typename TestFixture::Type;

  const size_t nConstraints = 10;

  // Face expression blend shapes only
  {
    const Character character_blend = withTestFaceExpressionBlendShapes(createTestCharacter());
    const ModelParametersT<T> modelParams =
        0.25 * VectorX<T>::Random(character_blend.parameterTransform.numAllModelParameters());

    // TODO: Add PLANE, NORMAL and SYMMETRIC_NORMAL?
    for (VertexConstraintType type : {VertexConstraintType::POSITION}) {
      VertexErrorFunctionT<T> errorFunction(character_blend, type);
      for (size_t iCons = 0; iCons < nConstraints; ++iCons) {
        errorFunction.addConstraint(
            uniform<int>(0, character_blend.mesh->vertices.size() - 1),
            uniform<float>(0, 1),
            uniform<Vector3<T>>(0, 1),
            uniform<Vector3<T>>(0, 1).normalized());
      }

      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          modelParams,
          character_blend.skeleton,
          character_blend.parameterTransform.cast<T>(),
          Eps<T>(1e-2f, 1e-5),
          Eps<T>(1e-6f, 5e-16),
          true,
          false);
    }
  }

  // Face expression blend shapes plus shape blend shapes
  {
    const Character character_blend =
        withTestBlendShapes(withTestFaceExpressionBlendShapes(createTestCharacter()));
    const ModelParametersT<T> modelParams =
        0.25 * VectorX<T>::Random(character_blend.parameterTransform.numAllModelParameters());

    // TODO: Add PLANE, NORMAL and SYMMETRIC_NORMAL?
    for (VertexConstraintType type : {VertexConstraintType::POSITION}) {
      VertexErrorFunctionT<T> errorFunction(character_blend, type);
      for (size_t iCons = 0; iCons < nConstraints; ++iCons) {
        errorFunction.addConstraint(
            uniform<int>(0, character_blend.mesh->vertices.size() - 1),
            uniform<float>(0, 1),
            uniform<Vector3<T>>(0, 1),
            uniform<Vector3<T>>(0, 1).normalized());
      }

      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          modelParams,
          character_blend.skeleton,
          character_blend.parameterTransform.cast<T>(),
          Eps<T>(1e-2f, 1e-5),
          Eps<T>(1e-6f, 5e-16),
          true,
          false);
    }
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, PositionErrorL2_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  Random rand;
  PositionErrorFunctionT<T> errorFunction(
      skeleton, character.parameterTransform, GeneralizedLossT<T>::kL2, rand.uniform<T>(0.2, 10));
  const T TEST_WEIGHT_VALUE = 4.5;
  {
    SCOPED_TRACE("PositionL2 Constraint Test");
    std::vector<PositionDataT<T>> cl{
        PositionDataT<T>(Vector3<T>::Random(), Vector3<T>::Random(), 2, TEST_WEIGHT_VALUE),
        PositionDataT<T>(Vector3<T>::Random(), Vector3<T>::Random(), 1, TEST_WEIGHT_VALUE)};
    errorFunction.setConstraints(cl);

    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        skeleton,
        transform,
        Eps<T>(5e-2f, 5e-6));
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters = VectorX<T>::Random(transform.numAllModelParameters());
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          parameters,
          skeleton,
          transform,
          Eps<T>(5e-2f, 5e-6),
          Eps<T>(1e-6f, 1e-7));
    }
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, PositionErrorL1_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  Random rand;
  PositionErrorFunctionT<T> errorFunction(
      skeleton, character.parameterTransform, GeneralizedLossT<T>::kL1, rand.uniform<T>(0.5, 2));
  const T TEST_WEIGHT_VALUE = 4.5;
  {
    SCOPED_TRACE("PositionL1 Constraint Test");
    std::vector<PositionDataT<T>> cl{
        PositionDataT<T>(Vector3<T>::Random(), Vector3<T>::Random(), 2, TEST_WEIGHT_VALUE),
        PositionDataT<T>(Vector3<T>::Random(), Vector3<T>::Random(), 1, TEST_WEIGHT_VALUE)};
    errorFunction.setConstraints(cl);

    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        skeleton,
        transform,
        Eps<T>(5e-2f, 5e-9),
        Eps<T>(1e-6f, 1e-7),
        false,
        false);
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters = VectorX<T>::Random(transform.numAllModelParameters());
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          parameters,
          skeleton,
          transform,
          Eps<T>(5e-2f, 5e-9),
          Eps<T>(1e-6f, 1e-7),
          false,
          false);
    }
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, PositionErrorCauchy_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  Random rand;
  PositionErrorFunctionT<T> errorFunction(
      skeleton,
      character.parameterTransform,
      GeneralizedLossT<T>::kCauchy,
      rand.uniform<T>(0.5, 2));
  const T TEST_WEIGHT_VALUE = 4.5;
  {
    SCOPED_TRACE("PositionCauchy Constraint Test");
    std::vector<PositionDataT<T>> cl{
        PositionDataT<T>(Vector3<T>::Random(), Vector3<T>::Random(), 2, TEST_WEIGHT_VALUE),
        PositionDataT<T>(Vector3<T>::Random(), Vector3<T>::Random(), 1, TEST_WEIGHT_VALUE)};
    errorFunction.setConstraints(cl);

    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        skeleton,
        transform,
        Eps<T>(5e-2f, 5e-9),
        Eps<T>(1e-6f, 1e-7),
        false,
        false);
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters = VectorX<T>::Random(transform.numAllModelParameters());
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          parameters,
          skeleton,
          transform,
          Eps<T>(5e-2f, 5e-9),
          Eps<T>(1e-6f, 1e-7),
          false,
          false);
    }
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, PositionErrorWelsch_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  PositionErrorFunctionT<T> errorFunction(
      skeleton, character.parameterTransform, GeneralizedLossT<T>::kWelsch);
  const T TEST_WEIGHT_VALUE = 4.5;
  {
    SCOPED_TRACE("PositionWelsch Constraint Test");
    std::vector<PositionDataT<T>> cl{
        PositionDataT<T>(Vector3<T>::Random(), Vector3<T>::Random(), 2, TEST_WEIGHT_VALUE),
        PositionDataT<T>(Vector3<T>::Random(), Vector3<T>::Random(), 1, TEST_WEIGHT_VALUE)};
    errorFunction.setConstraints(cl);

    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        skeleton,
        transform,
        Eps<T>(5e-2f, 5e-9),
        Eps<T>(1e-6f, 1e-7),
        false,
        false);
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters = VectorX<T>::Random(transform.numAllModelParameters());
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          parameters,
          skeleton,
          transform,
          Eps<T>(1e-1f, 5e-9),
          Eps<T>(1e-6f, 1e-7),
          false,
          false);
    }
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, PositionErrorGeneral_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  Random rand;
  PositionErrorFunctionT<T> errorFunction(
      skeleton, character.parameterTransform, rand.uniform<T>(0.1, 10));
  const T TEST_WEIGHT_VALUE = 4.5;
  {
    SCOPED_TRACE("PositionGeneral Constraint Test");
    std::vector<PositionDataT<T>> cl{
        PositionDataT<T>(Vector3<T>::Random(), Vector3<T>::Random(), 2, TEST_WEIGHT_VALUE),
        PositionDataT<T>(Vector3<T>::Random(), Vector3<T>::Random(), 1, TEST_WEIGHT_VALUE)};
    errorFunction.setConstraints(cl);

    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        skeleton,
        transform,
        Eps<T>(1e-1f, 5e-9),
        Eps<T>(1e-6f, 1e-7),
        false,
        false);
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters = VectorX<T>::Random(transform.numAllModelParameters());
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          parameters,
          skeleton,
          transform,
          Eps<T>(5e-1f, 5e-9),
          Eps<T>(1e-6f, 1e-7),
          false,
          false);
    }
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, PlaneErrorL2_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  PlaneErrorFunctionT<T> errorFunction(skeleton, character.parameterTransform);
  const T TEST_WEIGHT_VALUE = 4.5;
  {
    SCOPED_TRACE("PlaneL2 Constraint Test");
    std::vector<PlaneDataT<T>> cl{
        PlaneDataT<T>(
            uniform<Vector3<T>>(0, 1),
            uniform<Vector3<T>>(0, 1).normalized(),
            uniform<float>(0, 1),
            2,
            TEST_WEIGHT_VALUE),
        PlaneDataT<T>(
            uniform<Vector3<T>>(0, 1),
            uniform<Vector3<T>>(0, 1).normalized(),
            uniform<float>(0, 1),
            1,
            TEST_WEIGHT_VALUE)};
    errorFunction.setConstraints(cl);

    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        skeleton,
        transform,
        Eps<T>(5e-2f, 5e-6));

    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters = VectorX<T>::Random(transform.numAllModelParameters());
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          parameters,
          skeleton,
          transform,
          Eps<T>(5e-2f, 1e-5),
          Eps<T>(1e-6f, 1e-7));
    }
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, HalfPlaneErrorL2_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  PlaneErrorFunctionT<T> errorFunction(skeleton, character.parameterTransform, true);
  const T TEST_WEIGHT_VALUE = 4.5;
  {
    SCOPED_TRACE("PlaneL2 Constraint Test");
    std::vector<PlaneDataT<T>> cl{
        PlaneDataT<T>(
            uniform<Vector3<T>>(0, 1),
            uniform<Vector3<T>>(0, 1).normalized(),
            uniform<float>(0, 1),
            2,
            TEST_WEIGHT_VALUE),
        PlaneDataT<T>(
            uniform<Vector3<T>>(0, 1),
            uniform<Vector3<T>>(0, 1).normalized(),
            uniform<float>(0, 1),
            1,
            TEST_WEIGHT_VALUE)};
    errorFunction.setConstraints(cl);

    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        skeleton,
        transform,
        Eps<T>(1e-2f, 5e-6));

    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters = VectorX<T>::Random(transform.numAllModelParameters());
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          parameters,
          skeleton,
          transform,
          Eps<T>(5e-2f, 5e-6),
          Eps<T>(1e-6f, 1e-7));
    }
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, OrientationErrorL2_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  OrientationErrorFunctionT<T> errorFunction(skeleton, character.parameterTransform);
  const T TEST_WEIGHT_VALUE = 4.5;

  {
    SCOPED_TRACE("Orientation Constraint Test");
    std::vector<OrientationDataT<T>> cl{
        OrientationDataT<T>(
            rotVecToQuaternion<T>(Vector3<T>::Random()),
            rotVecToQuaternion<T>(Vector3<T>::Random()),
            2,
            TEST_WEIGHT_VALUE),
        OrientationDataT<T>(
            rotVecToQuaternion<T>(Vector3<T>::Random()),
            rotVecToQuaternion<T>(Vector3<T>::Random()),
            1,
            TEST_WEIGHT_VALUE)};
    errorFunction.setConstraints(std::move(cl));

    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        skeleton,
        transform,
        Eps<T>(0.03f, 5e-6));
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters = VectorX<T>::Random(transform.numAllModelParameters()) * 0.25;
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          parameters,
          skeleton,
          transform,
          Eps<T>(0.03f, 5e-6),
          Eps<T>(1e-6f, 1e-7));
    }
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, FixedAxisDiffErrorL2_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  FixedAxisDiffErrorFunctionT<T> errorFunction(skeleton, character.parameterTransform);
  const T TEST_WEIGHT_VALUE = 4.5;
  {
    SCOPED_TRACE("FixedAxisDiffL2 Constraint Test");
    std::vector<FixedAxisDataT<T>> cl{
        FixedAxisDataT<T>(Vector3<T>::Random(), Vector3<T>::Random(), 2, TEST_WEIGHT_VALUE),
        FixedAxisDataT<T>(Vector3<T>::Random(), Vector3<T>::Random(), 1, TEST_WEIGHT_VALUE)};
    errorFunction.setConstraints(cl);

    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        skeleton,
        transform,
        Eps<T>(5e-2f, 5e-6));
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters = VectorX<T>::Random(transform.numAllModelParameters());
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          parameters,
          skeleton,
          transform,
          Eps<T>(8e-2f, 5e-6),
          Eps<T>(1e-6f, 1e-7));
    }
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, FixedAxisCosErrorL2_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  FixedAxisCosErrorFunctionT<T> errorFunction(skeleton, character.parameterTransform);
  const T TEST_WEIGHT_VALUE = 4.5;
  {
    SCOPED_TRACE("FixedAxisCosL2 Constraint Test");
    std::vector<FixedAxisDataT<T>> cl{
        FixedAxisDataT<T>(Vector3<T>::Random(), Vector3<T>::Random(), 2, TEST_WEIGHT_VALUE),
        FixedAxisDataT<T>(Vector3<T>::Random(), Vector3<T>::Random(), 1, TEST_WEIGHT_VALUE)};
    errorFunction.setConstraints(cl);

    TEST_GRADIENT_AND_JACOBIAN(
        T,
        &errorFunction,
        ModelParametersT<T>::Zero(transform.numAllModelParameters()),
        skeleton,
        transform,
        Eps<T>(2e-2f, 1e-5));
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters = VectorX<T>::Random(transform.numAllModelParameters());
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          parameters,
          skeleton,
          transform,
          Eps<T>(5e-2f, 1e-5),
          Eps<T>(1e-6f, 1e-7));
    }
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, FixedAxisAngleErrorL2_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  FixedAxisAngleErrorFunctionT<T> errorFunction(skeleton, character.parameterTransform);
  const T TEST_WEIGHT_VALUE = 4.5;
  {
    SCOPED_TRACE("FixedAxisAngleL2 Constraint Test");
    std::vector<FixedAxisDataT<T>> cl{
        FixedAxisDataT<T>(Vector3<T>::Random(), Vector3<T>::Random(), 2, TEST_WEIGHT_VALUE),
        // corner case when the angle is close to zero
        FixedAxisDataT<T>(
            Vector3<T>::UnitY(), Vector3<T>(1e-16, 1 + 1e-16, 1e-16), 1, TEST_WEIGHT_VALUE),
        FixedAxisDataT<T>(Vector3<T>::UnitX(), Vector3<T>::UnitX(), 2, TEST_WEIGHT_VALUE)};
    errorFunction.setConstraints(cl);

    if constexpr (std::is_same_v<T, float>)
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          ModelParametersT<T>::Zero(transform.numAllModelParameters()),
          skeleton,
          transform,
          5e-2f);
    else if constexpr (std::is_same_v<T, double>)
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          ModelParametersT<T>::Zero(transform.numAllModelParameters()),
          skeleton,
          transform,
          1e-8,
          1e-6,
          true,
          false); // jacobian test is inaccurate around the corner case
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters = VectorX<T>::Random(transform.numAllModelParameters());
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          parameters,
          skeleton,
          transform,
          Eps<T>(1e-1f, 5e-5),
          Eps<T>(1e-6f, 1e-7));
    }
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, NormalError_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  NormalErrorFunctionT<T> errorFunction(skeleton, character.parameterTransform);
  const T TEST_WEIGHT_VALUE = 4.5;
  {
    SCOPED_TRACE("Normal Constraint Test");
    std::vector<NormalDataT<T>> cl{
        NormalDataT<T>(
            Vector3<T>::Random(), Vector3<T>::Random(), Vector3<T>::Random(), 1, TEST_WEIGHT_VALUE),
        NormalDataT<T>(
            Vector3<T>::Random(),
            Vector3<T>::Random(),
            Vector3<T>::Random(),
            2,
            TEST_WEIGHT_VALUE)};
    errorFunction.setConstraints(cl);

    if constexpr (std::is_same_v<T, float>)
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          ModelParametersT<T>::Zero(transform.numAllModelParameters()),
          skeleton,
          transform,
          1e-2f);
    else if constexpr (std::is_same_v<T, double>)
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          ModelParametersT<T>::Zero(transform.numAllModelParameters()),
          skeleton,
          transform,
          1e-8,
          1e-6,
          true,
          false); // jacobian test is inaccurate around the corner case
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters = VectorX<T>::Random(transform.numAllModelParameters());
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          parameters,
          skeleton,
          transform,
          Eps<T>(1e-1f, 2e-5),
          Eps<T>(1e-6f, 1e-7));
    }
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, AimDistError_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  AimDistErrorFunctionT<T> errorFunction(skeleton, character.parameterTransform);
  const T TEST_WEIGHT_VALUE = 4.5;
  {
    SCOPED_TRACE("AimDist Constraint Test");
    std::vector<AimDataT<T>> cl{
        AimDataT<T>(
            Vector3<T>::Random(), Vector3<T>::Random(), Vector3<T>::Random(), 1, TEST_WEIGHT_VALUE),
        AimDataT<T>(
            Vector3<T>::Random(),
            Vector3<T>::Random(),
            Vector3<T>::Random(),
            2,
            TEST_WEIGHT_VALUE)};
    errorFunction.setConstraints(cl);

    if constexpr (std::is_same_v<T, float>)
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          ModelParametersT<T>::Zero(transform.numAllModelParameters()),
          skeleton,
          transform,
          1e-2f);
    else if constexpr (std::is_same_v<T, double>)
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          ModelParametersT<T>::Zero(transform.numAllModelParameters()),
          skeleton,
          transform,
          1e-8,
          1e-6,
          true,
          false); // jacobian test is inaccurate around the corner case
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters = VectorX<T>::Random(transform.numAllModelParameters());
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          parameters,
          skeleton,
          transform,
          Eps<T>(1e-1f, 2e-5),
          Eps<T>(1e-6f, 1e-7));
    }
  }
}

TYPED_TEST(Momentum_ErrorFunctionsTest, AimDirError_GradientsAndJacobians) {
  using T = typename TestFixture::Type;

  // create skeleton and reference values
  const Character character = createTestCharacter();
  const Skeleton& skeleton = character.skeleton;
  const ParameterTransformT<T> transform = character.parameterTransform.cast<T>();

  // create constraints
  AimDirErrorFunctionT<T> errorFunction(skeleton, character.parameterTransform);
  const T TEST_WEIGHT_VALUE = 4.5;
  {
    SCOPED_TRACE("AimDir Constraint Test");
    std::vector<AimDataT<T>> cl{
        AimDataT<T>(
            Vector3<T>::Random(), Vector3<T>::Random(), Vector3<T>::Random(), 1, TEST_WEIGHT_VALUE),
        AimDataT<T>(
            Vector3<T>::Random(),
            Vector3<T>::Random(),
            Vector3<T>::Random(),
            2,
            TEST_WEIGHT_VALUE)};
    errorFunction.setConstraints(cl);

    if constexpr (std::is_same_v<T, float>)
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          ModelParametersT<T>::Zero(transform.numAllModelParameters()),
          skeleton,
          transform,
          5e-2f);
    else if constexpr (std::is_same_v<T, double>)
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          ModelParametersT<T>::Zero(transform.numAllModelParameters()),
          skeleton,
          transform,
          1e-8,
          1e-6,
          true,
          false); // jacobian test is inaccurate around the corner case
    for (size_t i = 0; i < 10; i++) {
      ModelParametersT<T> parameters = VectorX<T>::Random(transform.numAllModelParameters());
      TEST_GRADIENT_AND_JACOBIAN(
          T,
          &errorFunction,
          parameters,
          skeleton,
          transform,
          Eps<T>(1e-1f, 2e-5),
          Eps<T>(1e-6f, 1e-7));
    }
  }
}
