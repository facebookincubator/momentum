/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/math/random.h"

using namespace momentum;

using Types = testing::Types<float, double>;

template <typename T>
struct Momentum_JointTest : testing::Test {
  using Type = T;
};

TYPED_TEST_SUITE(Momentum_JointTest, Types);

TYPED_TEST(Momentum_JointTest, Definitions) {
  EXPECT_TRUE(kParametersPerJoint == 7);
}

TYPED_TEST(Momentum_JointTest, Construction) {
  using T = typename TestFixture::Type;

  JointT<T> jt;

  // create single joint state
  EXPECT_TRUE(jt.name == "uninitialized");
  EXPECT_TRUE(jt.parent == kInvalidIndex);
  EXPECT_TRUE(jt.preRotation.coeffs() == Quaternion<T>::Identity().coeffs());
  EXPECT_TRUE(jt.translationOffset == Vector3<T>::Zero());
}

template <typename T>
struct Momentum_JointStateTest : testing::Test {
  using Type = T;
};

TYPED_TEST_SUITE(Momentum_JointStateTest, Types);

TYPED_TEST(Momentum_JointStateTest, Identity) {
  using T = typename TestFixture::Type;

  JointT<T> jt;
  JointStateT<T> js;
  VectorX<T> params = VectorX<T>::Zero(kParametersPerJoint);

  // test zero parameters
  js.set(jt, params, nullptr);
  EXPECT_TRUE(js.localRotation().coeffs() == Quaternion<T>::Identity().coeffs());
  EXPECT_TRUE(js.localTranslation() == Vector3<T>::Zero());
  EXPECT_TRUE(js.localScale() == 1.0);

  EXPECT_TRUE(js.translationAxis == Matrix3<T>::Identity());
  EXPECT_TRUE(js.rotationAxis == Matrix3<T>::Identity());

  EXPECT_TRUE(js.rotation().coeffs() == Quaternion<T>::Identity().coeffs());
  EXPECT_TRUE(js.translation() == Vector3<T>::Zero());
  EXPECT_TRUE(js.scale() == 1.0);
}

TYPED_TEST(Momentum_JointStateTest, Translations) {
  using T = typename TestFixture::Type;

  JointT<T> jt;
  JointStateT<T> js;
  VectorX<T> params = VectorX<T>::Zero(kParametersPerJoint);

  Vector3<T> pos;
  Vector3<T> offset;

  // test random translation parameters
  for (size_t i = 0; i < 3; i++) {
    pos.setRandom();
    params.template topRows<3>() = pos;
    js.set(jt, params, nullptr);
    EXPECT_TRUE(js.localRotation().coeffs() == Quaternion<T>::Identity().coeffs());
    EXPECT_TRUE(js.localTranslation().isApprox(pos, Eps<T>(1e-7f, 1e-15)));
    EXPECT_TRUE(js.localScale() == 1.0);

    EXPECT_TRUE(js.translationAxis == Matrix3<T>::Identity());
    EXPECT_TRUE(js.rotationAxis == Matrix3<T>::Identity());

    EXPECT_TRUE(js.rotation().coeffs() == Quaternion<T>::Identity().coeffs());
    EXPECT_TRUE(js.translation().isApprox(pos));
    EXPECT_TRUE(js.scale() == 1.0);
  }

  // add offset
  for (size_t i = 0; i < 3; i++) {
    pos.setRandom();
    offset.setRandom();
    params.template topRows<3>() = pos;
    jt.translationOffset = offset;
    js.set(jt, params, nullptr);
    EXPECT_TRUE(js.localRotation().coeffs() == Quaternion<T>::Identity().coeffs());
    EXPECT_TRUE(js.localTranslation().isApprox(pos + offset, Eps<T>(1e-7f, 1e-7)));
    EXPECT_TRUE(js.localScale() == 1.0);

    EXPECT_TRUE(js.translationAxis == Matrix3<T>::Identity());
    EXPECT_TRUE(js.rotationAxis == Matrix3<T>::Identity());

    EXPECT_TRUE(js.rotation().coeffs() == Quaternion<T>::Identity().coeffs());
    EXPECT_TRUE(js.translation().isApprox(pos + offset, Eps<T>(1e-7f, 1e-7)));
    EXPECT_TRUE(js.scale() == 1.0);
  }
}

TYPED_TEST(Momentum_JointStateTest, Rotations) {
  using T = typename TestFixture::Type;

  JointT<T> jt;
  JointStateT<T> js;
  VectorX<T> params = VectorX<T>::Zero(kParametersPerJoint);

  jt.translationOffset.setZero();
  Vector3<T> rot;
  Quaternion<T> preRot;

  // test random translation parameters
  for (size_t i = 0; i < 3; i++) {
    rot.setRandom();
    params.template middleRows<3>(3) = rot;
    js.set(jt, params, nullptr);
    Quaternion<T> res = Quaternion<T>(Eigen::AngleAxis<T>(rot[2], Vector3<T>::UnitZ())) *
        Quaternion<T>(Eigen::AngleAxis<T>(rot[1], Vector3<T>::UnitY())) *
        Quaternion<T>(Eigen::AngleAxis<T>(rot[0], Vector3<T>::UnitX()));
    EXPECT_TRUE(js.localRotation().coeffs().isApprox(res.coeffs(), Eps<T>(1e-7f, 1e-15)));
    EXPECT_TRUE(js.localTranslation() == Vector3<T>::Zero());
    EXPECT_TRUE(js.localScale() == 1.0);

    EXPECT_TRUE(js.translationAxis == Matrix3<T>::Identity());
    EXPECT_TRUE(js.rotationAxis.col(2) == Vector3<T>::UnitZ());

    EXPECT_TRUE(js.rotation().coeffs().isApprox(res.coeffs()));
    EXPECT_TRUE(js.translation() == Vector3<T>::Zero());
    EXPECT_TRUE(js.scale() == 1.0);
  }

  // add pre-rotation
  for (size_t i = 0; i < 3; i++) {
    rot.setRandom();
    params.template middleRows<3>(3) = rot;
    const Quaternion<T> prot = uniformQuaternion<T>();
    jt.preRotation = prot;
    js.set(jt, params, nullptr);
    Quaternion<T> res = eulerToQuaternion<T>(rot, 0, 1, 2, EulerConvention::Extrinsic);
    EXPECT_TRUE(js.localRotation().coeffs().isApprox((prot * res).coeffs(), Eps<T>(1e-6f, 1e-7)));
    EXPECT_TRUE(js.localTranslation() == Vector3<T>::Zero());
    EXPECT_TRUE(js.localScale() == 1.0);

    EXPECT_TRUE(js.translationAxis == Matrix3<T>::Identity());
    EXPECT_TRUE(
        js.rotationAxis.col(2).isApprox(prot.toRotationMatrix().col(2), Eps<T>(1e-6f, 1e-6)));

    EXPECT_TRUE(js.rotation().coeffs().isApprox((prot * res).coeffs(), Eps<T>(1e-6f, 1e-7)));
    EXPECT_TRUE(js.translation() == Vector3<T>::Zero());
    EXPECT_TRUE(js.scale() == 1.0);
  }
}
