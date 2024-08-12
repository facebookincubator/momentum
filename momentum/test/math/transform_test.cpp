/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cmath>

#include "momentum/math/constants.h"
#include "momentum/math/random.h"
#include "momentum/math/transform.h"

using namespace momentum;

using Types = testing::Types<float, double>;

template <typename T>
TransformT<T> generateRandomTransform(bool useTranslation, bool useRotation, bool useScale) {
  Eigen::Vector3<T> trans = Eigen::Vector3<T>::Zero();
  if (useTranslation) {
    trans = uniform<Vector3<T>>(-10, 10);
  }

  Eigen::Quaternion<T> rot = Eigen::Quaternion<T>::Identity();
  if (useRotation) {
    const T u1 = uniform<T>(0, 1);
    const T u2 = uniform<T>(0, 1);
    const T u3 = uniform<T>(0, 1);
    Eigen::Vector4<T> coeffs(
        std::sqrt(T(1) - u1) * std::sin(twopi<T>() * u2),
        std::sqrt(T(1) - u1) * std::cos(twopi<T>() * u2),
        std::sqrt(u1) * std::sin(twopi<T>() * u3),
        std::sqrt(u1) * std::cos(twopi<T>() * u3));
    rot = Eigen::Quaternion<T>(coeffs).normalized();
  }

  T scale = T(1);
  if (useScale) {
    scale = uniform<T>(0.5, 2.5);
  }

  return TransformT<T>(trans, rot, scale);
}

template <typename T>
struct TransformTest : testing::Test {
  using Type = T;
};

TYPED_TEST_SUITE(TransformTest, Types);

TYPED_TEST(TransformTest, Multiplication) {
  using T = typename TestFixture::Type;

  for (size_t iTest = 0; iTest < 100; ++iTest) {
    const TransformT<T> trans1 = generateRandomTransform<T>(iTest > 1, iTest > 10, iTest > 50);
    const TransformT<T> trans2 = generateRandomTransform<T>(iTest > 1, iTest > 10, iTest > 50);

    const TransformT<T> tmp = trans1 * trans2;
    const Affine3<T> res1 = tmp.matrix();
    const Affine3<T> res2 = trans1.matrix() * trans2.matrix();

    EXPECT_LE(
        (res1.matrix() - res2.matrix()).template lpNorm<Eigen::Infinity>(), Eps<T>(1e-5f, 1e-13));
  }
}

TYPED_TEST(TransformTest, Inverse) {
  using T = typename TestFixture::Type;

  for (size_t iTest = 0; iTest < 100; ++iTest) {
    const TransformT<T> trans1 = generateRandomTransform<T>(iTest > 1, iTest > 10, iTest > 50);

    const Affine3<T> res1 = trans1.inverse().matrix();
    const Affine3<T> res2 = trans1.matrix().inverse();

    EXPECT_LE(
        (res1.matrix() - res2.matrix()).template lpNorm<Eigen::Infinity>(), Eps<T>(1e-4f, 5e-14));
  }
}

TYPED_TEST(TransformTest, TransformPoint) {
  using T = typename TestFixture::Type;

  for (size_t iTest = 0; iTest < 100; ++iTest) {
    const TransformT<T> trans1 = generateRandomTransform<T>(iTest > 1, iTest > 10, iTest > 50);
    const auto randomPoint = uniform<Vector3<T>>(-10, 10);

    const Eigen::Vector3<T> res1 = trans1.transformPoint(randomPoint);
    const Eigen::Vector3<T> res2 = trans1.matrix() * randomPoint;

    EXPECT_LE((res1 - res2).template lpNorm<Eigen::Infinity>(), Eps<T>(1e-5f, 5e-14));
  }
}

TYPED_TEST(TransformTest, TransformVec) {
  using T = typename TestFixture::Type;

  for (size_t iTest = 0; iTest < 100; ++iTest) {
    const TransformT<T> trans1 = generateRandomTransform<T>(iTest > 1, iTest > 10, iTest > 50);
    const Eigen::Vector3<T> randomVec = uniform<Vector3<T>>(-1, 1).normalized();

    const Eigen::Vector3<T> res1 = trans1.rotate(randomVec);
    const Eigen::Vector3<T> res2 = trans1.matrix().rotation() * randomVec;

    EXPECT_LE((res1 - res2).template lpNorm<Eigen::Infinity>(), Eps<T>(1e-6f, 5e-15));
  }
}
