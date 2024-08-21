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
struct TransformTest : testing::Test {
  using Type = T;
};

TYPED_TEST_SUITE(TransformTest, Types);

TYPED_TEST(TransformTest, Multiplication) {
  using T = typename TestFixture::Type;

  for (size_t iTest = 0; iTest < 100; ++iTest) {
    const TransformT<T> trans1 = TransformT<T>::makeRandom(iTest > 1, iTest > 10, iTest > 50);
    const TransformT<T> trans2 = TransformT<T>::makeRandom(iTest > 1, iTest > 10, iTest > 50);

    const TransformT<T> tmp = trans1 * trans2;
    const Affine3<T> res1 = tmp.toAffine3();
    const Affine3<T> res2 = trans1.toAffine3() * trans2.toAffine3();

    EXPECT_LE(
        (res1.matrix() - res2.matrix()).template lpNorm<Eigen::Infinity>(), Eps<T>(1e-5f, 1e-13));
  }
}

TYPED_TEST(TransformTest, Inverse) {
  using T = typename TestFixture::Type;

  for (size_t iTest = 0; iTest < 100; ++iTest) {
    const TransformT<T> trans1 = TransformT<T>::makeRandom(iTest > 1, iTest > 10, iTest > 50);

    const Affine3<T> res1 = trans1.inverse().toAffine3();
    const Affine3<T> res2 = trans1.toAffine3().inverse();

    EXPECT_LE(
        (res1.matrix() - res2.matrix()).template lpNorm<Eigen::Infinity>(), Eps<T>(1e-4f, 5e-14));
  }
}

TYPED_TEST(TransformTest, TransformPoint) {
  using T = typename TestFixture::Type;

  for (size_t iTest = 0; iTest < 100; ++iTest) {
    const TransformT<T> trans1 = TransformT<T>::makeRandom(iTest > 1, iTest > 10, iTest > 50);
    const auto randomPoint = uniform<Vector3<T>>(-10, 10);

    const Eigen::Vector3<T> res1 = trans1.transformPoint(randomPoint);
    const Eigen::Vector3<T> res2 = trans1.toAffine3() * randomPoint;

    EXPECT_LE((res1 - res2).template lpNorm<Eigen::Infinity>(), Eps<T>(1e-5f, 5e-14));
  }
}

TYPED_TEST(TransformTest, TransformVec) {
  using T = typename TestFixture::Type;

  for (size_t iTest = 0; iTest < 100; ++iTest) {
    const TransformT<T> trans1 = TransformT<T>::makeRandom(iTest > 1, iTest > 10, iTest > 50);
    const Eigen::Vector3<T> randomVec = uniform<Vector3<T>>(-1, 1).normalized();

    const Eigen::Vector3<T> res1 = trans1.rotate(randomVec);
    const Eigen::Vector3<T> res2 = trans1.toAffine3().rotation() * randomVec;

    EXPECT_LE((res1 - res2).template lpNorm<Eigen::Infinity>(), Eps<T>(1e-6f, 5e-15));
  }
}

// This test is to make sure that the Eigen::Affine3f and momentum::Transformf are
// intercompatible.
TYPED_TEST(TransformTest, CompatibleWithEigenAffine) {
  using T = typename TestFixture::Type;

  for (size_t iTest = 0; iTest < 100; ++iTest) {
    // const TransformT<T> tf1 = TransformT<T>::makeRandom(iTest > 1, iTest > 10, iTest > 50);
    const TransformT<T> tf1 = TransformT<T>::makeRandom();
    const Affine3<T> tf2 = tf1.toAffine3();
    TransformT<T> tf3;
    tf3 = tf2;
    const Affine3<T> tf4 = tf3.toAffine3();

    EXPECT_LE(
        (tf1.toMatrix() - tf2.matrix()).template lpNorm<Eigen::Infinity>(), Eps<T>(1e-4f, 5e-14));
    EXPECT_LE(
        (tf2.matrix() - tf3.toMatrix()).template lpNorm<Eigen::Infinity>(), Eps<T>(1e-4f, 5e-14));
    EXPECT_LE(
        (tf3.toMatrix() - tf4.matrix()).template lpNorm<Eigen::Infinity>(), Eps<T>(1e-4f, 5e-14));
    EXPECT_LE(
        (tf4.matrix() - tf1.toMatrix()).template lpNorm<Eigen::Infinity>(), Eps<T>(1e-4f, 5e-14));
  }
}
