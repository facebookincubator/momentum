/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "momentum/math/constants.h"
#include "momentum/math/covariance_matrix.h"

using namespace momentum;

using Types = testing::Types<float, double>;

template <typename T>
struct CovarianceMatrixTest : testing::Test {
  using Type = T;
};

TYPED_TEST_SUITE(CovarianceMatrixTest, Types);

TYPED_TEST(CovarianceMatrixTest, Inverse) {
  using T = typename TestFixture::Type;

  const int d = 10;
  const int q = 4;

  const T sigma = 0.1;
  const Eigen::MatrixX<T> A = Eigen::MatrixX<T>::Random(q, d);

  LowRankCovarianceMatrixT<T> cov;
  cov.reset(sigma, A);

  const Eigen::MatrixX<T> cov2 =
      sigma * sigma * Eigen::MatrixX<T>::Identity(d, d) + A.transpose() * A;
  const Eigen::MatrixX<T> cov2_inverse = cov2.inverse();

  const Eigen::VectorX<T> testVec = Eigen::VectorX<T>::Random(d);
  const Eigen::MatrixX<T> testMat = Eigen::MatrixX<T>::Random(d, 5);

  EXPECT_LT((cov2 * testVec - cov.times_vec(testVec)).norm(), Eps<T>(1e-6f, 5e-15));
  EXPECT_LT((cov2 * testMat - cov.times_mat(testMat)).norm(), Eps<T>(5e-6f, 5e-15));
  EXPECT_LT((cov2_inverse * testVec - cov.inverse_times_vec(testVec)).norm(), Eps<T>(5e-3f, 1e-11));
  EXPECT_LT((cov2_inverse * testMat - cov.inverse_times_mat(testMat)).norm(), Eps<T>(5e-3f, 1e-11));
  EXPECT_NEAR(std::log(cov2.determinant()), cov.logDeterminant(), Eps<T>(5e-5f, 5e-13));
  EXPECT_NEAR(
      std::log(cov2_inverse.determinant()), cov.inverse_logDeterminant(), Eps<T>(5e-5f, 5e-13));
}
