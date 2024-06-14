/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "axel/BoundingBox.h"
#include "axel/common/Constants.h"

#include <perception/test_helpers/EigenChecks.h>

namespace axel::test {
namespace {

using ScalarTypes = testing::Types<float, double>;

template <typename T>
struct BoundingBoxTest : testing::Test {
  using Type = T;
};

TYPED_TEST_SUITE(BoundingBoxTest, ScalarTypes);

TYPED_TEST(BoundingBoxTest, Basic) {
  using T = typename TestFixture::Type;

  const Eigen::Vector3<T> min{0., 0., 0.};
  const Eigen::Vector3<T> max{1., 2., 3.};
  const Eigen::Vector3<T> center{(min + max) / 2.0};

  BoundingBox<T> bbox{min, max};

  EXPECT_EIGEN_MATRIX_NEAR(min, bbox.min(), detail::eps<T>());
  EXPECT_EIGEN_MATRIX_NEAR(max, bbox.max(), detail::eps<T>());
  EXPECT_EIGEN_MATRIX_NEAR(center, bbox.center(), detail::eps<T>());
  EXPECT_EQ(2, bbox.maxDimension());

  EXPECT_TRUE(bbox.intersects(Eigen::Vector3<T>{0., 0., 0.}, Eigen::Vector3<T>{1., 1., 1.}));
  EXPECT_FALSE(bbox.intersects(Eigen::Vector3<T>{-1., 0., 0.}, Eigen::Vector3<T>{1., 1., -1.}));
  EXPECT_TRUE(bbox.intersects(
      BoundingBox<T>{Eigen::Vector3<T>{1., 1., 1.}, Eigen::Vector3<T>{2., 2., 2.}}));
  EXPECT_FALSE(bbox.intersects(
      BoundingBox<T>{Eigen::Vector3<T>{-1., -1., -1.}, Eigen::Vector3<T>{-2., -2., -2.}}));

  const Eigen::Vector3<T> p{2., 2., 2.};
  bbox.extend(p);

  const Eigen::Vector3<T> newMax{max.array().max(p.array())};
  const Eigen::Vector3<T> newCenter{(min + newMax) / 2.0};

  EXPECT_EIGEN_MATRIX_NEAR(min, bbox.min(), detail::eps<T>());
  EXPECT_EIGEN_MATRIX_NEAR(newMax, bbox.max(), detail::eps<T>());
  EXPECT_EIGEN_MATRIX_NEAR(newCenter, bbox.center(), detail::eps<T>());
}

TYPED_TEST(BoundingBoxTest, MaxDim) {
  using T = typename TestFixture::Type;
  // Regular cases.
  EXPECT_EQ(
      BoundingBox<T>(Eigen::Vector3<T>::Zero(), Eigen::Vector3<T>{1, 2, 3}).maxDimension(), 2);
  EXPECT_EQ(
      BoundingBox<T>(Eigen::Vector3<T>::Zero(), Eigen::Vector3<T>{2, 1, 3}).maxDimension(), 2);
  EXPECT_EQ(
      BoundingBox<T>(Eigen::Vector3<T>::Zero(), Eigen::Vector3<T>{1, 3, 2}).maxDimension(), 1);
  EXPECT_EQ(
      BoundingBox<T>(Eigen::Vector3<T>::Zero(), Eigen::Vector3<T>{2, 3, 1}).maxDimension(), 1);
  EXPECT_EQ(
      BoundingBox<T>(Eigen::Vector3<T>::Zero(), Eigen::Vector3<T>{3, 2, 1}).maxDimension(), 0);
  EXPECT_EQ(
      BoundingBox<T>(Eigen::Vector3<T>::Zero(), Eigen::Vector3<T>{3, 1, 2}).maxDimension(), 0);
}

TYPED_TEST(BoundingBoxTest, MaxDimEqualityOrdering) {
  using T = typename TestFixture::Type;
  // When equal, we should get the highest index axis due to implementation details.
  EXPECT_EQ(
      BoundingBox<T>(Eigen::Vector3<T>::Zero(), Eigen::Vector3<T>{1, 1, 1}).maxDimension(), 2);
  EXPECT_EQ(
      BoundingBox<T>(Eigen::Vector3<T>::Zero(), Eigen::Vector3<T>{0, 1, 1}).maxDimension(), 2);
  EXPECT_EQ(
      BoundingBox<T>(Eigen::Vector3<T>::Zero(), Eigen::Vector3<T>{1, 0, 1}).maxDimension(), 2);
  EXPECT_EQ(
      BoundingBox<T>(Eigen::Vector3<T>::Zero(), Eigen::Vector3<T>{1, 1, 0}).maxDimension(), 1);
}

TYPED_TEST(BoundingBoxTest, RayIntersection_Basic) {
  using T = typename TestFixture::Type;

  const Eigen::Vector3<T> min{0., 0., 0.};
  const Eigen::Vector3<T> max{1., 1., 1.};
  const BoundingBox<T> bbox{min, max};

  EXPECT_TRUE(bbox.intersects(Eigen::Vector3<T>{0.5, 0.5, 1.0}, Eigen::Vector3<T>{0.0, 0.0, -1.0}));
  EXPECT_FALSE(
      bbox.intersects(Eigen::Vector3<T>{-0.5, 0.5, 1.0}, Eigen::Vector3<T>{0.0, 0.0, -1.0}));
  EXPECT_TRUE(bbox.intersectsBranchless(
      Eigen::Vector3<T>{0.5, 0.5, 1.0}, Eigen::Vector3<T>{0.0, 0.0, -1.0}.cwiseInverse()));
  EXPECT_FALSE(bbox.intersectsBranchless(
      Eigen::Vector3<T>{-0.5, 0.5, 1.0}, Eigen::Vector3<T>{0.0, 0.0, -1.0}.cwiseInverse()));
}

// We add this test to ensure that min and max behave as expected with NaN.
// The basic property of NaN is that it returns false in all boolean expressions.
// By using a valid value for y in the expression below, we avoid introducing NaNs in the
// calculations. e.g. min(x, y) = x < y ? x : y, we can get y, even if X is NaN.
TEST(BoundingBoxTest, NansInStdMinMax) {
  EXPECT_EQ(std::min(1.0f, NAN), 1.0f);
  EXPECT_EQ(std::max(1.0f, NAN), 1.0f);
}

TYPED_TEST(BoundingBoxTest, RayIntersection_Edges) {
  using T = typename TestFixture::Type;

  const Eigen::Vector3<T> min{0., 0., 0.};
  const Eigen::Vector3<T> max{1., 1., 1.};
  const BoundingBox<T> bbox{min, max};

  // Check intersection against sides, where the ray is in the plane of the cube's side.
  EXPECT_TRUE(bbox.intersects(Eigen::Vector3<T>{0.0, 0.5, 1.0}, Eigen::Vector3<T>{0.0, 0.0, -1.0}));
  EXPECT_TRUE(bbox.intersects(Eigen::Vector3<T>{1.0, 0.5, 1.0}, Eigen::Vector3<T>{0.0, 0.0, -1.0}));
  EXPECT_TRUE(bbox.intersects(Eigen::Vector3<T>{0.5, 0.0, 1.0}, Eigen::Vector3<T>{0.0, 0.0, -1.0}));
  EXPECT_TRUE(bbox.intersects(Eigen::Vector3<T>{0.5, 1.0, 1.0}, Eigen::Vector3<T>{0.0, 0.0, -1.0}));

  EXPECT_TRUE(bbox.intersectsBranchless(
      Eigen::Vector3<T>{0.0, 0.5, 1.0}, Eigen::Vector3<T>{0.0, 0.0, -1.0}.cwiseInverse()));
  EXPECT_TRUE(bbox.intersectsBranchless(
      Eigen::Vector3<T>{1.0, 0.5, 1.0}, Eigen::Vector3<T>{0.0, 0.0, -1.0}.cwiseInverse()));
  EXPECT_TRUE(bbox.intersectsBranchless(
      Eigen::Vector3<T>{0.5, 0.0, 1.0}, Eigen::Vector3<T>{0.0, 0.0, -1.0}.cwiseInverse()));
  EXPECT_TRUE(bbox.intersectsBranchless(
      Eigen::Vector3<T>{0.5, 1.0, 1.0}, Eigen::Vector3<T>{0.0, 0.0, -1.0}.cwiseInverse()));
}

} // namespace
} // namespace axel::test
