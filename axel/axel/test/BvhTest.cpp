/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <unordered_set>

#include <gtest/gtest.h>
#include <Eigen/Core>

#include "axel/Bvh.h"
#include "axel/math/BoundingBoxUtils.h"
#include "axel/test/Helper.h"

namespace axel::test {
namespace {

using ScalarTypes = testing::Types<float, double>;

template <typename T>
struct BvhTest : testing::Test {
  using Type = T;
};

TYPED_TEST_SUITE(BvhTest, ScalarTypes);

TYPED_TEST(BvhTest, DefaultConstructor) {
  using S = typename TestFixture::Type;
  auto bvh = Bvh<S>();
  EXPECT_EQ(bvh.getNodeCount(), 0);
  EXPECT_EQ(bvh.getPrimitiveCount(), 0);
}

TYPED_TEST(BvhTest, BoxQuery_NoHits) {
  using S = typename TestFixture::Type;

  std::vector<BoundingBox<S>> boxes = {
      BoundingBox<S>(Eigen::Vector3<S>(0., 1., 2.), Eigen::Vector3<S>(1., 2., 3.)),
      BoundingBox<S>(Eigen::Vector3<S>(1., 1., 2.), Eigen::Vector3<S>(5., 6., 7.)),
      BoundingBox<S>(Eigen::Vector3<S>(2., 1., 2.), Eigen::Vector3<S>(3., 8., 9.)),
  };

  Bvh<S> bvh{};
  bvh.setBoundingBoxes(boxes);

  // No boxes will intersect with this one.
  const BoundingBox<S> queryBox{
      Eigen::Vector3<S>(-10., -10., -10.), Eigen::Vector3<S>(-9., -9., -9.)};
  typename Bvh<S>::QueryBuffer queryBuffer;
  const unsigned int hits{bvh.query(queryBox, queryBuffer)};
  ASSERT_EQ(hits, 0);

  const auto hitVec{bvh.query(queryBox)};
  ASSERT_EQ(hitVec.size(), hits);
  for (uint32_t i = 0; i < hits; ++i) {
    EXPECT_EQ(queryBuffer[i], hitVec[i]);
  }
}

TYPED_TEST(BvhTest, BoxQuery_AllHits) {
  using S = typename TestFixture::Type;

  std::vector<BoundingBox<S>> boxes = {
      BoundingBox<S>(Eigen::Vector3<S>(0., 1., 2.), Eigen::Vector3<S>(1., 2., 3.), 0),
      BoundingBox<S>(Eigen::Vector3<S>(1., 1., 2.), Eigen::Vector3<S>(5., 6., 7.), 1),
      BoundingBox<S>(Eigen::Vector3<S>(2., 1., 2.), Eigen::Vector3<S>(3., 8., 9.), 2),
  };

  Bvh<S> bvh{};
  bvh.setBoundingBoxes(boxes);

  // All of the boxes will intersect with this large one.
  const BoundingBox<S> queryBox{Eigen::Vector3<S>(-10., -10., -10.), Eigen::Vector3<S>(9., 9., 9.)};
  typename Bvh<S>::QueryBuffer queryBuffer;
  const unsigned int hits{bvh.query(queryBox, queryBuffer)};
  ASSERT_EQ(hits, 3);
  EXPECT_EQ(queryBuffer[0], 2);
  EXPECT_EQ(queryBuffer[1], 1);
  EXPECT_EQ(queryBuffer[2], 0);

  const auto hitVec{bvh.query(queryBox)};
  ASSERT_EQ(hitVec.size(), hits);
  for (uint32_t i = 0; i < hits; ++i) {
    EXPECT_EQ(queryBuffer[i], hitVec[i]);
  }
}

TYPED_TEST(BvhTest, RayQuery_NoHits) {
  using S = typename TestFixture::Type;

  const std::vector<BoundingBox<S>> boxes = {
      BoundingBox<S>(Eigen::Vector3<S>(0., 0., 0.), Eigen::Vector3<S>(1., 1., 1.), 0),
      BoundingBox<S>(Eigen::Vector3<S>(2., 0.25, 0.25), Eigen::Vector3<S>(3., 1.25, 1.25), 1),
      BoundingBox<S>(Eigen::Vector3<S>(4., -0.25, 0.), Eigen::Vector3<S>(5., 1.25, 1.), 2),
  };

  Bvh<S> bvh{};
  bvh.setBoundingBoxes(boxes);

  const Eigen::Vector3<S> rayOrigin{-1.0, 0.5, 0.5};
  const Eigen::Vector3<S> rayDir{-1.0, 0.0, 0.0};
  const auto hits{bvh.query(rayOrigin, rayDir)};
  EXPECT_EQ(hits.size(), 0);
}

template <typename S>
void testTraverseOverlappingPairsSingleBvh(size_t numBoxes, S extent = 10, S sparsityFactor = 1) {
  auto bvh = Bvh<S>();

  const auto [boxes, expectedOverlaps] =
      test::generateBoxesAndCollisions<S>(numBoxes, extent, sparsityFactor);

  bvh.setBoundingBoxes(boxes);

  test::IndexPairSet overlaps;
  auto callback = [&overlaps](Index indexA, Index indexB) {
    const Index min = std::min(indexA, indexB);
    const Index max = std::max(indexA, indexB);
    overlaps.emplace(min, max);
    return true; // continue traversal
  };

  bvh.traverseOverlappingPairs(std::move(callback));

  EXPECT_EQ(overlaps, expectedOverlaps);
}

TYPED_TEST(BvhTest, TraverseOverlappingPairsSingleBvhDense) {
  using S = typename TestFixture::Type;
  testTraverseOverlappingPairsSingleBvh<S>(1 << 4);
  testTraverseOverlappingPairsSingleBvh<S>(1 << 6);
  testTraverseOverlappingPairsSingleBvh<S>(1 << 8);
  testTraverseOverlappingPairsSingleBvh<S>(1 << 10);
}

TYPED_TEST(BvhTest, TraverseOverlappingPairsSingleBvhSparse) {
  using S = typename TestFixture::Type;
  testTraverseOverlappingPairsSingleBvh<S>(1 << 4, 10, 10);
  testTraverseOverlappingPairsSingleBvh<S>(1 << 6, 10, 10);
  testTraverseOverlappingPairsSingleBvh<S>(1 << 8, 10, 10);
  testTraverseOverlappingPairsSingleBvh<S>(1 << 10, 10, 10);
}

template <typename S>
void testTraverseOverlappingPairsDoubleBvh(
    size_t numBoxesA,
    size_t numBoxesB,
    S extent = 10,
    S sparsityFactor = 1) {
  auto bvhA = Bvh<S>();
  auto bvhB = Bvh<S>();

  const auto [boxesA, boxesB, expectedOverlaps] =
      test::generateBoxesAndInterTreeCollisions<S>(numBoxesA, numBoxesB, extent, sparsityFactor);

  bvhA.setBoundingBoxes(boxesA);
  bvhB.setBoundingBoxes(boxesB);

  test::IndexPairSet overlaps;
  auto callback = [&overlaps](Index indexA, Index indexB) {
    overlaps.emplace(indexA, indexB);
    return true; // continue traversal
  };

  bvhA.traverseOverlappingPairs(bvhB, std::move(callback));

  EXPECT_EQ(overlaps, expectedOverlaps);
}

TYPED_TEST(BvhTest, TraverseOverlappingPairsDoubleBvhDense) {
  using S = typename TestFixture::Type;

  testTraverseOverlappingPairsDoubleBvh<S>(1 << 4, 1 << 4);
  testTraverseOverlappingPairsDoubleBvh<S>(1 << 6, 1 << 6);
  testTraverseOverlappingPairsDoubleBvh<S>(1 << 8, 1 << 8);
  testTraverseOverlappingPairsDoubleBvh<S>(1 << 10, 1 << 10);
}

TYPED_TEST(BvhTest, TraverseOverlappingPairsDoubleBvhSparse) {
  using S = typename TestFixture::Type;

  testTraverseOverlappingPairsDoubleBvh<S>(1 << 4, 1 << 4, 10, 10);
  testTraverseOverlappingPairsDoubleBvh<S>(1 << 6, 1 << 6, 10, 10);
  testTraverseOverlappingPairsDoubleBvh<S>(1 << 8, 1 << 8, 10, 10);
  testTraverseOverlappingPairsDoubleBvh<S>(1 << 10, 1 << 10, 10, 10);
}

TYPED_TEST(BvhTest, Refit) {
  using S = typename TestFixture::Type;

  auto bvh = Bvh<S>();
  EXPECT_TRUE(bvh.checkBoundingBoxes());

  constexpr uint32_t kNumBoxes = 100;
  constexpr S kExtent = 10;
  constexpr S kSparsityFactor = 10;

  const auto [boxes, expectedOverlaps] =
      test::generateBoxesAndCollisions<S>(kNumBoxes, kExtent, kSparsityFactor);

  bvh.setBoundingBoxes(boxes);
  EXPECT_TRUE(bvh.checkBoundingBoxes());

  for (auto& box : bvh.getPrimitives()) {
    const auto randomA = Eigen::Vector3<S>::Random();
    const auto randomB = Eigen::Vector3<S>::Random();
    box = BoundingBox<S>(randomA.cwiseMin(randomB), randomA.cwiseMax(randomB), box.id);
  }

  bvh.refit();
  EXPECT_TRUE(bvh.checkBoundingBoxes());
}

} // namespace
} // namespace axel::test
