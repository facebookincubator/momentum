/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "axel/BvhEmbree.h"
#include "axel/Bvh.h"

#include <gmock/gmock.h>

#include <random>

namespace axel::test {
namespace {

// Temporary to minimize the changes.
using BBox = BoundingBoxd;

using ::testing::SizeIs;

TEST(BvhEmbreeTest, BoundingBoxQuery_NoHits) {
  std::vector<BBox> boxes = {
      BoundingBoxd(Eigen::Vector3d(0., 1., 2.), Eigen::Vector3d(1., 2., 3.)),
      BoundingBoxd(Eigen::Vector3d(1., 1., 2.), Eigen::Vector3d(5., 6., 7.)),
      BoundingBoxd(Eigen::Vector3d(2., 1., 2.), Eigen::Vector3d(3., 8., 9.)),
  };

  BVHEmbree bvh{};
  bvh.setBoundingBoxes(boxes);

  // No boxes will intersect with this one.
  const BBox queryBox{Eigen::Vector3d(-10., -10., -10.), Eigen::Vector3d(-9., -9., -9.)};
  BVHEmbree::QueryBuffer queryBuffer;
  const unsigned int hits{bvh.query(queryBox, queryBuffer)};
  ASSERT_EQ(hits, 0);

  const auto hitVec{bvh.query(queryBox)};
  ASSERT_EQ(hitVec.size(), hits);
  for (uint32_t i = 0; i < hits; ++i) {
    EXPECT_EQ(queryBuffer[i], hitVec[i]);
  }
}

TEST(BvhEmbreeTest, BoundingBoxQuery_AllHits) {
  std::vector<BBox> boxes = {
      BBox(Eigen::Vector3d(0., 1., 2.), Eigen::Vector3d(1., 2., 3.), 0),
      BBox(Eigen::Vector3d(1., 1., 2.), Eigen::Vector3d(5., 6., 7.), 1),
      BBox(Eigen::Vector3d(2., 1., 2.), Eigen::Vector3d(3., 8., 9.), 2),
  };

  BVHEmbree bvh{};
  bvh.setBoundingBoxes(boxes);

  // All of the boxes will intersect with this large one.
  const BBox queryBox{Eigen::Vector3d(-10., -10., -10.), Eigen::Vector3d(9., 9., 9.)};
  BVHEmbree::QueryBuffer queryBuffer;
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

TEST(BvhEmbreeTest, MoveSemantics) {
  std::vector<BBox> boxes = {
      BBox(Eigen::Vector3d(0., 1., 2.), Eigen::Vector3d(1., 2., 3.), 0),
      BBox(Eigen::Vector3d(1., 1., 2.), Eigen::Vector3d(5., 6., 7.), 1),
      BBox(Eigen::Vector3d(2., 1., 2.), Eigen::Vector3d(3., 8., 9.), 2),
  };

  BVHEmbree bvh{};
  bvh.setBoundingBoxes(boxes);

  // All of the boxes will intersect with this large one.
  const BBox queryBox{Eigen::Vector3d(-10., -10., -10.), Eigen::Vector3d(9., 9., 9.)};
  const auto hitVec{bvh.query(queryBox)};
  ASSERT_EQ(hitVec.size(), 3);

  // Move constructor.
  BVHEmbree bvh2{std::move(bvh)};
  const auto hitVec2{bvh2.query(queryBox)};
  ASSERT_EQ(hitVec, hitVec2);
  ASSERT_EQ(bvh.query(queryBox).size(), 0); // Was moved-from, so it's "empty". NOLINT

  // Move assignment.
  BVHEmbree bvh3{};
  bvh3 = std::move(bvh2);
  const auto hitVec3{bvh3.query(queryBox)};
  ASSERT_EQ(hitVec, hitVec3);
  ASSERT_EQ(bvh2.query(queryBox).size(), 0); // Was moved-from, so it's "empty". NOLINT

  // No observable change.
  BVHEmbree& bvh3Ref{bvh3};
  bvh3 = std::move(bvh3Ref);
  const auto anotherHitVec3{bvh3.query(queryBox)};
  ASSERT_EQ(hitVec, anotherHitVec3);
}

TEST(BvhEmbreeTest, RayQuery_AllHits) {
  const std::vector<BBox> boxes = {
      BBox(Eigen::Vector3d(0., 0., 0.), Eigen::Vector3d(1., 1., 1.), 0),
      BBox(Eigen::Vector3d(2., 0.25, 0.25), Eigen::Vector3d(3., 1.25, 1.25), 1),
      BBox(Eigen::Vector3d(4., -0.25, 0.), Eigen::Vector3d(5., 1.25, 1.), 2),
  };

  BVHEmbree bvhEmbree{};
  bvhEmbree.setBoundingBoxes(boxes);

  const Eigen::Vector3d rayOrigin{-1.0, 0.5, 0.5};
  const Eigen::Vector3d rayDir{1.0, 0.0, 0.0};
  const auto hits{bvhEmbree.query(rayOrigin, rayDir)};
  EXPECT_EQ(hits.size(), 3);
}

TEST(BvhEmbreeTest, RayQuery_NoHits) {
  const std::vector<BBox> boxes = {
      BBox(Eigen::Vector3d(0., 0., 0.), Eigen::Vector3d(1., 1., 1.), 0),
      BBox(Eigen::Vector3d(2., 0.25, 0.25), Eigen::Vector3d(3., 1.25, 1.25), 1),
      BBox(Eigen::Vector3d(4., -0.25, 0.), Eigen::Vector3d(5., 1.25, 1.), 2),
  };

  BVHEmbree bvhEmbree{};
  bvhEmbree.setBoundingBoxes(boxes);

  const Eigen::Vector3d rayOrigin{-1.0, 0.5, 0.5};
  const Eigen::Vector3d rayDir{-1.0, 0.0, 0.0};
  const auto hits{bvhEmbree.query(rayOrigin, rayDir)};
  EXPECT_EQ(hits.size(), 0);
}

TEST(BvhEmbreeTest, RayQuery_NaNs) {
  const std::vector<BBox> boxes = {
      BBox(Eigen::Vector3d(0., 0., 0.), Eigen::Vector3d(1., 1., 1.), 0),
      BBox(Eigen::Vector3d(2., 0.25, 0.25), Eigen::Vector3d(3., 1.25, 1.25), 1),
      BBox(Eigen::Vector3d(4., -0.25, 0.), Eigen::Vector3d(5., 1.25, 1.), 2),
  };

  BVHEmbree bvhEmbree{};
  bvhEmbree.setBoundingBoxes(boxes);

  const Eigen::Vector3d rayOrigin{0.0, 0.0, 0.0};
  const Eigen::Vector3d rayDir{-1.0, 0.0, 0.0};
  EXPECT_THAT(bvhEmbree.query(rayOrigin, rayDir), SizeIs(1));
}

TEST(BvhTest, Query_EmptyBVH_NoHits) {
  BVHEmbree bvh{};

  const BBox bbox(Eigen::Vector3d(0., 0., 0.), Eigen::Vector3d(1., 1., 1.), 0);
  const Eigen::Vector3d rayOrigin{-1.0, 0.5, 0.5};
  const Eigen::Vector3d rayDir{-1.0, 0.0, 0.0};

  const auto hits{bvh.query(rayOrigin, rayDir)};
  EXPECT_EQ(hits.size(), 0);

  const auto boxHits{bvh.query(bbox)};
  EXPECT_EQ(boxHits.size(), 0);

  BVHEmbree::QueryBuffer queryBuffer;
  EXPECT_EQ(bvh.query(bbox, queryBuffer), 0);
}

TEST(BvhEmbreeTest, RayQuery_RandomParityCheck) {
  std::default_random_engine eng{42};
  std::uniform_real_distribution<double> dist{};

  // Set up bounding boxes that will be shared between BVH and BVHEmbree.
  std::vector<BBox> boxes{};
  for (uint32_t i = 0; i < 100; ++i) {
    const Eigen::Vector3d size{Eigen::Vector3d{dist(eng), dist(eng), dist(eng)} * 1.0};
    const Eigen::Vector3d minCorner{Eigen::Vector3d{dist(eng), dist(eng), dist(eng)} * 100.0};
    boxes.emplace_back(minCorner, minCorner + size, i);
  }

  // Set up test rays.
  std::vector<Eigen::Vector3d> origins;
  std::vector<Eigen::Vector3d> directions;
  for (uint32_t i = 0; i < 100; ++i) {
    origins.emplace_back(Eigen::Vector3d{dist(eng), dist(eng), dist(eng)} * 10.0);
    directions.emplace_back(dist(eng), dist(eng), dist(eng));
    directions.back().normalize();
  }

  Bvhd bvh{};
  bvh.setBoundingBoxes(boxes);

  BVHEmbree bvhEmbree{};
  bvhEmbree.setBoundingBoxes(boxes);

  // We expect all the results to match up.
  for (uint32_t i = 0; i < origins.size(); ++i) {
    const auto hits{bvh.query(origins.at(i), directions.at(i))};
    const auto embreeHits{bvhEmbree.query(origins.at(i), directions.at(i))};
    ASSERT_EQ(hits.size(), embreeHits.size());
    for (uint32_t k = 0; k < hits.size(); ++k) {
      EXPECT_EQ(hits.at(k), embreeHits.at(k));
    }
  }
}

TEST(BvhEmbreeTest, SetEmpty) {
  std::default_random_engine eng{42};
  std::uniform_real_distribution<double> dist{};

  // Set up bounding boxes that will be shared between BVH and BVHEmbree.
  std::vector<BBox> boxes{};
  for (uint32_t i = 0; i < 100; ++i) {
    const Eigen::Vector3d size{Eigen::Vector3d{dist(eng), dist(eng), dist(eng)} * 1.0};
    const Eigen::Vector3d minCorner{Eigen::Vector3d{dist(eng), dist(eng), dist(eng)} * 100.0};
    boxes.emplace_back(minCorner, minCorner + size, i);
  }
  BVHEmbree bvhEmbree{};
  bvhEmbree.setBoundingBoxes(boxes);

  EXPECT_EQ(bvhEmbree.getPrimitiveCount(), boxes.size());

  bvhEmbree.setBoundingBoxes({});
  EXPECT_EQ(bvhEmbree.getPrimitiveCount(), 0);
}

} // namespace
} // namespace axel::test
