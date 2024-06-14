/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character/skeleton.h>
#include <momentum/test/helpers/expect_throw.h>

#include <gmock/gmock.h>

namespace momentum {

namespace {
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::SizeIs;

TEST(SkeletonTest, getJointIdIsInvalidOnEmpty) {
  const Skeleton skeleton;
  EXPECT_THAT(skeleton.getJointIdByName(std::string_view{"foo"}), Eq(kInvalidIndex));
}

TEST(SkeletonTest, getJointIdReturnsCorrectJoint) {
  const Joint jointA{"a", kInvalidIndex, Quaternionf::Identity(), Vector3f::Zero()};
  const Joint jointB{"b", kInvalidIndex, Quaternionf::Identity(), Vector3f::Zero()};
  const Skeleton skeleton{{jointA, jointB}};

  EXPECT_THAT(skeleton.getJointIdByName(jointA.name), Eq(0));
  EXPECT_THAT(skeleton.getJointIdByName(jointB.name), Eq(1));
}

TEST(SkeletonTest, getJointNamesIsEmptyOnEmpty) {
  const Skeleton skeleton;
  EXPECT_THAT(skeleton.getJointNames(), IsEmpty());
}

TEST(SkeletonTest, getJointNamesReturnsSortedNames) {
  const Joint jointA{"a", kInvalidIndex, Quaternionf::Identity(), Vector3f::Zero()};
  const Joint jointB{"b", kInvalidIndex, Quaternionf::Identity(), Vector3f::Zero()};
  const Skeleton skeleton{{jointA, jointB}};

  EXPECT_THAT(skeleton.getJointNames(), ElementsAre(jointA.name, jointB.name));
}

TEST(SkeletonTest, getChildrenJointsThrowsOnInvalidIndex) {
  const auto unitUnderTestFn = [] {
    const Skeleton skeleton;
    (void)skeleton.getChildrenJoints(0);
  };

  EXPECT_THROW_WITH_MESSAGE(unitUnderTestFn, std::out_of_range, HasSubstr("Out of bounds"));
}

TEST(SkeletonTest, getChildrenJointsGetsChildrenRecursivelyByDefault) {
  const Joint jointA{"a", kInvalidIndex, Quaternionf::Identity(), Vector3f::Zero()};
  const Joint jointB{"b", 0, Quaternionf::Identity(), Vector3f::Zero()};
  const Joint jointC{"c", 1, Quaternionf::Identity(), Vector3f::Zero()};
  const Joint jointD{"d", 2, Quaternionf::Identity(), Vector3f::Zero()};
  const Skeleton skeleton{{jointA, jointB, jointC, jointD}};

  EXPECT_THAT(skeleton.getChildrenJoints(1), ElementsAre(2, 3));
}

TEST(SkeletonTest, getChildrenJointsGetsDirectChild) {
  const Joint jointA{"a", kInvalidIndex, Quaternionf::Identity(), Vector3f::Zero()};
  const Joint jointB{"b", 0, Quaternionf::Identity(), Vector3f::Zero()};
  const Joint jointC{"c", 1, Quaternionf::Identity(), Vector3f::Zero()};
  const Joint jointD{"d", 2, Quaternionf::Identity(), Vector3f::Zero()};
  const Skeleton skeleton{{jointA, jointB, jointC, jointD}};

  constexpr bool kRecursive{false};
  EXPECT_THAT(skeleton.getChildrenJoints(1, kRecursive), ElementsAre(2));
}

TEST(SkeletonTest, getChildrenJointsMultipleRoots) {
  // 0 <-- 1 <-- x
  // 2 <-- 3 <-- 5 <-- x
  // 3 <-- x
  const Joint jointA{"a", kInvalidIndex, Quaternionf::Identity(), Vector3f::Zero()};
  const Joint jointB{"b", 0, Quaternionf::Identity(), Vector3f::Zero()};
  const Joint jointC{"c", kInvalidIndex, Quaternionf::Identity(), Vector3f::Zero()};
  const Joint jointD{"d", 2, Quaternionf::Identity(), Vector3f::Zero()};
  const Joint jointE{"e", kInvalidIndex, Quaternionf::Identity(), Vector3f::Zero()};
  const Joint jointF{"f", 3, Quaternionf::Identity(), Vector3f::Zero()};
  const Skeleton skeleton{{jointA, jointB, jointC, jointD, jointE, jointF}};

  constexpr bool kRecursive{true};
  EXPECT_THAT(skeleton.getChildrenJoints(0, kRecursive), ElementsAre(1));
  EXPECT_THAT(skeleton.getChildrenJoints(1, kRecursive), ElementsAre());
  EXPECT_THAT(skeleton.getChildrenJoints(2, kRecursive), ElementsAre(3, 5));
  EXPECT_THAT(skeleton.getChildrenJoints(3, kRecursive), ElementsAre(5));
  EXPECT_THAT(skeleton.getChildrenJoints(4, kRecursive), ElementsAre());
  EXPECT_THAT(skeleton.getChildrenJoints(5, kRecursive), ElementsAre());
}

TEST(SkeletonTest, CommonAncestor) {
  const Joint joint0{"a", kInvalidIndex, Quaternionf::Identity(), Vector3f::Zero()};
  const Joint joint1{"b", 0, Quaternionf::Identity(), Vector3f::Zero()};
  const Joint joint2{"c", 1, Quaternionf::Identity(), Vector3f::Zero()};
  const Joint joint3{"d", 2, Quaternionf::Identity(), Vector3f::Zero()};
  const Joint joint4{"e", 1, Quaternionf::Identity(), Vector3f::Zero()};
  const Joint joint5{"f", 0, Quaternionf::Identity(), Vector3f::Zero()};
  const Skeleton skeleton{{joint0, joint1, joint2, joint3, joint4, joint5}};

  EXPECT_THAT(skeleton.commonAncestor(1, 2), Eq(1));
  EXPECT_THAT(skeleton.commonAncestor(2, 1), Eq(1)); // should be symmetric
  EXPECT_THAT(skeleton.commonAncestor(0, 0), Eq(0));
  EXPECT_THAT(skeleton.commonAncestor(5, 3), Eq(0));
  EXPECT_THAT(skeleton.commonAncestor(4, 3), Eq(1));
  EXPECT_THAT(skeleton.commonAncestor(4, 0), Eq(0));
}

} // namespace
} // namespace momentum
