/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "axel/BvhFactory.h"

#include <gmock/gmock.h>

#include "axel/Bvh.h"

#ifdef AXEL_BVH_EMBREE
#include "axel/BvhEmbree.h"
#endif

namespace axel::test {
namespace {

using ::testing::IsNull;
using ::testing::Not;
using ::testing::WhenDynamicCastTo;

template <typename T>
const auto IsInstanceOf = WhenDynamicCastTo<const T*>(Not(IsNull()));

TEST(AccelerationStructureFactoryTest, CreatesCorrectBVH) {
  EXPECT_THAT(createBvh("fast-bvh").get(), IsInstanceOf<Bvhd>);

#ifdef AXEL_BVH_EMBREE
  EXPECT_THAT(createBvh("embree").get(), IsInstanceOf<BVHEmbree>);
  EXPECT_THAT(createBvh("default").get(), IsInstanceOf<BVHEmbree>);
#else
  EXPECT_THAT(createBvh("default").get(), IsInstanceOf<Bvhd>);
#endif
}
} // namespace
} // namespace axel::test
