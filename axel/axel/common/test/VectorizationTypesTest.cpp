/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "axel/common/VectorizationTypes.h"

#include <gtest/gtest.h>
#define DEFAULT_LOG_CHANNEL "Axel: VectorizationTypesTest"
#include "axel/Log.h"

namespace axel::test {
namespace {

TEST(VectorizationTypesTest, LaneWidthConstants) {
  if constexpr (has_avx512) {
    EXPECT_EQ(kNativeLaneWidth<double>, 8);
    EXPECT_EQ(kNativeLaneWidth<float>, 16);
    EXPECT_EQ(kNativeLaneWidth<int32_t>, 16);
  } else if constexpr (has_avx) {
    EXPECT_EQ(kNativeLaneWidth<double>, 4);
    EXPECT_EQ(kNativeLaneWidth<float>, 8);
    EXPECT_EQ(kNativeLaneWidth<int32_t>, 8);
  } else if constexpr (has_sse42 || has_neon) {
    EXPECT_EQ(kNativeLaneWidth<double>, 2);
    EXPECT_EQ(kNativeLaneWidth<float>, 4);
    EXPECT_EQ(kNativeLaneWidth<int32_t>, 4);
  } else {
    XR_LOGI(
        "No ISA detected. Are you compiling with Rosetta? Check your compilation flags. kNativeLaneWidth<float> = {}",
        kNativeLaneWidth<float>);
    EXPECT_EQ(kNativeLaneWidth<double>, 2);
    EXPECT_EQ(kNativeLaneWidth<float>, 4);
    EXPECT_EQ(kNativeLaneWidth<int32_t>, 4);
  }
}

} // namespace
} // namespace axel::test
