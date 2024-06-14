/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/common/aligned.h"

#include <gtest/gtest.h>

using namespace momentum;

// Tests for makeAlignedUnique
TEST(AlignedTest, MakeAlignedUnique_AllocatesAndDeallocatesAlignedMemory) {
  constexpr std::size_t Alignment = 16;
  constexpr std::size_t Count = 10;

  auto alignedPtr = makeAlignedUnique<int, Alignment>(Count);
  int* ptr = alignedPtr.get();

  ASSERT_NE(ptr, nullptr);

  // Ensure the alignment is correct
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % Alignment, 0);

  // Modify the memory
  for (std::size_t i = 0; i < Count; ++i) {
    ptr[i] = static_cast<int>(i);
  }

  // Verify the content
  for (std::size_t i = 0; i < Count; ++i) {
    EXPECT_EQ(ptr[i], static_cast<int>(i));
  }
}

// Tests for makeAlignedShared
TEST(AlignedTest, MakeAlignedShared_AllocatesAndDeallocatesAlignedMemory) {
  constexpr std::size_t Alignment = 16;
  constexpr std::size_t Count = 10;

  auto alignedPtr = makeAlignedShared<int, Alignment>(Count);
  int* ptr = alignedPtr.get();

  ASSERT_NE(ptr, nullptr);

  // Ensure the alignment is correct
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % Alignment, 0);

  // Modify the memory
  for (std::size_t i = 0; i < Count; ++i) {
    ptr[i] = static_cast<int>(i);
  }

  // Verify the content
  for (std::size_t i = 0; i < Count; ++i) {
    EXPECT_EQ(ptr[i], static_cast<int>(i));
  }
}

// Tests for makeAlignedUnique with invalid allocation
TEST(AlignedTest, MakeAlignedUniqueTest_ThrowsBadAllocOnInvalidAllocation) {
  constexpr std::size_t Count = std::numeric_limits<std::size_t>::max();
  auto makeAligned = [&]() { (void)makeAlignedUnique<int, 16>(Count); };
  ASSERT_THROW(makeAligned(), std::bad_array_new_length);
}

// Tests for makeAlignedShared with invalid allocation
TEST(AlignedTest, MakeAlignedSharedTest_ThrowsBadAllocOnInvalidAllocation) {
  constexpr std::size_t Count = std::numeric_limits<std::size_t>::max();
  auto makeAligned = [&]() { (void)makeAlignedShared<int, 16>(Count); };
  ASSERT_THROW(makeAligned(), std::bad_array_new_length);
}

// Tests for makeAlignedUnique with custom deleter
TEST(AlignedTest, MakeAlignedUnique_UsesCustomDeleter) {
  constexpr std::size_t Count = 10;
  bool wasDeleted = false;

  {
    auto deleter = [&wasDeleted](int* ptr) {
      aligned_free(ptr);
      wasDeleted = true;
    };

    auto alignedPtr = makeAlignedUnique<int, 16, decltype(deleter)>(Count, deleter);
  }

  ASSERT_TRUE(wasDeleted);
}

// Tests for makeAlignedShared with custom deleter
TEST(AlignedTest, MakeAlignedShared_UsesCustomDeleter) {
  constexpr std::size_t Count = 10;
  bool wasDeleted = false;

  {
    auto deleter = [&wasDeleted](int* ptr) {
      aligned_free(ptr);
      wasDeleted = true;
    };

    auto alignedPtr = makeAlignedShared<int, 16, decltype(deleter)>(Count, deleter);
  }

  ASSERT_TRUE(wasDeleted);
}

// Test that the allocator can handle a large allocation request.
TEST(AlignedTest, AlignedAllocator_CanAllocateLargeAmount) {
  AlignedAllocator<int, 16> allocator;
  int* ptr = allocator.allocate(1e6); // Allocate space for 1 million ints
  ASSERT_NE(ptr, nullptr);
  allocator.deallocate(ptr, 1e6);
}

// Test that the allocator throws an exception when asked to allocate an excessively large amount of
// memory.
TEST(AlignedTest, AlignedAllocator_ThrowsOnExcessiveAllocation) {
  AlignedAllocator<int, 16> allocator;
  ASSERT_THROW(
      (void)allocator.allocate(std::numeric_limits<std::size_t>::max()), std::bad_array_new_length);
}

// Test that the allocator works correctly when used with an STL container.
TEST(AlignedTest, AlignedAllocator_WorksInSTLContainer) {
  std::vector<int, AlignedAllocator<int, 16>> vec;
  vec.reserve(1000);
  for (int i = 0; i < 1000; ++i) {
    vec.push_back(i);
  }
  for (int i = 0; i < 1000; ++i) {
    EXPECT_EQ(vec[i], i);
  }
}

// Test the allocator's rebind functionality, which is necessary for use with many STL containers.
TEST(AlignedTest, AlignedAllocator_RebindWorks) {
  AlignedAllocator<int, 16>::template rebind<double>::other double_allocator;
  double* ptr = double_allocator.allocate(10);
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % 16, 0);
  double_allocator.deallocate(ptr, 10);
}

// Test if the allocator properly aligns the memory with a different alignment.
TEST(AlignedTest, AlignedAllocator_DifferentAlignmentCheck) {
  AlignedAllocator<int, 64> allocator;
  int* ptr = allocator.allocate(10);
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % 64, 0);
  allocator.deallocate(ptr, 10);
}
