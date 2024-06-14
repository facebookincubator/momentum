/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/character/locator.h>
#include <momentum/character/types.h>

namespace momentum {

// base locatorstate class
struct LocatorState {
  std::vector<Vector3f>
      position; // the current position of all locators according to a given skeleton state

 public:
  LocatorState() noexcept {

  };

  LocatorState(const SkeletonState& skeletonState, const LocatorList& referenceLocators) noexcept {
    update(skeletonState, referenceLocators);
  };

  void update(const SkeletonState& skeletonState, const LocatorList& referenceLocators) noexcept;
};

} // namespace momentum
