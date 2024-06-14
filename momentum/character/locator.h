/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/parameter_transform.h>
#include <momentum/math/utility.h>

#include <gsl/span>

namespace momentum {

// base locator representation
struct Locator // a single locator
{
  std::string name; // name of the locator
  size_t parent; // parent joint of the locator
  Vector3f offset; // relative offset to the parent
  Vector3i locked; // defines which axes are moveable
  float weight; // weight of the locator
  Vector3f limitOrigin; // defines the limit reference position. equal to offset on loading
  Vector3f limitWeight; // defines how close an unlocked locator should stay to it's original
                        // position (0 = free)

  Locator(
      const std::string& name = "uninitialized",
      const size_t parent = kInvalidIndex,
      const Vector3f& offset = Vector3f::Zero(),
      const Vector3i& locked = Vector3i::Zero(),
      const float weight = 1.0f,
      const Vector3f& limitOrigin = Vector3f::Zero(),
      const Vector3f& limitWeight = Vector3f::Zero())
      : name(name),
        parent(parent),
        offset(offset),
        locked(locked),
        weight(weight),
        limitOrigin(limitOrigin),
        limitWeight(limitWeight) {}

  inline bool operator==(const Locator& locator) const {
    return (
        (name == locator.name) && (parent == locator.parent) && offset.isApprox(locator.offset) &&
        locked.isApprox(locator.locked) && isApprox(weight, locator.weight) &&
        limitOrigin.isApprox(locator.limitOrigin) && limitWeight.isApprox(locator.limitWeight));
  };
};

using LocatorList = std::vector<Locator>; // a list of locators attached to a skeleton

} // namespace momentum
