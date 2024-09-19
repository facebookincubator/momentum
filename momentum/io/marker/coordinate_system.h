/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace momentum {

/// This enumeration is used to define the primary up vector directions for various coordinate
/// systems and orientations in a 3D environment.
enum class UpVector {
  X = 0, ///< X-axis up vector
  Y, ///< Y-axis up vector
  Z, ///< Z-axis up vector
  YNeg, ///< Negative Y-axis up vector
};

/// An enumeration representing the units of measurement used in marker data files.
enum class Unit {
  M, ///< Meters
  DM, ///< Decimeter
  CM, ///< Centimeters
  MM, ///< Millimeters
  Unknown, ///< Unknown unit of measurement
};

} // namespace momentum
