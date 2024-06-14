/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/io/marker/coordinate_system.h>
#include <momentum/math/types.h>

namespace momentum {

/// @brief This function converts a 3D vector in the mocap marker coordinate system to the Momentum
/// coordinate system.
///
/// The mocap marker coordinate system uses a spatial index with a specified up-axis (default Z),
/// right-handedness, and units specified in the 'unit' parameter (default millimeters). The
/// Momentum coordinate system uses a spatial index with Y as the up-axis, right-handedness, and
/// units in centimeters.
///
/// @tparam T The numeric type of the input and output 3D vectors.
/// @param[in] vec The 3D vector in the mocap marker coordinate system to be converted.
/// @param[in] up (Optional) The up-axis convention of the input marker coordinate system (default:
/// UpVector::Z).
/// @param[in] unit (Optional) The units of measurement used in the input marker coordinate system
/// (default: Unit::MM).
/// @returns A 3D vector in the Momentum coordinate system.
///
/// @note This conversion has only been tested for TRC and C3D formats.
template <typename T>
[[nodiscard]] Vector3<T>
toMomentumVector3(const Vector3<T>& vec, UpVector up = UpVector::Z, Unit unit = Unit::MM);

/// @brief This function converts a 3D vector in the mocap marker coordinate system to the Momentum
/// coordinate system, using a string to specify the unit.
///
/// The mocap marker coordinate system uses a spatial index with a specified up-axis (default Z),
/// right-handedness, and units specified in the 'unitStr' parameter (e.g., "m", "cm", "mm"). The
/// Momentum coordinate system uses a spatial index with Y as the up-axis, right-handedness, and
/// units in centimeters.
///
/// @tparam T The numeric type of the input and output 3D vectors.
/// @param[in] vec The 3D vector in the mocap marker coordinate system to be converted.
/// @param[in] up (Optional) The up-axis convention of the input marker coordinate system (default:
/// UpVector::Z).
/// @param[in] unitStr The string representation of the units of measurement used in the input
/// marker coordinate system.
/// @returns A 3D vector in the Momentum coordinate system.
///
/// @note This conversion has only been tested for TRC and C3D formats.
template <typename T>
[[nodiscard]] Vector3<T>
toMomentumVector3(const Vector3<T>& vec, UpVector up, const std::string& unitStr);

} // namespace momentum
