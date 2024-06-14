/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/marker.h>
#include <momentum/io/marker/coordinate_system.h>

namespace momentum {

/// Loads marker data from a TRC file into a MarkerSequence.
///
/// The function reads the given TRC file and stores the marker data as a MarkerSequence. The unit
/// of the input file will be read from the file and converted to the momentum unit (i.e., cm). The
/// UpVector parameter 'up' is the up-vector convention of the input marker file.
///
/// @param[in] filename The TRC file to load marker data from.
/// @param[in] up (Optional) The up-vector convention of the input marker file (default:
/// UpVector::Y).
/// @return MarkerSequence A MarkerSequence containing the marker data from the TRC file.
[[nodiscard]] MarkerSequence loadTrc(const std::string& filename, UpVector up = UpVector::Y);

} // namespace momentum
