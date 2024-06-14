/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <rerun.hpp>

#include <string>

namespace momentum {

/// Redirects logs from the XR_LOG framework to the Rerun logger.
///
/// This function registers a custom log sink for the XR_LOG framework, redirecting its log messages
/// to the Rerun logger. The redirection is only performed once, and subsequent calls to this
/// function have no effect.
///
/// @param[in] rec The recording stream to which to write the logs. You should make sure the
/// lifetime of this is longer than the loggings from momentum libraries.
/// @return Returns true if the redirection is successful or has already been done.
bool redirectLogsToRerun(const rerun::RecordingStream& rec);

} // namespace momentum
