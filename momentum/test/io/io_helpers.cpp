/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/test/io/io_helpers.h"

#include <fmt/format.h>

#include <atomic>

namespace momentum {

TemporaryFile temporaryFile(const std::string& prefix, const std::string& extension) {
  static std::atomic<int> calls = 0;
  return TemporaryFile(
      fmt::format("{}_{}_{}", prefix, GetPID(), calls++), extension, temporaryDirectory());
}

TemporaryDirectory temporaryDirectory(const std::string& prefix) {
  static std::atomic<int> calls = 0;
  return TemporaryDirectory(
      fmt::format("{}_{}_{}", prefix, GetPID(), calls++), temporaryDirectory());
}

} // namespace momentum
