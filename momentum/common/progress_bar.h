/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>

namespace momentum {

// A simple progress bar that prints hash marks, e.g.
//   Reticulating splines #####################################
class ProgressBar {
  const int64_t maxWidth = 80;

 public:
  ProgressBar(const std::string& name, int64_t numOperations, bool visible = false);
  void increment(int64_t count = 1);
  void set(int64_t count);
  ~ProgressBar();

 private:
  const int64_t numHashes_;
  const int64_t numOperations_;
  const bool visible_;

  int64_t numHashesPrinted_ = 0;
  int64_t curOp_ = 0;
};

} // namespace momentum
