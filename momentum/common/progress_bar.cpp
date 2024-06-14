/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/common/progress_bar.h"

#include <iostream>

namespace momentum {

ProgressBar::ProgressBar(const std::string& name, const int64_t numOperations, bool visible)
    : numHashes_(std::max(maxWidth - (int64_t)name.size() - 1, (int64_t)50)),
      numOperations_(numOperations),
      visible_(visible) {
  if (visible_) {
    std::cout << name << " " << std::flush;
  }
}

void ProgressBar::increment(int64_t count) {
  set(curOp_ + count);
}

void ProgressBar::set(int64_t count) {
  curOp_ = count;
  const int64_t expectedPrinted = (curOp_ * numHashes_) / numOperations_;

  while (numHashesPrinted_ < expectedPrinted) {
    if (visible_) {
      std::cout.put('#');
      std::cout.flush();
    }
    ++numHashesPrinted_;
  }
}

ProgressBar::~ProgressBar() {
  const size_t nBackspace = maxWidth - (numHashes_ - numHashesPrinted_);
  if (visible_) {
    for (size_t i = 0; i < nBackspace; ++i) {
      std::cout << "\b";
    }
    for (size_t i = 0; i < nBackspace; ++i) {
      std::cout << " ";
    }
    std::cout << "\r" << std::flush;
  }
}

} // namespace momentum
