/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/types.h>
#include <momentum/math/types.h>

#include <cstdint>
#include <vector>

namespace momentum {

inline static constexpr uint32_t kMaxSkinJoints = 8;

// matrix for skinning indices
using IndexMatrix =
    Eigen::Matrix<uint32_t, Eigen::Dynamic, kMaxSkinJoints, Eigen::AutoAlign | Eigen::RowMajor>;

// matrix for skinning weights
using WeightMatrix =
    Eigen::Matrix<float, Eigen::Dynamic, kMaxSkinJoints, Eigen::AutoAlign | Eigen::RowMajor>;

// structure for storing skinning weights
struct SkinWeights {
  IndexMatrix index; // list of vertices influenced by each joint
  WeightMatrix weight; // list of skinning weight stored by each joint

  void set(const std::vector<std::vector<size_t>>& ind, const std::vector<std::vector<float>>& wgt);

  bool operator==(const SkinWeights& skinWeights) const;
};

} // namespace momentum
