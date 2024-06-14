/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/math/utility.h>

namespace momentum {

// Implementation of Mixture of Probabilistic PCA (http://www.miketipping.com/papers/met-mppca.pdf)
template <typename T>
struct MppcaT {
  // dimensions for convenience
  size_t d = 0;
  size_t p = 0;

  // names of the parameters in the structure
  std::vector<std::string> names;

  // we only need mu, Cinv, and Rpre for calculating the mppca efficiently
  // the actual base values of pi, W, and sigma2 are not stored
  Eigen::MatrixX<T> mu;
  std::vector<Eigen::MatrixX<T>> Cinv;
  std::vector<Eigen::MatrixX<T>> L;
  Eigen::VectorX<T> Rpre;

  // set data from loaded values
  void set(
      const VectorX<T>& pi,
      const MatrixX<T>& mmu,
      gsl::span<const MatrixX<T>> W,
      const VectorX<T>& sigma2);

  template <typename T2>
  MppcaT<T2> cast() const;

  bool isApprox(const MppcaT<T>& mppcaT) const;
};

} // namespace momentum
