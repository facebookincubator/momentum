/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/solver/fwd.h>
#include <momentum/solver/solver.h>

namespace momentum {

/// Gauss-Newton solver specific options.
struct GradientDescentSolverOptions : SolverOptions {
  /// Learning rate, fixed
  float learningRate = 0.01f;

  GradientDescentSolverOptions() = default;

  /* implicit */ GradientDescentSolverOptions(const SolverOptions& baseOptions)
      : SolverOptions(baseOptions) {
    // Empty
  }
};

template <typename T>
class GradientDescentSolverT : public SolverT<T> {
 public:
  GradientDescentSolverT(const SolverOptions& options, SolverFunctionT<T>* solver);

  [[nodiscard]] std::string_view getName() const override;

  void setOptions(const SolverOptions& options) final;

 protected:
  void doIteration() final;
  void initializeSolver() final;

 private:
  Eigen::VectorX<T> gradient_;
  float learningRate_;
};

} // namespace momentum
