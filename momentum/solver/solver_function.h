/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/math/types.h>
#include <momentum/solver/fwd.h>

#include <unordered_map>

namespace momentum {

/// Abstract solver function class.
template <typename T>
class SolverFunctionT {
 public:
  virtual ~SolverFunctionT() = default;

  /// Computes the error given a set of parameters.
  virtual double getError(const VectorX<T>& parameters) = 0;

  /// Computes the gradient given a set of parameters.
  virtual double getGradient(const VectorX<T>& parameters, VectorX<T>& gradient) = 0;

  /// Computes the Jacobian matrix, residual vector and actual row count given a set of parameters.
  virtual double getJacobian(
      const VectorX<T>& parameters,
      MatrixX<T>& jacobian,
      VectorX<T>& residual,
      size_t& actualRows) = 0;

  /// Computes the Hessian matrix given a set of parameters.
  virtual void getHessian(const VectorX<T>& parameters, MatrixX<T>& hessian);

  /// Computes the JtJ matrix and JtR vector given a set of parameters.
  virtual double getJtJR(const VectorX<T>& parameters, MatrixX<T>& jtj, VectorX<T>& jtr);

  /// Computes the sparse JtJ matrix and JtR vector given a set of parameters.
  virtual double
  getJtJR_Sparse(const VectorX<T>& parameters, SparseMatrix<T>& jtj, VectorX<T>& jtr);

  /// Computes the derivatives (Hessian and gradient) or their approximation ready for the solver.
  virtual double
  getSolverDerivatives(const VectorX<T>& parameters, MatrixX<T>& hess, VectorX<T>& grad);

  /// Updates the parameters given the gradient.
  virtual void updateParameters(VectorX<T>& parameters, const VectorX<T>& gradient) = 0;

  /// Sets the enabled parameters.
  virtual void setEnabledParameters(const ParameterSet& parameterSet);

  /// Returns the total number of parameters.
  size_t getNumParameters() const;

  /// Returns the last active parameter count.
  size_t getActualParameters() const;

  /// Stores the iteration history.
  virtual void storeHistory(
      std::unordered_map<std::string, MatrixX<T>>& history,
      size_t iteration,
      size_t maxIterations_);

 protected:
  /// The total number of parameters
  size_t numParameters_{};

  // Last active parameter, i.e. actualParameters <= numParameters
  size_t actualParameters_{};
};

} // namespace momentum
