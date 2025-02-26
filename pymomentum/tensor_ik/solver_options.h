// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>

namespace pymomentum {

enum class LinearSolverType { Cholesky, QR, TrustRegionQR };

inline const char* toString(LinearSolverType solverType) {
  switch (solverType) {
    default:
    case LinearSolverType::Cholesky:
      return "Cholesky";
    case LinearSolverType::QR:
      return "QR";
    case LinearSolverType::TrustRegionQR:
      return "TrustRegionQR";
  }
}

struct SolverOptions {
  LinearSolverType linearSolverType = LinearSolverType::QR;
  float levmar_lambda = 0.01f;
  size_t minIter = 4;
  size_t maxIter = 50;
  float threshold = 10.0f;
  bool lineSearch = true;

  bool operator==(const SolverOptions& rhs) const {
    return linearSolverType == rhs.linearSolverType &&
        levmar_lambda == rhs.levmar_lambda && minIter == rhs.minIter &&
        maxIter == rhs.maxIter && threshold == rhs.threshold &&
        lineSearch == rhs.lineSearch;
  }
};

} // namespace pymomentum
