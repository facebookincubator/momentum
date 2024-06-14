/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/math/covariance_matrix.h"

#include "momentum/common/checks.h"

namespace momentum {

template <typename T>
void LowRankCovarianceMatrixT<T>::reset(T sigma, Eigen::Ref<const Eigen::MatrixX<T>> A) {
  A_ = A;
  sigma_ = sigma;

  const Eigen::Index n = A_.cols();
  const Eigen::Index m = A_.rows();

  qrFactorization_.reset(n, sigma);
  Eigen::VectorX<T> b = Eigen::VectorX<T>::Zero(m);
  qrFactorization_.add(A_, b);
}

template <typename T>
Eigen::Index LowRankCovarianceMatrixT<T>::dimension() const {
  return A_.cols();
}

template <typename T>
const Eigen::MatrixX<T>& LowRankCovarianceMatrixT<T>::basis() const {
  return A_;
}

template <typename T>
T LowRankCovarianceMatrixT<T>::sigma() const {
  return sigma_;
}

template <typename T>
Eigen::VectorX<T> LowRankCovarianceMatrixT<T>::inverse_times_vec(
    Eigen::Ref<const Eigen::VectorX<T>> rhs) const {
  const auto& R = qrFactorization_.R();
  MT_CHECK(R.rows() == rhs.size());

  return R.template triangularView<Eigen::Upper>().solve(
      R.template triangularView<Eigen::Upper>().transpose().solve(rhs));
}

template <typename T>
Eigen::VectorX<T> LowRankCovarianceMatrixT<T>::times_vec(
    Eigen::Ref<const Eigen::VectorX<T>> rhs) const {
  Eigen::VectorX<T> result = (sigma_ * sigma_) * rhs;
  result.noalias() += A_.transpose() * (A_ * rhs);
  return result;
}

template <typename T>
Eigen::MatrixX<T> LowRankCovarianceMatrixT<T>::inverse_times_mat(
    Eigen::Ref<const Eigen::MatrixX<T>> rhs) const {
  const auto& R = qrFactorization_.R();
  MT_CHECK(rhs.rows() == dimension());

  return R.template triangularView<Eigen::Upper>().solve(
      R.template triangularView<Eigen::Upper>().transpose().solve(rhs));
}

template <typename T>
Eigen::MatrixX<T> LowRankCovarianceMatrixT<T>::times_mat(
    Eigen::Ref<const Eigen::MatrixX<T>> rhs) const {
  Eigen::MatrixX<T> result = (sigma_ * sigma_) * rhs;
  result.noalias() += A_.transpose() * (A_ * rhs);
  return result;
}

template <typename T>
const Eigen::MatrixX<T>& LowRankCovarianceMatrixT<T>::R() const {
  return qrFactorization_.R();
}

template <typename T>
T LowRankCovarianceMatrixT<T>::logDeterminant() const {
  // The determinant of a
  const auto& R = qrFactorization_.R();

  // The determinant of a triangular matrix is the product of the
  // diagonals, so the log of the determinant is the sum of the logs:
  T result = 0;
  for (Eigen::Index i = 0; i < dimension(); ++i) {
    MT_CHECK(R(i, i) > 0);
    result += std::log(R(i, i));
  }

  // We actually have A = R^T*R, and the determinant of a product is the
  // product of the determinants, so we need to double it:
  return T(2) * result;
}

template <typename T>
T LowRankCovarianceMatrixT<T>::inverse_logDeterminant() const {
  return -logDeterminant();
}

template class LowRankCovarianceMatrixT<float>;
template class LowRankCovarianceMatrixT<double>;

} // namespace momentum
