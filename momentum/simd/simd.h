/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/common/exception.h>

#include <drjit/array.h>
#include <drjit/array_router.h>
#include <drjit/packet.h>
#include <drjit/util.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

// Utilities for writing cross-platform SIMD code.
// This currently uses the DrJit library for SIMD primitives.

namespace momentum {

inline constexpr size_t kAvxPacketSize = 8;
inline constexpr size_t kAvxAlignment = kAvxPacketSize * sizeof(float);

inline constexpr size_t kSimdPacketSize = drjit::DefaultSize;
inline constexpr size_t kSimdAlignment = kSimdPacketSize * sizeof(float);

template <typename T>
using Packet = drjit::Packet<T, kSimdPacketSize>;

using FloatP = Packet<float>;
using DoubleP = Packet<double>;
using IntP = Packet<int>;

template <typename T, int Dim>
using VectorP = drjit::Array<Packet<T>, Dim>;

template <typename T>
using Vector2P = VectorP<T, 2>;

template <typename T>
using Vector3P = VectorP<T, 3>;

using Vector2fP = Vector2P<float>;
using Vector3fP = Vector3P<float>;

using Vector2dP = Vector2P<double>;
using Vector3dP = Vector3P<double>;

/// Checks if the data of the matrix is aligned correctly.
template <size_t Alignment>
void checkAlignment(const Eigen::Ref<Eigen::MatrixXf>& mat) {
  MT_THROW_IF(
      (uintptr_t(mat.data())) % Alignment != 0,
      "Matrix ({}x{}, ptr: {}) is not aligned ({}) correctly.",
      mat.rows(),
      mat.cols(),
      uintptr_t(mat.data()),
      Alignment);
}

/// Calculates dot product of Eigen::Vector3f and 3-vector of packets.
///
/// @tparam Derived A derived class of Eigen::MatrixBase, which should be compatible with
/// Eigen::Vector3f.
/// @param[in] v1 A 3D vector of Eigen.
/// @param[in] v2 A 3-vector of packets.
template <typename Derived, typename S>
FloatP dot(const Eigen::MatrixBase<Derived>& v1, const Vector3P<S>& v2) {
  return drjit::fmadd(v1.x(), v2.x(), drjit::fmadd(v1.y(), v2.y(), v1.z() * v2.z()));
}

/// Calculates dot product of Vector3P and Eigen::Vector3f.
///
/// @tparam Derived A derived class of Eigen::MatrixBase, which should be compatible with
/// Eigen::Vector3f.
/// @param[in] v1 A 3-vector of packets.
/// @param[in] v2 A 3D vector of Eigen.
template <typename Derived, typename S>
FloatP dot(const Vector3P<S>& v1, const Eigen::MatrixBase<Derived>& v2) {
  return drjit::fmadd(v1.x(), v2.x(), drjit::fmadd(v1.y(), v2.y(), v1.z() * v2.z()));
}

/// Calculates summation of Eigen::Vector3f and Vector3P
///
/// @tparam Derived A derived class of Eigen::MatrixBase, which should be compatible with
/// Eigen::Vector3f.
/// @param[in] v1 A 3D vector of Eigen.
/// @param[in] v2 A 3-vector of packets.
template <typename S, typename Derived>
Vector3P<S> operator+(const Eigen::MatrixBase<Derived>& v1, const Vector3P<S>& v2) {
  return {v1.x() + v2.x(), v1.y() + v2.y(), v1.z() + v2.z()};
}

/// Calculates summation of Vector3P and Eigen::Vector3f
///
/// @tparam Derived A derived class of Eigen::MatrixBase, which should be compatible with
/// Eigen::Vector3f.
/// @param[in] v1 A 3-vector of packets.
/// @param[in] v2 A 3D vector of Eigen.
template <typename S, typename Derived>
Vector3P<S> operator+(const Vector3P<S>& v1, const Eigen::MatrixBase<Derived>& v2) {
  return {v1.x() + v2.x(), v1.y() + v2.y(), v1.z() + v2.z()};
}

/// Calculates subtraction of Eigen::Vector3f and Vector3P
///
/// @tparam Derived A derived class of Eigen::MatrixBase, which should be compatible with
/// Eigen::Vector3f.
/// @param[in] v1 A 3D vector of Eigen.
/// @param[in] v2 A 3-vector of packets.
template <typename S, typename Derived>
Vector3P<S> operator-(const Eigen::MatrixBase<Derived>& v1, const Vector3P<S>& v2) {
  return {v1.x() - v2.x(), v1.y() - v2.y(), v1.z() - v2.z()};
}

/// Calculates subtraction of Vector3P and Eigen::Vector3f
///
/// @tparam Derived A derived class of Eigen::MatrixBase, which should be compatible with
/// Eigen::Vector3f.
/// @param[in] v1 A 3-vector of packets.
/// @param[in] v2 A 3D vector of Eigen.
template <typename S, typename Derived>
Vector3P<S> operator-(const Vector3P<S>& v1, const Eigen::MatrixBase<Derived>& v2) {
  return {v1.x() - v2.x(), v1.y() - v2.y(), v1.z() - v2.z()};
}

/// Calculates multiplication of 3x3 matrix and each 3x1 vector in packet
///
/// @tparam Derived A derived class of Eigen::MatrixBase, which should be compatible with
/// Eigen::Matrix3f.
/// @param[in] xf A 3x3 matrix.
/// @param[in] vec A 3-vector of packets.
template <typename S, typename Derived>
Vector3P<S> operator*(const Eigen::MatrixBase<Derived>& xf, const Vector3P<S>& vec) {
  return Vector3P<S>{
      momentum::dot(xf.row(0), vec), momentum::dot(xf.row(1), vec), momentum::dot(xf.row(2), vec)};
}

/// Calculates affine transformation on each 3x1 vector in packet
///
/// @param[in] xf An affine transformation matrix.
/// @param[in] vec A 3-vector of packets.
template <typename S>
Vector3P<S> operator*(const Eigen::Transform<S, 3, Eigen::Affine>& xf, const Vector3P<S>& vec) {
  return momentum::operator+(momentum::operator*(xf.linear(), vec), xf.translation());
}

/// Calculates cross product of Eigen::Vector3f and Vector3P
///
/// @tparam Derived A derived class of Eigen::MatrixBase, which should be compatible with
/// Eigen::Vector3f.
/// @param[in] v1 A 3D vector of Eigen.
/// @param[in] v2 A 3-vector of packets.
template <typename S, typename Derived>
Vector3P<S> cross(const Eigen::MatrixBase<Derived>& v1, const Vector3P<S>& v2) {
  return {
      drjit::fmsub(v1.y(), v2.z(), v1.z() * v2.y()),
      drjit::fmsub(v1.z(), v2.x(), v1.x() * v2.z()),
      drjit::fmsub(v1.x(), v2.y(), v1.y() * v2.x()),
  };
}

/// Calculates cross product of Vector3P and Eigen::Vector3f
///
/// @tparam Derived A derived class of Eigen::MatrixBase, which should be compatible with
/// Eigen::Vector3f.
/// @param[in] v1 A 3-vector of packets.
/// @param[in] v2 A 3D vector of Eigen.
template <typename S, typename Derived>
Vector3P<S> cross(const Vector3P<S>& v1, const Eigen::MatrixBase<Derived>& v2) {
  return {
      drjit::fmsub(v1.y(), v2.z(), v1.z() * v2.y()),
      drjit::fmsub(v1.z(), v2.x(), v1.x() * v2.z()),
      drjit::fmsub(v1.x(), v2.y(), v1.y() * v2.x()),
  };
}

} // namespace momentum
