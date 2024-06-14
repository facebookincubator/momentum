/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/math/constants.h>
#include <momentum/math/types.h>

#include <gsl/gsl>

#ifdef _WIN32
#include <malloc.h>
#endif // _WIN32

namespace momentum {

// isnan() function that isn't optimized away by fastmath, unlike std::isnan()
[[nodiscard]] inline bool IsNanNoOpt(float f) {
  union {
    float f;
    std::uint32_t x;
  } u = {f};
  return (u.x << 1) > 0xff000000u;
}

// -----------------------------------------------
//  math utilities
// -----------------------------------------------

// A ParameterSet where all parameters are active.
inline const ParameterSet allParams() {
  ParameterSet params;
  params.set();
  return params;
}

// simple square function because obviously c++ doesn't have one
template <class T>
inline T sqr(const T& x) {
  return x * x;
};

/// Determines if two values are approximately equal within a given tolerance.
template <typename T = float>
[[nodiscard]] bool isApprox(T l, T r, T eps = Eps<T>(1e-4, 1e-6)) noexcept {
  return (std::abs(l - r) < eps);
};

// -----------------------------------------------
//  matrix utilities
// -----------------------------------------------

// Calculate matrix pseudo inverse
template <typename T>
MatrixX<T> pseudoInverse(const MatrixX<T>& mat);

// Calculate sparse matrix pseudo inverse - warning, this involves converting the sparse matrix to a
// dense matrix for now
template <typename T>
MatrixX<T> pseudoInverse(const SparseMatrix<T>& mat);

// -----------------------------------------------
//  geometry utilities
// -----------------------------------------------

/// Converts a quaternion to a rotation vector.
template <typename T>
Vector3<T> quaternionToRotVec(const Quaternion<T>& q);

/// Converts a rotation vector to a quaternion.
template <typename T>
Quaternion<T> rotVecToQuaternion(const Vector3<T>& v);

/// The Euler angles convention.
enum class EulerConvention {
  /// The intrinsic convention. Intrinsic rotations (also called local, relative, or body-fixed
  /// rotations) are performed around the axes of the coordinate system that is attached to the
  /// rotating object.
  ///
  /// Intrinsic rotations are usually represented by a sequence of three letters, like XYZ, which
  /// means rotation around the X-axis, followed by rotation around the Y-axis, and finally rotation
  /// around the Z-axis. For example, intrinsic XYZ is equivalent to RotX(a) * RotY(b) * RotZ(c).
  INTRINSIC,

  /// The extrinsic convention. Extrinsic rotations (also called global, absolute, or space-fixed
  /// rotations) are performed around the axes of a fixed coordinate system.
  ///
  /// Extrinsic rotations are often represented by uppercase letters, like XYZ, which means rotation
  /// around the fixed X-axis, followed by rotation around the fixed Y-axis, and finally rotation
  /// around the fixed Z-axis. For example, extrinsic XYZ is equivalent to RotZ(c) * RotY(b) *
  /// RotX(a).
  EXTRINSIC,
};

/// Converts rotation matrix to Euler angles
///
/// @tparam T The scalar type.
/// @param[in] m The rotation matrix to convert.
/// @param[in] axis0 The index of the first rotation's axis, one of {0, 1, 2}.
/// @param[in] axis1 The index of the second rotation's axis, one of {0, 1, 2}.
/// @param[in] axis2 The index of the third rotation's axis, one of {0, 1, 2}.
/// @param[in] convention The Euler angles convention.
template <typename T>
[[nodiscard]] Vector3<T> rotationMatrixToEuler(
    const Matrix3<T>& m,
    int axis0,
    int axis1,
    int axis2,
    EulerConvention convention = EulerConvention::INTRINSIC);

/// An optimized version of rotationMatrixToEuler(m, 0, 1, 2, convention) or
/// rotationMatrixToEuler(m, 2, 1, 0, convention).reverse().
template <typename T>
[[nodiscard]] Vector3<T> rotationMatrixToEulerXYZ(
    const Matrix3<T>& m,
    EulerConvention convention = EulerConvention::INTRINSIC);

/// An optimized version of rotationMatrixToEuler(m, 2, 1, 0, convention) or
/// rotationMatrixToEuler(m, 0, 1, 2, convention).reverse().
template <typename T>
[[nodiscard]] Vector3<T> rotationMatrixToEulerZYX(
    const Matrix3<T>& m,
    EulerConvention convention = EulerConvention::INTRINSIC);

/// Converts Euler angles to quaternion
///
/// @tparam T The scalar type.
/// @param[in] angles The 3-dimensional vector of Euler angles in order.
/// @param[in] axis0 The index of the first rotation's axis, one of {0, 1, 2}.
/// @param[in] axis1 The index of the second rotation's axis, one of {0, 1, 2}.
/// @param[in] axis2 The index of the third rotation's axis, one of {0, 1, 2}.
/// @param[in] convention The Euler angles convention.
template <typename T>
[[nodiscard]] Quaternion<T> eulerToQuaternion(
    const Vector3<T>& angles,
    int axis0,
    int axis1,
    int axis2,
    EulerConvention convention = EulerConvention::INTRINSIC);

/// Converts Euler angles to rotation matrix
///
/// @tparam T The scalar type.
/// @param[in] angles The 3-dimensional vector of Euler angles in order.
/// @param[in] axis0 The index of the first rotation's axis, one of {0, 1, 2}.
/// @param[in] axis1 The index of the second rotation's axis, one of {0, 1, 2}.
/// @param[in] axis2 The index of the third rotation's axis, one of {0, 1, 2}.
/// @param[in] convention The Euler angles convention.
template <typename T>
[[nodiscard]] Matrix3<T> eulerToRotationMatrix(
    const Vector3<T>& angles,
    int axis0,
    int axis1,
    int axis2,
    EulerConvention convention = EulerConvention::INTRINSIC);

/// An optimized version of eulerToRotationMatrix(angles, 0, 1, 2, convention) or
/// eulerToRotationMatrix(angles.reverse(), 2, 1, 0, convention).
template <typename T>
[[nodiscard]] Matrix3<T> eulerXYZToRotationMatrix(
    const Vector3<T>& angles,
    EulerConvention convention = EulerConvention::INTRINSIC);

/// An optimized version of eulerToRotationMatrix(angles, 2, 1, 0, convention) or
/// eulerToRotationMatrix(angles.reverse(), 0, 1, 2, convention).
template <typename T>
[[nodiscard]] Matrix3<T> eulerZYXToRotationMatrix(
    const Vector3<T>& angles,
    EulerConvention convention = EulerConvention::INTRINSIC);

// convert quaternion to euler
template <typename T>
Vector3<T> quaternionToEuler(const Quaternion<T>& q);

// weighted average of quaternions
Quaternionf quaternionAverage(
    gsl::span<const Quaternionf> q,
    gsl::span<const float> w = std::vector<float>());

/// Returns the closest points on two line segments where the line segments are represented in
/// origin and direction
///
/// @tparam T Scalar type
/// @param o1 Origin of the first line segment
/// @param d1 Direction of the first line segment
/// @param o2 Origin of the second line segment
/// @param d2 Direction of the second line segment
/// @param maxDist Maximum distance to check
/// @return [success, distance, [dist0, dist1]]:
/// - success: Whether the closest points were successfully found. On false, the rest return values
/// are invalid.
/// - distance: Distance between the closest points
/// - [dist0, dist1]: Distances of the closest points from their origins to the directions
template <typename T>
[[nodiscard]] std::tuple<bool, T, Eigen::Vector2<T>> closestPointsOnSegments(
    const Eigen::Vector3<T>& o1,
    const Eigen::Vector3<T>& d1,
    const Eigen::Vector3<T>& o2,
    const Eigen::Vector3<T>& d2,
    const T maxDist = std::numeric_limits<T>::max());

// The skew-symmetric matrix that corresponds to the
// cross product v x (something):
template <typename T>
Eigen::Matrix<T, 3, 3> crossProductMatrix(const Eigen::Matrix<T, 3, 1>& v) {
  Eigen::Matrix<T, 3, 3> result;
  result << T(0), -v.z(), v.y(), v.z(), T(0), -v.x(), -v.y(), v.x(), T(0);
  return result;
}

} // namespace momentum
