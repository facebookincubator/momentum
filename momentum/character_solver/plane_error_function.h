/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character_solver/constraint_error_function.h>

namespace momentum {

template <typename T>
struct PlaneDataT : ConstraintData {
  Vector3<T> offset;
  /// plane equation: x.dot(normal) - d = 0; use minus to be compatible with previous
  /// implementation.
  Vector3<T> normal;
  T d;

  explicit PlaneDataT(
      const Vector3<T>& inOffset,
      const Vector3<T>& inNormal,
      const T inD,
      size_t pIndex,
      float w,
      const std::string& n = "")
      : ConstraintData(pIndex, w, n), offset(inOffset), normal(inNormal.normalized()), d(inD) {}
};

/// Create non-penetration half-plane constraints from locators with the input prefix
template <typename T>
std::vector<PlaneDataT<T>> createFloorConstraints(
    const std::string& prefix,
    const LocatorList& locators,
    const Vector3<T>& floorNormal,
    const T& floorOffset,
    float weight);

/// The PlaneErrorFunction computes the 3D positional errors from a list of Constraints.
/// Each constraint specifies a locator on the skeleton (parent joint and offset), and its target 3D
/// position (usually but not enforced) in the world space.
template <typename T>
class PlaneErrorFunctionT : public ConstraintErrorFunctionT<T, PlaneDataT<T>, 1> {
 public:
  /// Constructor
  ///
  /// @param[in] skel: character skeleton
  /// @param[in] pt: character parameter transformation
  /// @param[in] above: true means "above the plane" (half plane inequality), false means "on the
  /// plane" (equality).
  /// @param[in] lossAlpha: alpha parameter for the loss function
  /// @param[in] lossC: c parameter for the loss function
  explicit PlaneErrorFunctionT(
      const Skeleton& skel,
      const ParameterTransform& pt,
      const bool above = false,
      const T& lossAlpha = GeneralizedLossT<T>::kL2,
      const T& lossC = T(1))
      : ConstraintErrorFunctionT<T, PlaneDataT<T>, 1>(skel, pt, lossAlpha, lossC),
        halfPlane_(above) {}

  /// A convenience api where character contains info of the skeleton and parameter transform.
  ///
  /// @param[in] character: character definition
  /// @param[in] above: true means "above the plane" (half plane inequality), false means "on the
  /// plane" (equality).
  /// @param[in] lossAlpha: alpha parameter for the loss function
  /// @param[in] lossC: c parameter for the loss function
  explicit PlaneErrorFunctionT(
      const Character& character,
      const bool above = false,
      const T& lossAlpha = GeneralizedLossT<T>::kL2,
      const T& lossC = T(1))
      : PlaneErrorFunctionT(
            character.skeleton,
            character.parameterTransform,
            above,
            lossAlpha,
            lossC) {}

  /// Default constant weight in MarkerErrorFunction. This can be used for backwards compatibility
  /// in setWeight().
  static constexpr T kLegacyWeight = 1e-4f;

 protected:
  void evalFunction(
      size_t constrIndex,
      const JointStateT<T>& state,
      Vector<T, 1>& f,
      optional_ref<std::array<Vector3<T>, 1>> v = {},
      optional_ref<std::array<Eigen::Matrix<T, 1, 3>, 1>> /*dfdv*/ = {}) const final;

 private:
  /// True for an inequality constraint to be *above* the plane rather than *on* the plane
  bool halfPlane_ = false;
};

} // namespace momentum
