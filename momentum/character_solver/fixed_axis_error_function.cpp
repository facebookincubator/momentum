/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/fixed_axis_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/common/profile.h"

namespace momentum {

template <typename T>
void FixedAxisDiffErrorFunctionT<T>::evalFunction(
    const size_t constrIndex,
    const JointStateT<T>& state,
    Vector3<T>& f,
    optional_ref<std::array<Vector3<T>, 1>> v,
    optional_ref<std::array<Matrix3<T>, 1>> dfdv) const {
  MT_PROFILE_FUNCTION();

  const FixedAxisDataT<T>& constr = this->constraints_[constrIndex];
  Vector3<T> vec = state.rotation() * constr.localAxis;
  f = vec - constr.globalAxis;
  if (v) {
    v->get().at(0) = std::move(vec);
  }
  if (dfdv) {
    dfdv->get().at(0).setIdentity();
  }
}

template <typename T>
void FixedAxisCosErrorFunctionT<T>::evalFunction(
    const size_t constrIndex,
    const JointStateT<T>& state,
    Vector<T, 1>& f,
    optional_ref<std::array<Vector3<T>, 1>> v,
    optional_ref<std::array<Eigen::Matrix<T, 1, 3>, 1>> dfdv) const {
  MT_PROFILE_FUNCTION();

  const FixedAxisDataT<T>& constr = this->constraints_[constrIndex];
  Vector3<T> vec = state.rotation() * constr.localAxis;
  f[0] = 1 - vec.dot(constr.globalAxis);
  if (v) {
    v->get().at(0) = std::move(vec);
  }
  if (dfdv) {
    dfdv->get().at(0) = -constr.globalAxis;
  }
}

template <typename T>
void FixedAxisAngleErrorFunctionT<T>::evalFunction(
    const size_t constrIndex,
    const JointStateT<T>& state,
    Vector<T, 1>& f,
    optional_ref<std::array<Vector3<T>, 1>> v,
    optional_ref<std::array<Eigen::Matrix<T, 1, 3>, 1>> dfdv) const {
  MT_PROFILE_FUNCTION();

  const FixedAxisDataT<T>& constr = this->constraints_[constrIndex];
  Vector3<T> vec = state.rotation() * constr.localAxis;
  const T dot = vec.dot(constr.globalAxis);
  f[0] = std::acos(std::clamp(dot, -T(1), T(1)));
  if (v) {
    v->get().at(0) = std::move(vec);
  }
  if (dfdv) {
    // The derivative of d[acos(x)]/dx  = -1/sqrt(1-x^2), where x is the cosine of the angle.
    // When the angle is 0 or 180, x=+/-1.0, and d[acos(x)] is infinity. But because dx=sine(angle)
    // is also zero, the final derivative will be zero as well.
    // Comparing to the Cos version, this Acos version scales up the jacobian by the inverse of
    // sine.
    const T sine = sqrt(1 - dot * dot);
    if (sine > 1e-9) {
      dfdv->get().at(0) = -constr.globalAxis / sine;
    } else {
      dfdv->get().at(0).setZero();
    }
  }
}

template class FixedAxisDiffErrorFunctionT<float>;
template class FixedAxisDiffErrorFunctionT<double>;

template class FixedAxisCosErrorFunctionT<float>;
template class FixedAxisCosErrorFunctionT<double>;

template class FixedAxisAngleErrorFunctionT<float>;
template class FixedAxisAngleErrorFunctionT<double>;

} // namespace momentum
