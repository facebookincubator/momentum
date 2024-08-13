/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/parameter_transform.h>
#include <momentum/character/skeleton_state.h>

#include <ceres/jet.h>
#include <Eigen/Core>

namespace momentum {

template <typename T, typename T2>
Eigen::Matrix<T, 3, 1> getRotationDerivative(
    const JointStateT<T2>& js,
    const size_t index,
    const Eigen::Matrix<T, 3, 1>& ref) {
  return js.rotationAxis.col(index).cross(ref);
}

template <typename T, typename T2>
Eigen::Matrix<T, 3, 1> getScaleDerivative(
    const JointStateT<T2>& js,
    const Eigen::Matrix<T, 3, 1>& ref) {
  (void)js;
  return ref * ln2<T2>();
}

template <typename T, int N>
Eigen::Matrix<ceres::Jet<T, N>, N, 1> buildJetVec(const Eigen::Matrix<T, N, 1>& v) {
  Eigen::Matrix<ceres::Jet<T, N>, N, 1> result;
  for (int k = 0; k < N; ++k) {
    result(k).a = v(k);
    result(k).v.setZero();
    result(k).v(k) = 1;
  }

  return result;
}

template <typename T, typename T2>
T times_parameterTransform_times_v(
    const T& value,
    const size_t jointParamIdx,
    const ParameterTransform& parameterTransform,
    Eigen::Ref<const Eigen::VectorX<T2>> v) {
  T result = T();
  for (auto index = parameterTransform.transform.outerIndexPtr()[jointParamIdx];
       index < parameterTransform.transform.outerIndexPtr()[jointParamIdx + 1];
       ++index) {
    result += value * static_cast<T2>(parameterTransform.transform.valuePtr()[index]) *
        v[parameterTransform.transform.innerIndexPtr()[index]];
  }

  return result;
}

} // namespace momentum
