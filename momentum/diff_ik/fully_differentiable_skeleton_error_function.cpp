/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/diff_ik/fully_differentiable_skeleton_error_function.h>

#include "momentum/common/exception.h"

namespace momentum {

template <typename T>
Eigen::VectorX<T> FullyDifferentiableSkeletonErrorFunctionT<T>::getInput(
    const std::string& name) const {
  Eigen::VectorX<T> result = Eigen::VectorX<T>::Zero(getInputSize(name));
  getInput(name, result);
  return result;
}

template <typename T>
void FullyDifferentiableSkeletonErrorFunctionT<T>::getInput(
    const std::string& name,
    Eigen::Ref<Eigen::VectorX<T>> value) const {
  const auto expectedSize = getInputSize(name);
  MT_THROW_IF(
      value.size() != expectedSize,
      "In {}::getInput({}): expected size {} but got {}",
      this->name(),
      name,
      expectedSize,
      value.size());

  getInputImp(name, value);
}

template <typename T>
void FullyDifferentiableSkeletonErrorFunctionT<T>::setInput(
    const std::string& name,
    Eigen::Ref<const Eigen::VectorX<T>> value) {
  const auto expectedSize = getInputSize(name);
  MT_THROW_IF(
      value.size() != expectedSize,
      "In {}::setInput({}): expected size {} but got {}",
      this->name(),
      name,
      expectedSize,
      value.size());

  setInputImp(name, value);
}

template class FullyDifferentiableSkeletonErrorFunctionT<float>;
template class FullyDifferentiableSkeletonErrorFunctionT<double>;

} // namespace momentum
