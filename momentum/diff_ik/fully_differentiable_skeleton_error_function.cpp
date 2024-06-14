/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/diff_ik/fully_differentiable_skeleton_error_function.h>

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
  if (value.size() != expectedSize) {
    std::ostringstream oss;
    oss << "In " << this->name() << "::getInput(" << name << "): expected size " << expectedSize
        << " but got " << value.size();
    throw std::runtime_error(oss.str());
  }

  getInputImp(name, value);
}

template <typename T>
void FullyDifferentiableSkeletonErrorFunctionT<T>::setInput(
    const std::string& name,
    Eigen::Ref<const Eigen::VectorX<T>> value) {
  const auto expectedSize = getInputSize(name);
  if (value.size() != expectedSize) {
    std::ostringstream oss;
    oss << "In " << this->name() << "::setInput(" << name << "): expected size " << expectedSize
        << " but got " << value.size();
    throw std::runtime_error(oss.str());
  }

  setInputImp(name, value);
}

template class FullyDifferentiableSkeletonErrorFunctionT<float>;
template class FullyDifferentiableSkeletonErrorFunctionT<double>;

} // namespace momentum
