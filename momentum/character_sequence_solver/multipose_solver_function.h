/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/character/parameter_transform.h>
#include <momentum/character/types.h>
#include <momentum/character_sequence_solver/fwd.h>
#include <momentum/character_solver/fwd.h>
#include <momentum/solver/solver_function.h>

namespace momentum {

template <typename T>
class MultiposeSolverFunctionT : public SolverFunctionT<T> {
 public:
  MultiposeSolverFunctionT(
      const Skeleton* skel,
      const ParameterTransformT<T>* parameterTransform,
      gsl::span<const int> universal,
      size_t frames);

  double getError(const Eigen::VectorX<T>& parameters) final;

  double getGradient(const Eigen::VectorX<T>& parameters, Eigen::VectorX<T>& gradient) final;

  double getJacobian(
      const Eigen::VectorX<T>& parameters,
      Eigen::MatrixX<T>& jacobian,
      Eigen::VectorX<T>& residual,
      size_t& actualRows) final;

  void updateParameters(Eigen::VectorX<T>& parameters, const Eigen::VectorX<T>& gradient) final;
  void setEnabledParameters(const ParameterSet&) final;

  void addErrorFunction(size_t frame, SkeletonErrorFunctionT<T>* errorFunction);

  size_t getNumFrames() const {
    return states_.size();
  }

  const ModelParametersT<T>& getFrameParameters(const size_t frame) const {
    return frameParameters_[frame];
  }

  void setFrameParameters(size_t frame, const ModelParametersT<T>& parameters);
  Eigen::VectorX<T> getUniversalParameters();
  Eigen::VectorX<T> getJoinedParameterVector() const;
  void setJoinedParameterVector(const Eigen::VectorX<T>& joinedParameters);

  ParameterSet getUniversalParameterSet() const;
  ParameterSet getUniversalLocatorParameterSet() const;

 private:
  void setFrameParametersFromJoinedParameterVector(const Eigen::VectorX<T>& parameters);

 private:
  const Skeleton* skeleton_;
  const ParameterTransformT<T>* parameterTransform_;
  std::vector<std::unique_ptr<SkeletonStateT<T>>> states_;
  VectorX<bool> activeJointParams_;

  std::vector<ModelParametersT<T>> frameParameters_;
  Eigen::VectorX<T> universal_;
  std::vector<size_t> parameterIndexMap_;

  std::vector<size_t> genericParameters_;
  std::vector<size_t> universalParameters_;

  std::vector<std::vector<SkeletonErrorFunctionT<T>*>> errorFunctions_;

  friend class MultiposeSolverT<T>;
};

} // namespace momentum
