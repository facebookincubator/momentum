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

#include <atomic>

namespace momentum {

inline constexpr size_t kAllFrames = SIZE_MAX;

template <typename T>
class SequenceSolverFunctionT : public SolverFunctionT<T> {
 public:
  SequenceSolverFunctionT(
      const Skeleton* skel,
      const ParameterTransformT<T>* parameterTransform,
      const ParameterSet& universal,
      const size_t nFrames);

  double getError(const Eigen::VectorX<T>& parameters) final;

  double getGradient(const Eigen::VectorX<T>& parameters, Eigen::VectorX<T>& gradient) final;

  double getJacobian(
      const Eigen::VectorX<T>& parameters,
      Eigen::MatrixX<T>& jacobian,
      Eigen::VectorX<T>& residual,
      size_t& actualRows) final;

  void updateParameters(Eigen::VectorX<T>& parameters, const Eigen::VectorX<T>& gradient) final;
  void setEnabledParameters(const ParameterSet&) final;

  ParameterSet getUniversalParameterSet() const {
    return universalParameters_;
  }

  // Passing in the special frame index kAllFrames will add the error function to every frame; this
  // is convenient for e.g. limit errors but requires that the error function be stateless. Note:
  // you are allowed to call this in a multithreaded context but you must ensure the frame indices
  // are different between the different threads.
  void addErrorFunction(
      const size_t frame,
      std::shared_ptr<SkeletonErrorFunctionT<T>> errorFunction);
  void addSequenceErrorFunction(
      const size_t startFrame,
      std::shared_ptr<SequenceErrorFunctionT<T>> errorFunction);

  size_t getNumFrames() const {
    return states_.size();
  }

  const ModelParametersT<T>& getFrameParameters(const size_t frame) const {
    return frameParameters_[frame];
  }

  void setFrameParameters(const size_t frame, const ModelParametersT<T>& parameters);
  ModelParametersT<T> getUniversalParameters() const;
  Eigen::VectorX<T> getJoinedParameterVector() const;
  void setJoinedParameterVector(const Eigen::VectorX<T>& joinedParameters);

 private:
  void setFrameParametersFromJoinedParameterVector(const Eigen::VectorX<T>& parameters);

 private:
  const Skeleton* skeleton_;
  const ParameterTransformT<T>* parameterTransform_;
  std::vector<SkeletonStateT<T>> states_;
  VectorX<bool> activeJointParams_;

  std::vector<ModelParametersT<T>> frameParameters_;

  void updateParameterSets(const ParameterSet& activeParams);

  // Indices of parameters that are active and solved per-frame:
  std::vector<Eigen::Index> perFrameParameterIndices_;

  // Indices of parameters that are active and solved universally:
  std::vector<Eigen::Index> universalParameterIndices_;

  const ParameterSet universalParameters_;

  std::vector<std::vector<std::shared_ptr<SkeletonErrorFunctionT<T>>>> perFrameErrorFunctions_;
  std::vector<std::vector<std::shared_ptr<SequenceErrorFunctionT<T>>>> sequenceErrorFunctions_;

  std::atomic<size_t> numTotalPerFrameErrorFunctions_ = 0;
  std::atomic<size_t> numTotalSequenceErrorFunctions_ = 0;

  friend class SequenceSolverT<T>;
};

} // namespace momentum
