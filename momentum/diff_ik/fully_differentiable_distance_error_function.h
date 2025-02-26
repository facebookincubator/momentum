/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <momentum/character/locator.h>
#include <momentum/character_solver/distance_error_function.h>
#include <momentum/character_solver/skeleton_error_function.h>
#include <momentum/diff_ik/fully_differentiable_skeleton_error_function.h>
#include <momentum/diff_ik/fwd.h>

namespace momentum {

MOMENTUM_FWD_DECLARE_TEMPLATE_CLASS(DistanceErrorFunction);
MOMENTUM_FWD_DECLARE_TEMPLATE_STRUCT(DistanceConstraintData);

template <typename T>
class FullyDifferentiableDistanceErrorFunctionT
    : public DistanceErrorFunctionT<T>,
      public virtual momentum::FullyDifferentiableSkeletonErrorFunctionT<T> {
 public:
  static constexpr const char* kOrigins = "origins";
  static constexpr const char* kParents = "parents";
  static constexpr const char* kOffsets = "offsets";
  static constexpr const char* kWeights = "weights";
  static constexpr const char* kTargets = "targets";

  FullyDifferentiableDistanceErrorFunctionT(
      const momentum::Skeleton& skel,
      const momentum::ParameterTransform& pt);

  [[nodiscard]] std::vector<std::string> inputs() const override;
  [[nodiscard]] Eigen::Index getInputSize(const std::string& name) const override;
  void getInputImp(const std::string& name, Eigen::Ref<Eigen::VectorX<T>> result) const override;
  void setInputImp(const std::string& name, Eigen::Ref<const Eigen::VectorX<T>> value) override;
  [[nodiscard]] const char* name() const override {
    return "DistanceErrorFunction";
  }

  Eigen::VectorX<T> d_gradient_d_input_dot(
      const std::string& inputName,
      const momentum::ModelParametersT<T>& modelParams,
      const momentum::SkeletonStateT<T>& state,
      Eigen::Ref<const Eigen::VectorX<T>> inputVec) override;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  // Compute dError/dModelParams^T * v for some v.
  template <typename JetType>
  JetType constraintGradient_dot(
      const momentum::ModelParametersT<T>& modelParams,
      const momentum::SkeletonStateT<T>& skelState,
      const Eigen::Vector3<JetType>& origin_cons,
      int parentJointIndex,
      const Eigen::Vector3<JetType>& offset,
      const JetType& weight_cons,
      const JetType& target_cons,
      Eigen::Ref<const Eigen::VectorX<T>> vec);

  // weights for the error functions
  using DistanceErrorFunctionT<T>::kDistanceWeight;
  using DistanceErrorFunctionT<T>::constraints_;
};

} // namespace momentum
