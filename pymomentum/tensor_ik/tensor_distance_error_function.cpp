// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "pymomentum/tensor_ik/tensor_distance_error_function.h"

#include "pymomentum/tensor_ik/tensor_error_function_utility.h"

#include <momentum/character/character.h>
#include <momentum/diff_ik/fully_differentiable_distance_error_function.h>
#include <momentum/diff_ik/fwd.h>

namespace pymomentum {

using momentum::DistanceErrorFunctionT;
using momentum::FullyDifferentiableDistanceErrorFunction;
using momentum::FullyDifferentiableDistanceErrorFunctionT;

namespace {

const static int NCONS_IDX = -1;

template <typename T>
class TensorDistanceErrorFunction : public TensorErrorFunction<T> {
 public:
  TensorDistanceErrorFunction(
      size_t batchSize,
      size_t nFrames,
      at::Tensor origins,
      at::Tensor parents,
      at::Tensor offsets,
      at::Tensor weights,
      at::Tensor targets);

 protected:
  std::shared_ptr<momentum::SkeletonErrorFunctionT<T>> createErrorFunctionImp(
      const momentum::Character& character,
      size_t iBatch,
      size_t kFrame) const override;
};

template <typename T>
TensorDistanceErrorFunction<T>::TensorDistanceErrorFunction(
    size_t batchSize,
    size_t nFrames,
    at::Tensor origins,
    at::Tensor parents,
    at::Tensor offsets,
    at::Tensor weights,
    at::Tensor targets)
    : TensorErrorFunction<T>(
          "Distance",
          "distance_cons",
          batchSize,
          nFrames,
          {{FullyDifferentiableDistanceErrorFunction::kOrigins,
            origins,
            {NCONS_IDX, 3},
            TensorType::TYPE_FLOAT,
            TensorInput::DIFFERENTIABLE,
            TensorInput::REQUIRED},
           {FullyDifferentiableDistanceErrorFunction::kParents,
            parents,
            {NCONS_IDX},
            TensorType::TYPE_INT,
            TensorInput::NON_DIFFERENTIABLE,
            TensorInput::REQUIRED},
           {FullyDifferentiableDistanceErrorFunction::kOffsets,
            offsets,
            {NCONS_IDX, 3},
            TensorType::TYPE_FLOAT,
            TensorInput::DIFFERENTIABLE,
            TensorInput::OPTIONAL},
           {FullyDifferentiableDistanceErrorFunction::kWeights,
            weights,
            {NCONS_IDX},
            TensorType::TYPE_FLOAT,
            TensorInput::DIFFERENTIABLE,
            TensorInput::OPTIONAL},
           {FullyDifferentiableDistanceErrorFunction::kTargets,
            targets,
            {NCONS_IDX},
            TensorType::TYPE_FLOAT,
            TensorInput::DIFFERENTIABLE,
            TensorInput::REQUIRED}},
          {
              {NCONS_IDX, "nConstraints"},
          }) {}

template <typename T>
std::shared_ptr<momentum::SkeletonErrorFunctionT<T>>
TensorDistanceErrorFunction<T>::createErrorFunctionImp(
    const momentum::Character& character,
    size_t iBatch,
    size_t jFrame) const {
  auto errorFun =
      std::make_unique<momentum::FullyDifferentiableDistanceErrorFunctionT<T>>(
          character.skeleton, character.parameterTransform);

  const auto weights =
      this->getTensorInput(FullyDifferentiableDistanceErrorFunction::kWeights)
          .template toEigenMap<T>(iBatch, jFrame);
  const auto origins =
      this->getTensorInput(FullyDifferentiableDistanceErrorFunction::kOrigins)
          .template toEigenMap<T>(iBatch, jFrame);
  const auto offsets =
      this->getTensorInput(FullyDifferentiableDistanceErrorFunction::kOffsets)
          .template toEigenMap<T>(iBatch, jFrame);
  const auto parents =
      this->getTensorInput(FullyDifferentiableDistanceErrorFunction::kParents)
          .template toEigenMap<int>(iBatch, jFrame);
  const auto targets =
      this->getTensorInput(FullyDifferentiableDistanceErrorFunction::kTargets)
          .template toEigenMap<T>(iBatch, jFrame);

  const auto nCons = this->sharedSize(NCONS_IDX);
  for (Eigen::Index kCons = 0; kCons < nCons; ++kCons) {
    momentum::DistanceConstraintDataT<T> constraintData;
    constraintData.origin =
        extractVector<T, 3>(origins, kCons, Eigen::Vector3<T>::Zero());
    constraintData.target = extractScalar<T>(targets, kCons);
    constraintData.parent = extractScalar<int>(parents, kCons);
    constraintData.offset =
        extractVector<T, 3>(offsets, kCons, Eigen::Vector3<T>::Zero());
    constraintData.weight = extractScalar<T>(weights, kCons, T(1));
    errorFun->addConstraint(constraintData);
  }

  return errorFun;
}

} // End of anonymous namespace

template <typename T>
std::unique_ptr<TensorErrorFunction<T>> createDistanceErrorFunction(
    size_t batchSize,
    size_t nFrames,
    at::Tensor cameras,
    at::Tensor parents,
    at::Tensor offsets,
    at::Tensor weights,
    at::Tensor targets) {
  return std::make_unique<TensorDistanceErrorFunction<T>>(
      batchSize, nFrames, cameras, parents, offsets, weights, targets);
}

template std::unique_ptr<TensorErrorFunction<float>>
createDistanceErrorFunction<float>(
    size_t batchSize,
    size_t nFrames,
    at::Tensor cameras,
    at::Tensor parents,
    at::Tensor offsets,
    at::Tensor weights,
    at::Tensor targets);

template std::unique_ptr<TensorErrorFunction<double>>
createDistanceErrorFunction<double>(
    size_t batchSize,
    size_t nFrames,
    at::Tensor cameras,
    at::Tensor parents,
    at::Tensor offsets,
    at::Tensor weights,
    at::Tensor targets);

} // namespace pymomentum
