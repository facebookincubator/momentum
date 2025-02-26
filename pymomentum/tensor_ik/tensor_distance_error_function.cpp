// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "pymomentum/tensor_ik/tensor_distance_error_function.h"

#include "pymomentum/nimble/tensor_camera/camera_utility.h"
#include "pymomentum/tensor_ik/tensor_error_function_utility.h"

#include <momentum/character/character.h>
#include <momentum/diff_ik/union_error_function.h>
#include <nimble/common/BodyTracking/ErrorFunctions/BodyJointDistanceConstraint.h> // @manual=fbsource//arvr/projects/nimble/common/BodyTracking:ErrorFunctions
#include <nimble/common/Utility/CameraData.h> // @manual=fbsource//arvr/projects/nimble/common/Utility:CameraModel

namespace pymomentum {

using nimble::BodyTracking::DistanceErrorFunction;
using nimble::BodyTracking::DistanceErrorFunctionT;

namespace {

const static int NCAM_IDX = -1;
const static int NCONS_IDX = -2;
const static int CAM_VEC_SIZE = -3;

template <typename T>
class TensorDistanceErrorFunction : public TensorErrorFunction<T> {
 public:
  TensorDistanceErrorFunction(
      size_t batchSize,
      size_t nFrames,
      at::Tensor cameras_cm,
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
    at::Tensor cameras_cm,
    at::Tensor parents,
    at::Tensor offsets,
    at::Tensor weights,
    at::Tensor targets)
    : TensorErrorFunction<T>(
          "Distance",
          "distance_cons",
          batchSize,
          nFrames,
          {{DistanceErrorFunction::kCameras,
            cameras_cm,
            {NCAM_IDX, CAM_VEC_SIZE},
            TensorType::TYPE_FLOAT,
            TensorInput::NON_DIFFERENTIABLE,
            TensorInput::REQUIRED},
           {DistanceErrorFunction::kParents,
            parents,
            {NCAM_IDX, NCONS_IDX},
            TensorType::TYPE_INT,
            TensorInput::NON_DIFFERENTIABLE,
            TensorInput::REQUIRED},
           {DistanceErrorFunction::kOffsets,
            offsets,
            {NCAM_IDX, NCONS_IDX, 3},
            TensorType::TYPE_FLOAT,
            TensorInput::DIFFERENTIABLE,
            TensorInput::OPTIONAL},
           {DistanceErrorFunction::kWeights,
            weights,
            {NCAM_IDX, NCONS_IDX},
            TensorType::TYPE_FLOAT,
            TensorInput::DIFFERENTIABLE,
            TensorInput::OPTIONAL},
           {DistanceErrorFunction::kTargets,
            targets,
            {NCAM_IDX, NCONS_IDX},
            TensorType::TYPE_FLOAT,
            TensorInput::DIFFERENTIABLE,
            TensorInput::REQUIRED}},
          {{NCAM_IDX, "nCameras"},
           {NCONS_IDX, "nConstraints"},
           {CAM_VEC_SIZE, "nCameraParams"}}) {}

template <typename T>
std::shared_ptr<momentum::SkeletonErrorFunctionT<T>>
TensorDistanceErrorFunction<T>::createErrorFunctionImp(
    const momentum::Character& character,
    size_t iBatch,
    size_t jFrame) const {
  std::vector<std::shared_ptr<momentum::SkeletonErrorFunctionT<T>>>
      errorFunctions;
  // Build some distance error functions, one for each camera,
  // and combine them to form a multiDistanceErrorFunction.
  const size_t numCameras = this->sharedSize(NCAM_IDX);
  const size_t cameraVecSize = this->sharedSize(CAM_VEC_SIZE);
  const auto cameraVectors =
      this->getTensorInput(DistanceErrorFunction::kCameras)
          .template toEigenMap<T>(iBatch, jFrame);
  for (size_t jCam = 0; jCam < numCameras; ++jCam) {
    nimble::Utility::CameraData3<T> camera_cm;
    if constexpr (std::is_same_v<T, float>) {
      camera_cm = cameraFromFlatVectorFloat(
          cameraVectors.segment(jCam * cameraVecSize, cameraVecSize));
    } else {
      camera_cm = cameraFromFlatVectorDouble(
          cameraVectors.segment(jCam * cameraVecSize, cameraVecSize));
    }
    auto errorFun =
        std::make_unique<nimble::BodyTracking::DistanceErrorFunctionT<T>>(
            character.skeleton, character.parameterTransform, camera_cm);

    // The Eigen map below are vectors for data of this sample in a batch.
    // One sample is made by NCAM cameras.
    // Each camera has NCONS constraints.
    // So each Eigen map actually stores 2D data.
    const auto weights = this->getTensorInput(DistanceErrorFunction::kWeights)
                             .template toEigenMap<T>(iBatch, jFrame);
    const auto offsets = this->getTensorInput(DistanceErrorFunction::kOffsets)
                             .template toEigenMap<T>(iBatch, jFrame);
    const auto parents = this->getTensorInput(DistanceErrorFunction::kParents)
                             .template toEigenMap<int>(iBatch, jFrame);
    const auto targets = this->getTensorInput(DistanceErrorFunction::kTargets)
                             .template toEigenMap<T>(iBatch, jFrame);

    const auto nCons = this->sharedSize(NCONS_IDX);
    for (Eigen::Index kCons = 0; kCons < nCons; ++kCons) {
      nimble::BodyTracking::DistanceConstraintDataT<T> constraintData;
      const size_t consIdx = jCam * nCons + kCons;
      constraintData.target = extractScalar<T>(targets, consIdx);
      constraintData.parent = extractScalar<int>(parents, consIdx);
      constraintData.offset =
          extractVector<T, 3>(offsets, consIdx, Eigen::Vector3<T>::Zero());
      constraintData.weight = extractScalar<T>(weights, consIdx, T(1));
      errorFun->addConstraint(constraintData);
    }
    errorFunctions.emplace_back(std::move(errorFun));
  }

  auto result = std::make_unique<momentum::UnionErrorFunctionT<T>>(
      character.skeleton, character.parameterTransform, errorFunctions);

  return result;
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
