// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "pymomentum/tensor_ik/tensor_projection_error_function.h"

#include "pymomentum/nimble/tensor_camera/camera_utility.h"
#include "pymomentum/tensor_ik/tensor_error_function_utility.h"

#include <momentum/character/character.h>
#include <momentum/diff_ik/union_error_function.h>
#include <nimble/common/BodyTracking/ErrorFunctions/LinearizedProjectionErrorFunction.h> // @manual=fbsource//arvr/projects/nimble/common/BodyTracking:ErrorFunctions
#include <nimble/common/Utility/CameraData.h> // @manual=fbsource//arvr/projects/nimble/common/Utility:CameraModel

namespace pymomentum {

using nimble::BodyTracking::LinearizedProjectionErrorFunction;

namespace {

const static int NCAM_IDX = -1;
const static int NCONS_IDX = -2;
const static int CAM_VEC_SIZE = -3;

template <typename T>
class TensorProjectionErrorFunction : public TensorErrorFunction<T> {
 public:
  TensorProjectionErrorFunction(
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
      size_t jFrame) const override;
};

template <typename T>
TensorProjectionErrorFunction<T>::TensorProjectionErrorFunction(
    size_t batchSize,
    size_t nFrames,
    at::Tensor cameras_cm,
    at::Tensor parents,
    at::Tensor offsets,
    at::Tensor weights,
    at::Tensor targets)
    : TensorErrorFunction<T>(
          "Projection",
          "projection_cons",
          batchSize,
          nFrames,
          {{LinearizedProjectionErrorFunction::kCameras,
            cameras_cm,
            {NCAM_IDX, CAM_VEC_SIZE},
            TensorType::TYPE_FLOAT,
            TensorInput::NON_DIFFERENTIABLE,
            TensorInput::REQUIRED},
           {LinearizedProjectionErrorFunction::kParents,
            parents,
            {NCAM_IDX, NCONS_IDX},
            TensorType::TYPE_INT,
            TensorInput::NON_DIFFERENTIABLE,
            TensorInput::REQUIRED},
           {LinearizedProjectionErrorFunction::kOffsets,
            offsets,
            {NCAM_IDX, NCONS_IDX, 3},
            TensorType::TYPE_FLOAT,
            TensorInput::DIFFERENTIABLE,
            TensorInput::OPTIONAL},
           {LinearizedProjectionErrorFunction::kWeights,
            weights,
            {NCAM_IDX, NCONS_IDX},
            TensorType::TYPE_FLOAT,
            TensorInput::DIFFERENTIABLE,
            TensorInput::OPTIONAL},
           {LinearizedProjectionErrorFunction::kTargets,
            targets,
            {NCAM_IDX, NCONS_IDX, 2},
            TensorType::TYPE_FLOAT,
            TensorInput::DIFFERENTIABLE,
            TensorInput::REQUIRED}},
          {{NCAM_IDX, "nCameras"},
           {NCONS_IDX, "nConstraints"},
           {CAM_VEC_SIZE, "nCameraParams"}}) {}

template <typename T>
std::shared_ptr<momentum::SkeletonErrorFunctionT<T>>
TensorProjectionErrorFunction<T>::createErrorFunctionImp(
    const momentum::Character& character,
    size_t iBatch,
    size_t jFrame) const {
  std::vector<std::shared_ptr<momentum::SkeletonErrorFunctionT<T>>>
      errorFunctions;
  // Build some projection error functions, one for each camera,
  // and combine them to form a multiProjectionErrorFunction.
  const size_t numCameras = this->sharedSize(NCAM_IDX);
  const size_t cameraVecSize = this->sharedSize(CAM_VEC_SIZE);
  const auto cameraVectors =
      this->getTensorInput(LinearizedProjectionErrorFunction::kCameras)
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
    auto errorFun = std::make_unique<
        nimble::BodyTracking::LinearizedProjectionErrorFunctionT<T>>(
        character.skeleton, character.parameterTransform, camera_cm);

    // The Eigen map below are vectors for data of this sample in a batch.
    // One sample is made by NCAM cameras.
    // Each camera has NCONS constraints.
    // So each Eigen map actually stores 2D data.
    const auto weights =
        this->getTensorInput(LinearizedProjectionErrorFunction::kWeights)
            .template toEigenMap<T>(iBatch, jFrame);
    const auto offsets =
        this->getTensorInput(LinearizedProjectionErrorFunction::kOffsets)
            .template toEigenMap<T>(iBatch, jFrame);
    const auto parents =
        this->getTensorInput(LinearizedProjectionErrorFunction::kParents)
            .template toEigenMap<int>(iBatch, jFrame);
    const auto targets =
        this->getTensorInput(LinearizedProjectionErrorFunction::kTargets)
            .template toEigenMap<T>(iBatch, jFrame);

    const auto nCons = this->sharedSize(NCONS_IDX);
    for (Eigen::Index kCons = 0; kCons < nCons; ++kCons) {
      nimble::BodyTracking::ProjectionConstraintDataT<T> constraintData;
      const size_t consIdx = jCam * nCons + kCons;
      constraintData.target = extractVector<T, 2>(targets, consIdx);
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
std::unique_ptr<TensorErrorFunction<T>> createProjectionErrorFunction(
    size_t batchSize,
    size_t nFrames,
    at::Tensor cameras,
    at::Tensor parents,
    at::Tensor offsets,
    at::Tensor weights,
    at::Tensor targets) {
  return std::make_unique<TensorProjectionErrorFunction<T>>(
      batchSize, nFrames, cameras, parents, offsets, weights, targets);
}

template std::unique_ptr<TensorErrorFunction<float>>
createProjectionErrorFunction<float>(
    size_t batchSize,
    size_t nFrames,
    at::Tensor cameras,
    at::Tensor parents,
    at::Tensor offsets,
    at::Tensor weights,
    at::Tensor targets);

template std::unique_ptr<TensorErrorFunction<double>>
createProjectionErrorFunction<double>(
    size_t batchSize,
    size_t nFrames,
    at::Tensor cameras,
    at::Tensor parents,
    at::Tensor offsets,
    at::Tensor weights,
    at::Tensor targets);

} // namespace pymomentum
