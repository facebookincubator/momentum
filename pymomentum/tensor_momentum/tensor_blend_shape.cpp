/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/tensor_momentum/tensor_blend_shape.h"

#include "pymomentum/tensor_utility/tensor_utility.h"

#include <momentum/character/blend_shape.h>
#include <momentum/character/character.h>

#include <dispenso/parallel_for.h> // @manual
#include <torch/csrc/jit/python/python_ivalue.h>
#include <torch/library.h>
#include <torch/torch.h>
#include <Eigen/Core>

namespace pymomentum {

using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

namespace {

struct ApplyBlendShapeCoefficientsFunction
    : public torch::autograd::Function<ApplyBlendShapeCoefficientsFunction> {
 public:
  static variable_list forward(
      AutogradContext* ctx,
      PyObject* blendShape_in,
      at::Tensor blendShapeCoefficients);

  static variable_list backward(
      AutogradContext* ctx,
      variable_list grad_jointParameters);
};

variable_list ApplyBlendShapeCoefficientsFunction::forward(
    AutogradContext* ctx,
    PyObject* blendShape_in,
    at::Tensor blendShapeCoefficients) {
  const auto nCoeffs_idx = -1;

  ctx->saved_data["blendShape"] =
      c10::ivalue::ConcretePyObjectHolder::create(blendShape_in);
  ctx->save_for_backward({blendShapeCoefficients});

  TensorChecker checker("applyBlendShapeCoefficients");

  bool squeeze = false;
  blendShapeCoefficients = checker.validateAndFixTensor(
      blendShapeCoefficients,
      "blendShapeCoefficients",
      {nCoeffs_idx},
      {"nBlendShapeCoeffs"},
      at::kFloat,
      true,
      false,
      &squeeze);

  const int64_t nCoeffs = checker.getBoundValue(nCoeffs_idx);
  const int64_t nBatch = checker.getBatchSize();

  const momentum::BlendShape* blendShape =
      py::cast<const momentum::BlendShape*>(blendShape_in);

  MT_THROW_IF(
      nCoeffs > blendShape->shapeSize(),
      "In applyBlendShapeCoeffs, invalid blend shape count; expected at most {} coefficients but got {}.",
      blendShape->shapeSize(),
      nCoeffs);

  const int64_t nPoints = blendShape->modelSize();

  at::Tensor result = at::zeros({nBatch, nPoints, 3}, at::CPU(at::kFloat));

  const Eigen::MatrixXf& shapeVectors = blendShape->getShapeVectors();
  const std::vector<Eigen::Vector3f>& baseShape = blendShape->getBaseShape();

  dispenso::parallel_for(0, nBatch, [&](int64_t iBatch) {
    at::Tensor coeffs_cur = blendShapeCoefficients.select(0, iBatch);
    at::Tensor result_cur = result.select(0, iBatch);

    Eigen::Map<Eigen::VectorXf> result_cur_map = toEigenMap<float>(result_cur);
    for (size_t i = 0; i < baseShape.size(); ++i) {
      result_cur_map.segment<3>(3 * i) = baseShape[i];
    }

    result_cur_map +=
        shapeVectors.leftCols(nCoeffs) * toEigenMap<float>(coeffs_cur);
  });

  if (squeeze) {
    result = result.squeeze(0);
  }

  return {result};
}

variable_list ApplyBlendShapeCoefficientsFunction::backward(
    AutogradContext* ctx,
    variable_list grad_outputs) {
  MT_THROW_IF(
      grad_outputs.size() != 1,
      "Invalid grad_outputs in ApplyParameterTransformFunction::backward");

  const momentum::BlendShape* blendShape =
      py::cast<const momentum::BlendShape*>(
          ctx->saved_data["blendShape"].toPyObject());

  auto dLoss_dPositions =
      grad_outputs[0].contiguous().to(at::DeviceType::CPU, at::kFloat);

  bool squeeze = false;

  const int nPoints = blendShape->modelSize();

  const auto saved = ctx->get_saved_variables();
  MT_THROW_IF(saved.empty(), "Missing saved variable");

  // The only thing we actually need the blend shape vector for here is to
  // know how many blend shapes the user passed in:
  at::Tensor blendShapes = saved[0];
  const int64_t nBlendShapes = blendShapes.size(-1);

  TensorChecker checker("applyBlendShapeCoefficients");

  dLoss_dPositions = checker.validateAndFixTensor(
      dLoss_dPositions,
      "dLoss_dPositions",
      {nPoints, 3},
      {"nJointParameters", "xyz"},
      at::kFloat,
      true,
      false,
      &squeeze);
  const auto nBatch = checker.getBatchSize();

  at::Tensor dLoss_dBlendShapeCoeffs =
      at::zeros({nBatch, nBlendShapes}, at::CPU(at::kFloat));

  for (int64_t k = 0; k < nBatch; ++k) {
    at::Tensor dLoss_dCoeffsCur = dLoss_dBlendShapeCoeffs.select(0, k);
    at::Tensor dLoss_dPos_cur = dLoss_dPositions.select(0, k);

    toEigenMap<float>(dLoss_dCoeffsCur) =
        blendShape->getShapeVectors().leftCols(nBlendShapes).transpose() *
        toEigenMap<float>(dLoss_dPos_cur);
  }

  if (squeeze) {
    dLoss_dBlendShapeCoeffs = dLoss_dBlendShapeCoeffs.sum(0);
  }

  return {at::Tensor(), dLoss_dBlendShapeCoeffs};
}

} // anonymous namespace

// This is really just a big matrix multiplication, so it feels silly to
// do it explicitly rather than just defer to torch operations, but the
// problem is that to do the latter we'd end up copying the whole tensor
// every time we apply blend shapes.  Explicitly implementing this one
// operation seems like a reasonable compromise.
at::Tensor applyBlendShapeCoefficients(
    py::object blendShape,
    at::Tensor coeffs) {
  return ApplyBlendShapeCoefficientsFunction::apply(
      blendShape.ptr(), coeffs)[0];
}

} // namespace pymomentum
