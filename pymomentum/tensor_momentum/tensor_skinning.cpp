/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/tensor_momentum/tensor_skinning.h"

#include "pymomentum/python_utility/python_utility.h"
#include "pymomentum/tensor_momentum/tensor_transforms.h"
#include "pymomentum/tensor_utility/autograd_utility.h"
#include "pymomentum/tensor_utility/tensor_utility.h"

#include <momentum/character/linear_skinning.h>
#include <momentum/character/skin_weights.h>
#include <momentum/math/mesh.h>

#include <dispenso/parallel_for.h> // @manual
#include <torch/csrc/jit/python/python_ivalue.h>

namespace pymomentum {

namespace py = pybind11;
namespace mm = momentum;

using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

namespace {

template <typename T>
struct SkinPointsFunction
    : public torch::autograd::Function<SkinPointsFunction<T>> {
 public:
  static variable_list forward(
      AutogradContext* ctx,
      PyObject* characters_in,
      at::Tensor transforms,
      at::Tensor restPoints);

  static variable_list backward(
      AutogradContext* ctx,
      variable_list grad_jointParameters);
};

template <typename T>
std::vector<Eigen::Matrix4<T>> multiply(
    const std::vector<Eigen::Matrix4<T>>& lhs,
    const std::vector<Eigen::Matrix4<T>>& rhs) {
  MT_THROW_IF(
      lhs.size() != rhs.size(), "Mismatch in list sizes in multiply().");
  std::vector<Eigen::Matrix4<T>> result;
  result.reserve(lhs.size());
  for (size_t i = 0; i < lhs.size(); ++i) {
    result.emplace_back(lhs[i] * rhs[i]);
  }
  return result;
}

template <typename T>
std::vector<Eigen::Matrix4<T>> affine3fToMatrixList(
    const momentum::TransformationList& xfs) {
  std::vector<Eigen::Matrix4<T>> result;
  result.reserve(xfs.size());
  for (const auto& xf : xfs) {
    result.emplace_back(xf.cast<T>().matrix());
  }
  return result;
}

template <typename T>
variable_list SkinPointsFunction<T>::forward(
    AutogradContext* ctx,
    PyObject* characters_in,
    at::Tensor transforms,
    at::Tensor restPoints) {
  const auto& firstCharacter = anyCharacter(characters_in, "skin_points()");

  MT_THROW_IF(
      !firstCharacter.mesh || !firstCharacter.skinWeights,
      "When skinning points, character is missing a mesh.");

  ctx->saved_data["character"] =
      c10::ivalue::ConcretePyObjectHolder::create(characters_in);
  ctx->save_for_backward({transforms, restPoints});

  const int nJoints = firstCharacter.skeleton.joints.size();
  const int nVertices = firstCharacter.mesh->vertices.size();

  TensorChecker checker("skin_points");
  bool squeeze;
  transforms = checker.validateAndFixTensor(
      transforms,
      "transforms",
      {nJoints, 4, 4},
      {"nJoints", "xf_rows", "xf_cols"},
      toScalarType<T>(),
      true,
      false,
      &squeeze);

  restPoints = checker.validateAndFixTensor(
      restPoints,
      "rest_points",
      {nVertices, 3},
      {"nVertices", "xyz"},
      toScalarType<T>(),
      true,
      true // allow empty
  );

  const auto nBatch = checker.getBatchSize();

  const auto characters =
      toCharacterList(characters_in, nBatch, "skin_points()");

  auto result = at::zeros({nBatch, nVertices, 3}, toScalarType<T>());
  dispenso::parallel_for(0, nBatch, [&](int64_t iBatch) {
    const momentum::Character* character = characters[iBatch];
    const auto& restMesh = *character->mesh;
    const auto& skin = *character->skinWeights;
    at::Tensor result_cur = result.select(0, iBatch);
    at::Tensor restPoints_cur = restPoints;
    if (!isEmpty(restPoints)) {
      restPoints_cur = restPoints.select(0, iBatch);
    }
    Eigen::Map<Eigen::VectorX<T>> restPoints_map =
        toEigenMap<T>(restPoints_cur);
    Eigen::Map<Eigen::VectorX<T>> results_map = toEigenMap<T>(result_cur);

    const std::vector<Eigen::Matrix4<T>> transforms_cur =
        toMatrixList<T, 4, 4>(transforms.select(0, iBatch));
    const std::vector<Eigen::Matrix4<T>> inverseBindPose =
        affine3fToMatrixList<T>(character->inverseBindPose);
    const std::vector<Eigen::Matrix4<T>> transforms_full =
        multiply<T>(transforms_cur, inverseBindPose);

    for (int64_t iVert = 0; iVert < nVertices; ++iVert) {
      // Use the default rest mesh if no rest points provided:
      const Eigen::Vector4<T> p_rest = [&]() -> Eigen::Vector4<T> {
        if (restPoints_map.size() == 0) {
          return restMesh.vertices[iVert].template cast<T>().homogeneous();
        } else {
          return restPoints_map.template segment<3>(3 * iVert).homogeneous();
        }
      }();
      Eigen::Vector4<T> p_world = Eigen::Vector4<T>::Zero();

      // loop over the weights
      for (size_t j = 0; j < momentum::kMaxSkinJoints; j++) {
        const float weight = skin.weight(iVert, j);
        if (weight == 0.0f) {
          break;
        }

        const int index = skin.index(iVert, j);

        // add up transforms
        p_world += weight * transforms_full[index] * p_rest;
      }

      results_map.template segment<3>(3 * iVert) = p_world.template head<3>();
    }
  });

  if (squeeze) {
    result = result.squeeze(0);
  }

  return {result};
}

template <typename T>
variable_list SkinPointsFunction<T>::backward(
    AutogradContext* ctx,
    variable_list grad_outputs) {
  MT_THROW_IF(
      grad_outputs.size() != 1,
      "Invalid grad_outputs in SkinPointsFunction::backward");

  const auto& firstCharacter =
      anyCharacter(ctx->saved_data["character"].toPyObject(), "skin_points()");
  const int nJoints = firstCharacter.skeleton.joints.size();
  const int nVertices = firstCharacter.mesh->vertices.size();

  // Restore variables:
  const auto saved = ctx->get_saved_variables();
  auto savedItr = std::begin(saved);
  auto transforms = *savedItr++;
  auto restPoints = *savedItr++;
  MT_THROW_IF(
      savedItr != std::end(saved), "Mismatch in saved variable counts.");

  TensorChecker checker("skin_points");

  bool squeeze_dLoss = false;
  auto dLoss_dWorldPositions = checker.validateAndFixTensor(
      grad_outputs[0],
      "dLoss_dWorldPositions",
      {(int)nVertices, 3},
      {"nVertices", "xyz"},
      toScalarType<T>(),
      true,
      false,
      &squeeze_dLoss);

  bool squeeze_transforms = false;
  transforms = checker.validateAndFixTensor(
      transforms,
      "transforms",
      {nJoints, 4, 4},
      {"nJoints", "xf_rows", "xf_cols"},
      toScalarType<T>(),
      true,
      false,
      &squeeze_transforms);

  bool squeeze_restPoints = false;
  restPoints = checker.validateAndFixTensor(
      restPoints,
      "rest_points",
      {nVertices, 3},
      {"nVertices", "xyz"},
      toScalarType<T>(),
      true,
      true, // allow empty
      &squeeze_restPoints);

  const auto nBatch = checker.getBatchSize();
  const auto characters = toCharacterList(
      ctx->saved_data["character"].toPyObject(), nBatch, "skin_points()");

  at::Tensor dLoss_dRestPositions;
  if (!isEmpty(restPoints)) {
    dLoss_dRestPositions = at::zeros({nBatch, nVertices, 3}, toScalarType<T>());
  }

  at::Tensor dLoss_dTransforms =
      at::zeros({nBatch, nJoints, 4, 4}, toScalarType<T>());

  dispenso::parallel_for(0, nBatch, [&](int64_t iBatch) {
    const momentum::Character* character = characters[iBatch];
    const auto& restMesh = *character->mesh;
    const auto& skin = *character->skinWeights;
    Eigen::Map<Eigen::VectorX<T>> dLoss_dWorldPositions_map =
        toEigenMap<T>(dLoss_dWorldPositions.select(0, iBatch));

    at::Tensor restPoints_cur = restPoints;
    if (!isEmpty(restPoints)) {
      restPoints_cur = restPoints.select(0, iBatch);
    }
    Eigen::Map<Eigen::VectorX<T>> restPoints_map =
        toEigenMap<T>(restPoints_cur);

    const std::vector<Eigen::Matrix4<T>> inverseBindPose =
        affine3fToMatrixList<T>(character->inverseBindPose);

    std::vector<Eigen::Matrix<T, 4, 4, Eigen::RowMajor>> dLoss_dTransform_accum(
        nJoints, Eigen::Matrix4<T>::Zero());

    // Accumulate derivatives wrt the transforms:
    for (int64_t iVert = 0; iVert < nVertices; ++iVert) {
      const Eigen::Vector3<T> dLoss_dWorldPos =
          dLoss_dWorldPositions_map.template segment<3>(3 * iVert);
      const Eigen::Vector4<T> p_rest = [&]() -> Eigen::Vector4<T> {
        if (restPoints_map.size() == 0) {
          return restMesh.vertices[iVert].template cast<T>().homogeneous();
        } else {
          return restPoints_map.template segment<3>(3 * iVert).homogeneous();
        }
      }();

      // loop over the weights
      for (size_t j = 0; j < momentum::kMaxSkinJoints; j++) {
        const float weight = skin.weight(iVert, j);
        if (weight == 0.0f) {
          break;
        }

        const int index = skin.index(iVert, j);
        const Eigen::Vector4<T> p_local =
            (inverseBindPose[index] * p_rest).template cast<T>();

        dLoss_dTransform_accum[index].template topLeftCorner<3, 3>() +=
            weight * dLoss_dWorldPos * p_local.template head<3>().transpose();
        dLoss_dTransform_accum[index].template block<3, 1>(0, 3) +=
            weight * dLoss_dWorldPos;
      }
    }

    Eigen::Map<Eigen::VectorX<T>> dLoss_dTransforms_map =
        toEigenMap<T>(dLoss_dTransforms.select(0, iBatch));
    for (int64_t iJoint = 0; iJoint < nJoints; ++iJoint) {
      std::copy(
          dLoss_dTransform_accum[iJoint].data(),
          dLoss_dTransform_accum[iJoint].data() + 4 * 4,
          dLoss_dTransforms_map.data() + 4 * 4 * iJoint);
    }

    // Derivatives wrt the rest points:
    if (!isEmpty(restPoints)) {
      const std::vector<Eigen::Matrix4<T>> transforms_cur =
          toMatrixList<T, 4, 4>(transforms.select(0, iBatch));
      const std::vector<Eigen::Matrix4<T>> transforms_full =
          multiply<T>(transforms_cur, inverseBindPose);

      Eigen::Map<Eigen::VectorX<T>> dLoss_dRestPositions_map =
          toEigenMap<T>(dLoss_dRestPositions.select(0, iBatch));
      for (int64_t iVert = 0; iVert < nVertices; ++iVert) {
        const Eigen::Vector3<T> dLoss_dWorldPos =
            dLoss_dWorldPositions_map.template segment<3>(3 * iVert);
        for (size_t j = 0; j < momentum::kMaxSkinJoints; j++) {
          const float weight = skin.weight(iVert, j);
          if (weight == 0.0f) {
            break;
          }

          const int index = skin.index(iVert, j);
          dLoss_dRestPositions_map.template segment<3>(3 * iVert) +=
              (weight *
               transforms_full[index]
                   .template topLeftCorner<3, 3>()
                   .transpose() *
               dLoss_dWorldPos);
        }
      }
    }
  });

  if (squeeze_transforms) {
    dLoss_dTransforms = dLoss_dTransforms.sum(0);
  }

  if (squeeze_restPoints) {
    dLoss_dRestPositions = dLoss_dRestPositions.sum(0);
  }

  return {at::Tensor(), dLoss_dTransforms, dLoss_dRestPositions};
}
} // namespace

at::Tensor skinPoints(
    pybind11::object characters,
    at::Tensor skel_state,
    std::optional<at::Tensor> restPoints) {
  if (skel_state.size(-1) == 8) {
    // Assumed to be a skeleton state.
    skel_state = skeletonStateToTransforms(skel_state);
  } else if (
      skel_state.ndimension() > 2 && skel_state.size(-1) == 4 &&
      skel_state.size(-2) == 4) {
    // Assumed to be a matrix.
    ;
  } else {
    MT_THROW(
        "In skin_points, skel_state tensor expected to be either a skel_state tensor ([nBatch x nJoints x 8]) or a tensor of 4x4 matrices ([nBatch x nJoints x 4 x 4]).  Got {}",
        formatTensorSizes(skel_state));
  }

  return applyTemplatedAutogradFunction<SkinPointsFunction>(
      characters.ptr(), skel_state, denullify(restPoints))[0];
}

at::Tensor computeVertexNormals(
    at::Tensor vertex_positions,
    at::Tensor triangles) {
  // vertex_positions shape: [..., n_vertices, 3]
  // triangles shape: [n_triangles, 3]
  MT_THROW_IF(
      vertex_positions.ndimension() < 2,
      "In compute_vertex_normals, expected vertex_positions to have at least two dimensions, got {}",
      formatTensorSizes(vertex_positions));
  MT_THROW_IF(
      vertex_positions.size(-1) != 3,
      "In compute_vertex_normals, expected vertex_positions to have last dimension equal to 3, got {}",
      formatTensorSizes(vertex_positions));
  MT_THROW_IF(
      triangles.ndimension() != 2 || triangles.size(-1) != 3,
      "In compute_vertex_normals, expected triangles to have shape [n_triangles, 3], got {}",
      formatTensorSizes(triangles));

  // x1, x2, x3: batch_size x n_triangles x 3
  at::Tensor x1 =
      at::index_select(vertex_positions, -2, triangles.select(-1, 0));
  at::Tensor x2 =
      at::index_select(vertex_positions, -2, triangles.select(-1, 1));
  at::Tensor x3 =
      at::index_select(vertex_positions, -2, triangles.select(-1, 2));

  at::Tensor triangle_normals = at::cross(x2 - x1, x3 - x1, -1);
  at::Tensor vertex_normals = at::zeros_like(vertex_positions);
  for (int64_t i = 0; i < 3; ++i) {
    vertex_normals.index_add_(-2, triangles.select(-1, i), triangle_normals);
  }

  return torch::nn::functional::normalize(
      vertex_normals,
      torch::nn::functional::NormalizeFuncOptions().p(2).dim(-1));
}

} // namespace pymomentum
