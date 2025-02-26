// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <pymomentum/tensor_ik/solver_options.h>

#include <momentum/character_solver/vertex_error_function.h>
#include <momentum/math/mppca.h>

#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h>

#include <optional>

namespace pymomentum {

// Note: we need to make this enum convertible to integers starting
// at 0 because we do ordering and mapping operations for those
// types.
enum class ErrorFunctionType {
  Position = 0,
  Orientation,
  Limit,
  Collision,
  PosePrior,
  Motion,
  Projection,
  Distance,
  Vertex,
  NumTypes
};

at::Tensor solveBodyIKProblem(
    pybind11::object characters,
    at::Tensor activeParameters,
    at::Tensor modelParameters_init,
    const std::vector<ErrorFunctionType>& activeErrorFunctions,
    at::Tensor errorFunctionWeights,
    SolverOptions options,
    std::optional<at::Tensor> positionCons_parents,
    std::optional<at::Tensor> positionCons_offsets,
    std::optional<at::Tensor> positionCons_weights,
    std::optional<at::Tensor> positionCons_targets,
    std::optional<at::Tensor> orientation_parents,
    std::optional<at::Tensor> orientation_offsets,
    std::optional<at::Tensor> orientation_weights,
    std::optional<at::Tensor> orientation_targets,
    pybind11::object posePrior_model,
    std::optional<at::Tensor> motion_targets,
    std::optional<at::Tensor> motion_weights,
    std::optional<at::Tensor> projectionCons_cameras,
    std::optional<at::Tensor> projectionCons_parents,
    std::optional<at::Tensor> projectionCons_offsets,
    std::optional<at::Tensor> projectionCons_weights,
    std::optional<at::Tensor> projectionCons_targets,
    std::optional<at::Tensor> distanceCons_cameras,
    std::optional<at::Tensor> distanceCons_parents,
    std::optional<at::Tensor> distanceCons_offsets,
    std::optional<at::Tensor> distanceCons_weights,
    std::optional<at::Tensor> distanceCons_targets,
    std::optional<at::Tensor> vertexCons_vertices,
    std::optional<at::Tensor> vertexCons_weights,
    std::optional<at::Tensor> vertexCons_target_positions,
    std::optional<at::Tensor> vertexCons_target_normals,
    momentum::VertexConstraintType vertexCons_type);

at::Tensor computeGradient(
    pybind11::object characters,
    at::Tensor modelParameters_init,
    const std::vector<ErrorFunctionType>& activeErrorFunctions,
    at::Tensor errorFunctionWeights,
    std::optional<at::Tensor> positionCons_parents,
    std::optional<at::Tensor> positionCons_offsets,
    std::optional<at::Tensor> positionCons_weights,
    std::optional<at::Tensor> positionCons_targets,
    std::optional<at::Tensor> orientation_parents,
    std::optional<at::Tensor> orientation_offsets,
    std::optional<at::Tensor> orientation_weights,
    std::optional<at::Tensor> orientation_targets,
    pybind11::object posePrior_model,
    std::optional<at::Tensor> motion_targets,
    std::optional<at::Tensor> motion_weights,
    std::optional<at::Tensor> projectionCons_cameras,
    std::optional<at::Tensor> projectionCons_parents,
    std::optional<at::Tensor> projectionCons_offsets,
    std::optional<at::Tensor> projectionCons_weights,
    std::optional<at::Tensor> projectionCons_targets,
    std::optional<at::Tensor> distanceCons_cameras,
    std::optional<at::Tensor> distanceCons_parents,
    std::optional<at::Tensor> distanceCons_offsets,
    std::optional<at::Tensor> distanceCons_weights,
    std::optional<at::Tensor> distanceCons_targets,
    std::optional<at::Tensor> vertexCons_vertices,
    std::optional<at::Tensor> vertexCons_weights,
    std::optional<at::Tensor> vertexCons_target_positions,
    std::optional<at::Tensor> vertexCons_target_normals,
    momentum::VertexConstraintType vertexCons_type);

at::Tensor computeResidual(
    pybind11::object characters,
    at::Tensor modelParameters_init,
    const std::vector<ErrorFunctionType>& activeErrorFunctions,
    at::Tensor errorFunctionWeights,
    std::optional<at::Tensor> positionCons_parents,
    std::optional<at::Tensor> positionCons_offsets,
    std::optional<at::Tensor> positionCons_weights,
    std::optional<at::Tensor> positionCons_targets,
    std::optional<at::Tensor> orientation_parents,
    std::optional<at::Tensor> orientation_offsets,
    std::optional<at::Tensor> orientation_weights,
    std::optional<at::Tensor> orientation_targets,
    pybind11::object posePrior_model,
    std::optional<at::Tensor> motion_targets,
    std::optional<at::Tensor> motion_weights,
    std::optional<at::Tensor> projectionCons_cameras,
    std::optional<at::Tensor> projectionCons_parents,
    std::optional<at::Tensor> projectionCons_offsets,
    std::optional<at::Tensor> projectionCons_weights,
    std::optional<at::Tensor> projectionCons_targets,
    std::optional<at::Tensor> distanceCons_cameras,
    std::optional<at::Tensor> distanceCons_parents,
    std::optional<at::Tensor> distanceCons_offsets,
    std::optional<at::Tensor> distanceCons_weights,
    std::optional<at::Tensor> distanceCons_targets,
    std::optional<at::Tensor> vertexCons_vertices,
    std::optional<at::Tensor> vertexCons_weights,
    std::optional<at::Tensor> vertexCons_target_positions,
    std::optional<at::Tensor> vertexCons_target_normals,
    momentum::VertexConstraintType vertexCons_type);

std::tuple<at::Tensor, at::Tensor> computeJacobian(
    pybind11::object characters,
    at::Tensor modelParameters_init,
    const std::vector<ErrorFunctionType>& activeErrorFunctions,
    at::Tensor errorFunctionWeights,
    std::optional<at::Tensor> positionCons_parents,
    std::optional<at::Tensor> positionCons_offsets,
    std::optional<at::Tensor> positionCons_weights,
    std::optional<at::Tensor> positionCons_targets,
    std::optional<at::Tensor> orientation_parents,
    std::optional<at::Tensor> orientation_offsets,
    std::optional<at::Tensor> orientation_weights,
    std::optional<at::Tensor> orientation_targets,
    pybind11::object posePrior_model,
    std::optional<at::Tensor> motion_targets,
    std::optional<at::Tensor> motion_weights,
    std::optional<at::Tensor> projectionCons_cameras,
    std::optional<at::Tensor> projectionCons_parents,
    std::optional<at::Tensor> projectionCons_offsets,
    std::optional<at::Tensor> projectionCons_weights,
    std::optional<at::Tensor> projectionCons_targets,
    std::optional<at::Tensor> distanceCons_cameras,
    std::optional<at::Tensor> distanceCons_parents,
    std::optional<at::Tensor> distanceCons_offsets,
    std::optional<at::Tensor> distanceCons_weights,
    std::optional<at::Tensor> distanceCons_targets,
    std::optional<at::Tensor> vertexCons_vertices,
    std::optional<at::Tensor> vertexCons_weights,
    std::optional<at::Tensor> vertexCons_target_positions,
    std::optional<at::Tensor> vertexCons_target_normals,
    momentum::VertexConstraintType vertexCons_type);

at::Tensor solveBodySequenceIKProblem(
    pybind11::object characters,
    at::Tensor activeParameters,
    at::Tensor sharedParameters,
    at::Tensor modelParameters_init,
    const std::vector<ErrorFunctionType>& activeErrorFunctions,
    at::Tensor errorFunctionWeights,
    SolverOptions options,
    std::optional<at::Tensor> positionCons_parents,
    std::optional<at::Tensor> positionCons_offsets,
    std::optional<at::Tensor> positionCons_weights,
    std::optional<at::Tensor> positionCons_targets,
    std::optional<at::Tensor> orientation_parents,
    std::optional<at::Tensor> orientation_offsets,
    std::optional<at::Tensor> orientation_weights,
    std::optional<at::Tensor> orientation_targets,
    pybind11::object posePrior_model,
    std::optional<at::Tensor> motion_targets,
    std::optional<at::Tensor> motion_weights,
    std::optional<at::Tensor> projectionCons_cameras,
    std::optional<at::Tensor> projectionCons_parents,
    std::optional<at::Tensor> projectionCons_offsets,
    std::optional<at::Tensor> projectionCons_weights,
    std::optional<at::Tensor> projectionCons_targets,
    std::optional<at::Tensor> distanceCons_cameras,
    std::optional<at::Tensor> distanceCons_parents,
    std::optional<at::Tensor> distanceCons_offsets,
    std::optional<at::Tensor> distanceCons_weights,
    std::optional<at::Tensor> distanceCons_targets,
    std::optional<at::Tensor> vertexCons_vertices,
    std::optional<at::Tensor> vertexCons_weights,
    std::optional<at::Tensor> vertexCons_target_positions,
    std::optional<at::Tensor> vertexCons_target_normals,
    momentum::VertexConstraintType vertexCons_type);

at::Tensor transformPose(
    const momentum::Character& character,
    at::Tensor modelParams,
    at::Tensor transforms,
    bool ensureContinuousOutput);

} // namespace pymomentum
