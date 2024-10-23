/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/tensor_momentum/tensor_kd_tree.h"

#include "pymomentum/tensor_utility/tensor_utility.h"

#include <axel/SimdKdTree.h>
#include <dispenso/parallel_for.h> // @manual
#include <pybind11/pybind11.h> // @manual=fbsource//third-party/pybind11/fbcode_py_versioned:pybind11

#include <cfloat>
#include <cstdint>

namespace py = pybind11;

namespace pymomentum {

namespace {

template <int Dimension>
void findClosestPoints_imp(
    at::Tensor points_source,
    at::Tensor points_target,
    float maxSqrDist,
    at::Tensor result_points,
    at::Tensor result_indices) {
  using Vec = typename axel::SimdKdTreef<Dimension>::Vec;
  assert(points_source.size(1) == Dimension);
  assert(points_target.size(1) == Dimension);
  const Vec* pts_tgt_ptr = (const Vec*)points_target.data_ptr();

  const int64_t nSrcPts = points_source.size(0);
  const int64_t nTgtPts = points_target.size(0);

  const axel::SimdKdTreef<Dimension> kdTree_target(
      {pts_tgt_ptr, pts_tgt_ptr + nTgtPts});

  Eigen::Map<Eigen::VectorXf> pts_src_map = toEigenMap<float>(points_source);
  Eigen::Map<Eigen::VectorXf> pts_tgt_map = toEigenMap<float>(points_target);
  Eigen::Map<Eigen::VectorXi> result_indices_map =
      toEigenMap<int>(result_indices);
  Eigen::Map<Eigen::VectorXf> result_points_map =
      toEigenMap<float>(result_points);

  dispenso::parallel_for(
      dispenso::makeChunkedRange(0, nSrcPts, 64),
      [&](int64_t srcStart, int64_t srcEnd) {
        for (int64_t k = srcStart; k < srcEnd; ++k) {
          const Vec p_src = pts_src_map.segment<Dimension>(Dimension * k);
          const auto [valid, tgt_index, sqrDist] =
              kdTree_target.closestPoint(p_src, maxSqrDist);
          if (valid) {
            result_indices_map(k) = tgt_index;
            result_points_map.segment<Dimension>(Dimension * k) =
                pts_tgt_map.segment<Dimension>(Dimension * tgt_index);
          } else {
            // No valid point found so return -1:
            result_indices_map(k) = -1;
            result_points_map.segment<Dimension>(Dimension * k).setZero();
          }
        }
      });
}

void findClosestPointsWithNormal_imp(
    at::Tensor points_source,
    at::Tensor normals_source,
    at::Tensor points_target,
    at::Tensor normals_target,
    float maxSqrDist,
    float maxNormalDot,
    at::Tensor result_points,
    at::Tensor result_normals,
    at::Tensor result_indices) {
  assert(points_source.size(1) == 3);
  assert(points_target.size(1) == 3);
  const Eigen::Vector3f* pts_tgt_ptr =
      (const Eigen::Vector3f*)points_target.data_ptr();
  const Eigen::Vector3f* normals_tgt_ptr =
      (const Eigen::Vector3f*)normals_target.data_ptr();

  const int64_t nSrcPts = points_source.size(0);
  const int64_t nTgtPts = points_target.size(0);

  const axel::SimdKdTreef<3> kdTree_target(
      {pts_tgt_ptr, pts_tgt_ptr + nTgtPts},
      {normals_tgt_ptr, normals_tgt_ptr + nTgtPts});

  Eigen::Map<Eigen::VectorXf> pts_src_map = toEigenMap<float>(points_source);
  Eigen::Map<Eigen::VectorXf> pts_tgt_map = toEigenMap<float>(points_target);
  Eigen::Map<Eigen::VectorXf> normals_src_map =
      toEigenMap<float>(normals_source);
  Eigen::Map<Eigen::VectorXf> normals_tgt_map =
      toEigenMap<float>(normals_target);

  Eigen::Map<Eigen::VectorXi> result_indices_map =
      toEigenMap<int>(result_indices);
  Eigen::Map<Eigen::VectorXf> result_points_map =
      toEigenMap<float>(result_points);
  Eigen::Map<Eigen::VectorXf> result_normals_map =
      toEigenMap<float>(result_normals);

  dispenso::parallel_for(
      dispenso::makeChunkedRange(0, nSrcPts, 64),
      [&](int64_t srcStart, int64_t srcEnd) {
        for (int64_t k = srcStart; k < srcEnd; ++k) {
          const Eigen::Vector3f p_src = pts_src_map.segment<3>(3 * k);
          const Eigen::Vector3f normal_src = normals_src_map.segment<3>(3 * k);
          const auto [valid, tgt_index, sqrDist] = kdTree_target.closestPoint(
              p_src, normal_src, maxSqrDist, maxNormalDot);
          if (valid) {
            result_indices_map(k) = tgt_index;
            result_points_map.segment<3>(3 * k) =
                pts_tgt_map.segment<3>(3 * tgt_index);
            result_normals_map.segment<3>(3 * k) =
                normals_tgt_map.segment<3>(3 * tgt_index);
          } else {
            // No valid point found so return -1:
            result_indices_map(k) = -1;
            result_points_map.segment<3>(3 * k).setZero();
            result_normals_map.segment<3>(3 * k).setZero();
          }
        }
      });
}

bool isNormalized(at::Tensor t) {
  if (isEmpty(t)) {
    return true;
  }

  if (t.size(-1) != 3) {
    return false;
  }

  t = t.reshape({-1, 3});

  Eigen::Map<Eigen::VectorXf> t_map = toEigenMap<float>(t);
  for (Eigen::Index i = 0; i < t.size(0) && i < 10; ++i) {
    const float norm = t_map.segment<3>(3 * i).norm();
    if (norm < 0.95 || norm > 1.05) {
      return false;
    }
  }

  return true;
}

} // anonymous namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> findClosestPoints(
    at::Tensor points_source,
    at::Tensor points_target,
    float maxDist) {
  TensorChecker checker("find_closest_points");

  bool squeeze_src = false;

  const float maxSqrDist = (maxDist == FLT_MAX) ? FLT_MAX : maxDist * maxDist;

  const int nSrcPtsIndex = -1;
  const int nTgtPtsIndex = -2;
  const int dimIdx = -3;
  points_source = checker.validateAndFixTensor(
      points_source,
      "points_source",
      {nSrcPtsIndex, dimIdx},
      {"nSrcPoints", "xyz"},
      at::kFloat,
      true,
      false,
      &squeeze_src);

  points_target = checker.validateAndFixTensor(
      points_target,
      "points_target",
      {nTgtPtsIndex, dimIdx},
      {"nTgtPts", "xyz"},
      at::kFloat,
      true,
      false,
      nullptr);

  const auto nBatch = checker.getBatchSize();
  const auto dim = checker.getBoundValue(dimIdx);
  const auto nSrcPts = checker.getBoundValue(nSrcPtsIndex);

  at::Tensor result_index =
      at::zeros({nBatch, nSrcPts}, at::CPU(toScalarType<int>()));
  at::Tensor result_points =
      at::zeros({nBatch, nSrcPts, dim}, at::CPU(toScalarType<float>()));

  dispenso::parallel_for((int64_t)0, nBatch, [&](int64_t iBatch) {
    if (dim == 2) {
      findClosestPoints_imp<2>(
          points_source.select(0, iBatch),
          points_target.select(0, iBatch),
          maxSqrDist,
          result_points.select(0, iBatch),
          result_index.select(0, iBatch));
    } else if (dim == 3) {
      findClosestPoints_imp<3>(
          points_source.select(0, iBatch),
          points_target.select(0, iBatch),
          maxSqrDist,
          result_points.select(0, iBatch),
          result_index.select(0, iBatch));
    }
  });

  if (squeeze_src) {
    result_points = result_points.squeeze(0);
    result_index = result_index.squeeze(0);
  }

  return {result_points, result_index, result_index >= 0};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
findClosestPointsWithNormals(
    at::Tensor points_source,
    at::Tensor normals_source,
    at::Tensor points_target,
    at::Tensor normals_target,
    float maxDist,
    float maxNormalDot) {
  TensorChecker checker("find_closest_points");

  bool squeeze_src = false;

  const float maxSqrDist = (maxDist == FLT_MAX) ? FLT_MAX : maxDist * maxDist;

  const int nSrcPtsIndex = -1;
  const int nTgtPtsIndex = -2;
  points_source = checker.validateAndFixTensor(
      points_source,
      "points_source",
      {nSrcPtsIndex, 3},
      {"nSrcPoints", "xyz"},
      at::kFloat,
      true,
      false,
      &squeeze_src);

  normals_source = checker.validateAndFixTensor(
      normals_source,
      "normals_source",
      {nSrcPtsIndex, 3},
      {"nSrcPoints", "xyz"},
      at::kFloat,
      true,
      false,
      nullptr);

  points_target = checker.validateAndFixTensor(
      points_target,
      "points_target",
      {nTgtPtsIndex, 3},
      {"nTgtPts", "xyz"},
      at::kFloat,
      true,
      false,
      nullptr);

  normals_target = checker.validateAndFixTensor(
      normals_target,
      "normals_target",
      {nTgtPtsIndex, 3},
      {"nTgtPts", "xyz"},
      at::kFloat,
      true,
      false,
      nullptr);

  if (!isNormalized(normals_source)) {
    py::print(
        "Inside find_closest_points, the tensor of source normals does not appear to be normalized.  This likely indicates a bug.");
  }

  if (!isNormalized(normals_target)) {
    py::print(
        "Inside find_closest_points, the tensor of target normals does not appear to be normalized.  This likely indicates a bug.");
  }

  const auto nBatch = checker.getBatchSize();
  const auto nSrcPts = checker.getBoundValue(nSrcPtsIndex);

  at::Tensor result_index =
      at::zeros({nBatch, nSrcPts}, at::CPU(toScalarType<int>()));
  at::Tensor result_points =
      at::zeros({nBatch, nSrcPts, 3}, at::CPU(toScalarType<float>()));
  at::Tensor result_normals =
      at::zeros({nBatch, nSrcPts, 3}, at::CPU(toScalarType<float>()));

  dispenso::parallel_for((int64_t)0, nBatch, [&](int64_t iBatch) {
    findClosestPointsWithNormal_imp(
        points_source.select(0, iBatch),
        normals_source.select(0, iBatch),
        points_target.select(0, iBatch),
        normals_target.select(0, iBatch),
        maxSqrDist,
        maxNormalDot,
        result_points.select(0, iBatch),
        result_normals.select(0, iBatch),
        result_index.select(0, iBatch));
  });

  if (squeeze_src) {
    result_points = result_points.squeeze(0);
    result_normals = result_normals.squeeze(0);
    result_index = result_index.squeeze(0);
  }

  return {result_points, result_normals, result_index, result_index >= 0};
}

} // namespace pymomentum
