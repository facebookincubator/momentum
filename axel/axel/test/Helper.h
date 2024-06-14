/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <type_traits>
#include <unordered_set>
#include <vector>

#include <Eigen/Core>

#include "axel/BoundingBox.h"
#include "axel/Checks.h"

namespace axel::test {

struct pair_hash {
  template <class T1, class T2>
  [[nodiscard]] std::size_t operator()(const std::pair<T1, T2>& p) const {
    const auto h1 = std::hash<T1>{}(p.first);
    const auto h2 = std::hash<T2>{}(p.second);
    return h1 ^ h2;
  }
};

using IndexPairSet = std::unordered_set<std::pair<size_t, size_t>, pair_hash>;

template <typename S>
struct BoxAndCollisionResults {
  std::vector<BoundingBox<S>> boxes;
  IndexPairSet collisionPairs;
};

template <typename S>
struct BoxAndInterTreeCollisionResults {
  std::vector<BoundingBox<S>> boxesA;
  std::vector<BoundingBox<S>> boxesB;
  IndexPairSet collisionPairs;
};

/// Generates a specified number of bounding boxes and a set of colliding pair indices.
///
/// @param numBoxes The number of bounding boxes to generate.
/// @param extent The extent of the space in each dimension.
/// @param sparsityFactor A factor controlling the spacing between bounding boxes; higher values
/// yield sparser distributions.
/// @return A pair containing a vector of generated bounding boxes and a vector of colliding pair
/// indices.
template <typename S>
[[nodiscard]] BoxAndCollisionResults<S>
generateBoxesAndCollisions(size_t numBoxes, S extent = 10, S sparsityFactor = 1) {
  BoxAndCollisionResults<S> result;

  for (size_t i = 0; i < numBoxes; ++i) {
    const Eigen::Vector3<S> randomA =
        Eigen::Vector3<S>::Random() * extent + Eigen::Vector3<S>::Constant(i * sparsityFactor);
    const Eigen::Vector3<S> randomB =
        Eigen::Vector3<S>::Random() * extent + Eigen::Vector3<S>::Constant(i * sparsityFactor);
    result.boxes.push_back(BoundingBox<S>(randomA.cwiseMin(randomB), randomA.cwiseMax(randomB), i));
  }

  for (size_t i = 0; i < result.boxes.size(); ++i) {
    for (size_t j = i + 1; j < result.boxes.size(); ++j) {
      if (result.boxes[i].intersects(result.boxes[j])) {
        result.collisionPairs.emplace(i, j);
      }
    }
  }

  return result;
}

/// Generates two sets of bounding boxes and computes the collisions between them.
///
/// @tparam S The scalar type used for the dimensions of the bounding boxes.
/// @param numBoxesA The number of bounding boxes to generate for the first set.
/// @param numBoxesB The number of bounding boxes to generate for the second set.
/// @param extent The extent of the space in each dimension for the bounding boxes.
/// @param sparsityFactor A factor controlling the spacing between bounding boxes; higher values
///                       yield sparser distributions.
/// @return A tuple containing two vectors of generated bounding boxes (one for each set) and
///         a set of colliding pair indices representing the intersections between the two sets.
template <typename S>
[[nodiscard]] BoxAndInterTreeCollisionResults<S> generateBoxesAndInterTreeCollisions(
    size_t numBoxesA,
    size_t numBoxesB,
    S extent = 10,
    S sparsityFactor = 1) {
  BoxAndInterTreeCollisionResults<S> result;

  const auto [boxesA, _] = generateBoxesAndCollisions<S>(numBoxesA, extent, sparsityFactor);
  sparsityFactor += 1; // Adjust sparsity factor for second set to ensure different box positions
  const auto [boxesB, __] = generateBoxesAndCollisions<S>(numBoxesB, extent, sparsityFactor);
  result.boxesA = std::move(boxesA);
  result.boxesB = std::move(boxesB);

  // Check for intersections between boxes from the two sets
  for (size_t i = 0; i < result.boxesA.size(); ++i) {
    for (size_t j = 0; j < result.boxesB.size(); ++j) {
      if (result.boxesA[i].intersects(result.boxesB[j])) {
        result.collisionPairs.emplace(i, j);
      }
    }
  }

  return result;
}

template <typename S>
struct MeshData {
  Eigen::MatrixX3<S> positions;
  Eigen::MatrixX3i triangles;
};

template <typename S>
void rotate(MeshData<S>& meshData, const Eigen::Matrix3<S>& R) {
  for (Eigen::Index i = 0; i < meshData.positions.rows(); ++i) {
    meshData.positions.row(i) = R * meshData.positions.row(i).transpose();
  }
}

template <typename S>
struct SphereMeshParameters {
  S radius;
  int32_t meridians;
  int32_t parallels;
};

template <typename S>
MeshData<S> generateSphere(const SphereMeshParameters<S>& params) {
  const Eigen::Index numVertices = params.meridians * params.parallels + 2;
  const Eigen::Index numTriangles =
      2 * params.meridians + 2 * params.meridians * (params.parallels - 1);

  MeshData<S> mesh{};
  mesh.positions.resize(numVertices, 3);
  mesh.triangles.resize(numTriangles, 3);

  // north pole
  constexpr Eigen::Index kNorthPoleVertex{0};
  mesh.positions.row(kNorthPoleVertex) = Eigen::Vector3<S>{0., 0., params.radius};

  Eigen::Index currVertexIdx{1};
  for (Eigen::Index p = 1; p <= params.parallels; ++p) {
    // theta denotes the angle with respect to the z-axis
    const S theta = M_PI * p / static_cast<S>(params.parallels + 1);
    for (Eigen::Index m = 0; m < params.meridians; ++m) {
      // phi denotes the angle with respect to the x-axis
      const S phi = 2.0 * M_PI * m / static_cast<S>(params.meridians);
      const S x = params.radius * std::sin(theta) * std::cos(phi);
      const S y = params.radius * std::sin(theta) * std::sin(phi);
      const S z = params.radius * std::cos(theta);
      mesh.positions.row(currVertexIdx++) = Eigen::Vector3<S>{x, y, z};
    }
  }
  XR_CHECK(currVertexIdx == (numVertices - 1), "Number of vertices does not match");

  // south pole
  const Eigen::Index southPoleVertex{currVertexIdx};
  mesh.positions.row(southPoleVertex) = Eigen::Vector3<S>{0., 0., -params.radius};

  // triangles between the north pole and the first parallel
  Eigen::Index currTriangleIdx{0};
  for (Eigen::Index m = 0; m < params.meridians; ++m) {
    const Eigen::Index l{m + 1};
    const Eigen::Index r{(m + 1) % params.meridians + 1};
    mesh.triangles.row(currTriangleIdx++) = Eigen::Vector3i(kNorthPoleVertex, l, r);
  }

  // triangles between parallels
  for (Eigen::Index p = 0; p < params.parallels - 1; ++p) {
    const Eigen::Index p1 = p * params.meridians + 1;
    const Eigen::Index p2 = (p + 1) * params.meridians + 1;
    for (Eigen::Index m = 0; m < params.meridians; ++m) {
      const Eigen::Index ul{p1 + m};
      const Eigen::Index ur{p1 + (m + 1) % params.meridians};
      const Eigen::Index ll{p2 + m};
      const Eigen::Index lr{p2 + (m + 1) % params.meridians};
      mesh.triangles.row(currTriangleIdx++) = Eigen::Vector3i(ul, ll, ur);
      mesh.triangles.row(currTriangleIdx++) = Eigen::Vector3i(ll, lr, ur);
    }
  }

  // triangles between the last parallel and the south pole
  for (Eigen::Index m = 0; m < params.meridians; ++m) {
    const Eigen::Index l{m + params.meridians * (params.parallels - 1) + 1};
    const Eigen::Index r{
        (m + 1) % params.meridians + params.meridians * (params.parallels - 1) + 1};
    mesh.triangles.row(currTriangleIdx++) = Eigen::Vector3i(southPoleVertex, r, l);
  }

  return mesh;
}

template <typename S>
struct GridMeshParameters {
  S width;
  S height;
  S cellSize;
};

template <typename S>
MeshData<S> generateGrid(
    const GridMeshParameters<S>& params,
    const Eigen::Vector3<S>& translation = Eigen::Vector3<S>::Zero()) {
  // Compute the number of vertices along the width and height dimensions.
  const int numVerticesWidth = static_cast<int>(std::floor(params.width / params.cellSize)) + 1;
  const int numVerticesHeight = static_cast<int>(std::floor(params.height / params.cellSize)) + 1;

  MeshData<S> mesh{};
  mesh.positions.resize(numVerticesWidth * numVerticesHeight, 3);
  mesh.triangles.resize(2 * (numVerticesWidth - 1) * (numVerticesHeight - 1), 3);

  const auto index{
      [numVerticesWidth](const int x, const int y) { return y * numVerticesWidth + x; }};

  Eigen::Index currVertexIdx{0};
  Eigen::Index currTriangleIdx{0};
  for (int y = 0; y < numVerticesHeight; ++y) {
    for (int x = 0; x < numVerticesWidth; ++x) {
      mesh.positions.row(currVertexIdx++) =
          Eigen::Vector3<S>{static_cast<S>(x), static_cast<S>(y), 0.0} * params.cellSize +
          translation;

      // Two triangles for each cell.
      if (x > 0 && y > 0) {
        const Eigen::Index i1{index(x - 1, y - 1)};
        const Eigen::Index i2{index(x, y - 1)};
        const Eigen::Index i3{index(x - 1, y)};
        const Eigen::Index i4{index(x, y)};
        mesh.triangles.row(currTriangleIdx++) = Eigen::Vector3i(i1, i2, i3);
        mesh.triangles.row(currTriangleIdx++) = Eigen::Vector3i(i2, i4, i3);
      }
    }
  }

  return mesh;
}

} // namespace axel::test
