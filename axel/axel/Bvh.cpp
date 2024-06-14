/*
 * Portions Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* which bore the following notice:
Copyright (c) 2012 Brandon Pelfrey
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions: The above copyright notice and this
permission notice shall be included in all copies or substantial portions of the Software. THE
SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "axel/Bvh.h"

#include <array>

#include "axel/math/PointTriangleProjection.h"
#include "axel/math/RayTriangleIntersection.h"

namespace axel {

template <typename S, size_t LeafCapacity>
Bvh<S, LeafCapacity>::Bvh(const std::vector<BoundingBox<S>>& buildPrimitives)
    : buildPrimitives_{buildPrimitives} {
  build();
}

template <typename S, size_t LeafCapacity>
void Bvh<S, LeafCapacity>::setBoundingBoxes(const std::vector<BoundingBox<S>>& bboxes) {
  buildPrimitives_ = bboxes;
  build();
}

template <typename S, size_t LeafCapacity>
std::vector<BoundingBox<S>>& Bvh<S, LeafCapacity>::getPrimitives() {
  return buildPrimitives_;
}

template <typename S, size_t LeafCapacity>
const std::vector<BoundingBox<S>>& Bvh<S, LeafCapacity>::getPrimitives() const {
  return buildPrimitives_;
}

template <typename S, size_t LeafCapacity>
std::vector<uint32_t> Bvh<S, LeafCapacity>::query(const BoundingBox<S>& box) const {
  std::vector<uint32_t> hits{};
  if (nodeCount_ == 0) {
    return hits;
  }

  std::array<uint32_t, kMaxStackDepth> todoStack{};
  int32_t stackPtr = 0;
  todoStack[stackPtr] = 0;

  while (stackPtr >= 0) {
    const uint32_t nodeIdx = todoStack[stackPtr--];
    const BvhFlatNode<S>& node = flatTree_[nodeIdx];

    if (node.isLeaf()) {
      node.eachPrimitive(buildPrimitives_, [&](const BoundingBox<S>& primitiveBoundingBox) {
        hits.push_back(primitiveBoundingBox.id);
      });
    } else {
      const bool hitLeft = flatTree_[node.getLeftIndex(nodeIdx)].bbox.intersects(box);
      const bool hitRight = flatTree_[node.getRightIndex(nodeIdx)].bbox.intersects(box);
      if (hitLeft) {
        todoStack[++stackPtr] = node.getLeftIndex(nodeIdx);
      }
      if (hitRight) {
        todoStack[++stackPtr] = node.getRightIndex(nodeIdx);
      }
    }
  }

  return hits;
}

template <typename S, size_t LeafCapacity>
uint32_t Bvh<S, LeafCapacity>::query(const BoundingBox<S>& box, QueryBuffer& hits) const {
  uint32_t hitCount{0};
  if (nodeCount_ == 0) {
    return hitCount;
  }

  std::array<uint32_t, kMaxStackDepth> todoStack{};
  int32_t stackPtr = 0;
  todoStack[stackPtr] = 0;

  while (stackPtr >= 0 && hitCount < hits.size()) {
    const uint32_t nodeIdx = todoStack[stackPtr--];
    const BvhFlatNode<S>& node = flatTree_[nodeIdx];

    if (node.isLeaf()) {
      node.eachPrimitive(buildPrimitives_, [&](const BoundingBox<S>& primitiveBoundingBox) {
        hits[hitCount++] = primitiveBoundingBox.id;
      });
    } else {
      const bool hitLeft = flatTree_[node.getLeftIndex(nodeIdx)].bbox.intersects(box);
      const bool hitRight = flatTree_[node.getRightIndex(nodeIdx)].bbox.intersects(box);
      if (hitLeft) {
        todoStack[++stackPtr] = node.getLeftIndex(nodeIdx);
      }
      if (hitRight) {
        todoStack[++stackPtr] = node.getRightIndex(nodeIdx);
      }
    }
  }

  return hitCount;
}

template <typename S, size_t LeafCapacity>
std::vector<uint32_t> Bvh<S, LeafCapacity>::query(
    const Eigen::Vector3<S>& origin,
    const Eigen::Vector3<S>& direction) const {
  std::vector<uint32_t> hits;

  // Working set
  std::array<uint32_t, kMaxStackDepth> todoStack{};
  int32_t stackPtr = 0;
  todoStack[stackPtr] = 0;

  while (stackPtr >= 0) {
    const uint32_t nodeIdx = todoStack[stackPtr--];
    const BvhFlatNode<S>& node(flatTree_[nodeIdx]);

    if (node.isLeaf()) {
      node.eachPrimitive(buildPrimitives_, [&](const BoundingBox<S>& primitiveBoundingBox) {
        if (primitiveBoundingBox.intersects(origin, direction)) {
          hits.push_back(primitiveBoundingBox.id);
        }
      });
    } else {
      const bool hitLeft = flatTree_[node.getLeftIndex(nodeIdx)].bbox.intersects(origin, direction);
      const bool hitRight =
          flatTree_[node.getRightIndex(nodeIdx)].bbox.intersects(origin, direction);
      if (hitLeft) {
        todoStack[++stackPtr] = node.getLeftIndex(nodeIdx);
      }
      if (hitRight) {
        todoStack[++stackPtr] = node.getRightIndex(nodeIdx);
      }
    }
  }
  return hits;
}

template <typename S, size_t LeafCapacity>
Size Bvh<S, LeafCapacity>::getNodeCount() const {
  return nodeCount_;
}

template <typename S, size_t LeafCapacity>
Size Bvh<S, LeafCapacity>::getPrimitiveCount() const {
  return buildPrimitives_.size();
}

namespace {

template <typename S>
void refitRecurse(
    std::vector<BvhFlatNode<S>>& nodes,
    const Index index,
    const std::vector<BoundingBox<S>>& primitives) {
  auto& node = nodes[index];
  if (node.isLeaf()) {
    node.bbox = primitives[node.start];
    node.eachPrimitive(primitives, 1, [&](const BoundingBox<S>& prim) { node.bbox.extend(prim); });
    return;
  }

  const auto leftChildIndex = node.getLeftIndex(index);
  const auto rightChildIndex = node.getRightIndex(index);
  refitRecurse<S>(nodes, leftChildIndex, primitives);
  refitRecurse<S>(nodes, rightChildIndex, primitives);

  const auto& leftChild = nodes[leftChildIndex];
  const auto& rightChild = nodes[rightChildIndex];
  node.bbox.extend(leftChild.bbox);
  node.bbox.extend(rightChild.bbox);
}

} // namespace

template <typename S, size_t LeafCapacity>
void Bvh<S, LeafCapacity>::refit() {
  if (nodeCount_ == 0) {
    return;
  }

  refitRecurse<S>(flatTree_, 0, buildPrimitives_);
}

namespace {
struct BvhBuildEntry {
  // If non-zero then this is the index of the parent. (used in offsets)
  uint32_t parent;
  // The range of objects in the object list covered by this node.
  uint32_t start;
  uint32_t end;
};
} // namespace

template <typename S, size_t LeafCapacity>
void Bvh<S, LeafCapacity>::build() {
  flatTree_.clear();
  nodeCount_ = 0;

  if (buildPrimitives_.empty()) {
    return;
  }

  flatTree_.reserve(buildPrimitives_.size() * 2);

  static constexpr uint32_t kUntouched = 0xFFFFFFFF;
  static constexpr uint32_t kTouchedTwice = 0xFFFFFFFD;
  static constexpr uint32_t kRootParentIndex = 0xFFFFFFFC;

  std::array<BvhBuildEntry, 128> todoStack{};
  uint32_t stackPtr = 0; // The stack pointer points right above the top of the stack.

  // Push the root. It covers all primitives at the beginning of the build.
  todoStack[stackPtr].start = 0;
  todoStack[stackPtr].end = static_cast<uint32_t>(buildPrimitives_.size());
  todoStack[stackPtr].parent = kRootParentIndex;
  stackPtr++;

  BvhFlatNode<S> currNode;
  while (stackPtr > 0) {
    BvhBuildEntry& buildNode = todoStack[--stackPtr];
    const uint32_t start = buildNode.start;
    const uint32_t end = buildNode.end;
    const uint32_t primitiveCount = end - start;

    nodeCount_++;
    currNode.start = start;
    currNode.nPrims = primitiveCount;
    currNode.rightOffset = kUntouched;

    // Calculate the bounding box for this node by going over its primitives.
    currNode.bbox = buildPrimitives_[start];
    BoundingBox<S> boundingBoxCenters(buildPrimitives_[start].center());
    for (uint32_t p = start + 1; p < end; ++p) {
      currNode.bbox.extend(buildPrimitives_[p]);
      boundingBoxCenters.extend(buildPrimitives_[p].center());
    }

    // If the number of primitives at this point is less than the leaf
    // size, then this will become a leaf. (Signified by isLeaf())
    if (primitiveCount <= LeafCapacity) {
      currNode.rightOffset = 0;
    }

    flatTree_.push_back(currNode);

    // Child touches parent...
    // Special case: Don't do this for the root.
    if (buildNode.parent != kRootParentIndex) {
      flatTree_[buildNode.parent].rightOffset--;

      // When this is the second touch, this is the right child.
      // The right child sets up the offset for the flat tree.
      if (flatTree_[buildNode.parent].rightOffset == kTouchedTwice) {
        flatTree_[buildNode.parent].rightOffset = nodeCount_ - 1 - buildNode.parent;
      }
    }

    // If this is a leaf, no need to subdivide.
    if (currNode.isLeaf()) {
      continue;
    }

    // Otherwise, find the axis index to be used for splitting.
    const Index splitDim = boundingBoxCenters.maxDimension();

    // Split on the center of the longest axis.
    const S splitCoord = static_cast<S>(0.5) *
        (boundingBoxCenters.min()[splitDim] + boundingBoxCenters.max()[splitDim]);
    const S splitCoordx2 = 2 * splitCoord;

    // Partition the list of objects on this split
    uint32_t mid = start;
    for (uint32_t i = start; i < end; ++i) {
      const auto aabbMin = buildPrimitives_[i].aabb.min()[splitDim];
      const auto aabbMax = buildPrimitives_[i].aabb.max()[splitDim];
      if (aabbMax <= splitCoord || (aabbMin <= splitCoord && aabbMax < (splitCoordx2 - aabbMin))) {
        if (i != mid) {
          std::swap(buildPrimitives_[i], buildPrimitives_[mid]);
        }
        ++mid;
      }
    }

    // If we get a bad split, just choose the center...
    if (mid == start || mid == end) {
      mid = start + (end - start) / 2;
    }

    // Push right child
    todoStack[stackPtr].start = mid;
    todoStack[stackPtr].end = end;
    todoStack[stackPtr].parent = nodeCount_ - 1;
    stackPtr++;

    // Push left child
    todoStack[stackPtr].start = start;
    todoStack[stackPtr].end = mid;
    todoStack[stackPtr].parent = nodeCount_ - 1;
    stackPtr++;
  }
}

namespace {

template <typename S>
bool checkBoundingBoxesRecurse(
    const uint32_t nodeIdx,
    const std::vector<BvhFlatNode<S>>& flatTree,
    const std::vector<BoundingBox<S>>& buildPrimitives) {
  const BvhFlatNode<S>& node = flatTree[nodeIdx];
  if (node.isLeaf()) {
    return node.eachPrimitive(buildPrimitives, [&](const BoundingBox<S>& obj) -> bool {
      return node.bbox.contains(obj);
    });
  }

  if (!node.bbox.contains(flatTree[node.getLeftIndex(nodeIdx)].bbox)) {
    return false;
  }

  if (!node.bbox.contains(flatTree[node.getRightIndex(nodeIdx)].bbox)) {
    return false;
  }

  const bool leftCorrect =
      checkBoundingBoxesRecurse(node.getLeftIndex(nodeIdx), flatTree, buildPrimitives);
  const bool rightCorrect =
      checkBoundingBoxesRecurse(node.getRightIndex(nodeIdx), flatTree, buildPrimitives);
  return leftCorrect && rightCorrect;
}

} // namespace

template <typename S, size_t LeafCapacity>
bool Bvh<S, LeafCapacity>::checkBoundingBoxes() const {
  if (nodeCount_ == 0) {
    return true;
  }
  return checkBoundingBoxesRecurse(0, flatTree_, buildPrimitives_);
}

template class Bvh<float, 1>;
template class Bvh<double, 1>;
template class Bvh<float, kNativeLaneWidth<float>>;
template class Bvh<double, kNativeLaneWidth<double>>;

} // namespace axel
