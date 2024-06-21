/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/types.h>
#include <momentum/math/types.h>

namespace momentum {

// skinning class including blendshape vectors/offsets
// it used to model facial expressions and (and potentially for other purposes, e.g. pose-dependent
// shape deformations)
struct BlendShapeBase {
 public:
  BlendShapeBase() {}
  BlendShapeBase(size_t modelSize, size_t numShapes);

  void setShapeVectors(const MatrixXf& shapeVectors) {
    shapeVectors_ = shapeVectors;
  }

  const MatrixXf& getShapeVectors() const {
    return shapeVectors_;
  }

  template <typename T>
  VectorX<T> computeDeltas(const BlendWeightsT<T>& blendWeights) const;

  template <typename T>
  void applyDeltas(const BlendWeightsT<T>& blendWeights, std::vector<Eigen::Vector3<T>>& result)
      const;

  void setShapeVector(size_t index, gsl::span<const Vector3f> shapeVector);

  Eigen::Index shapeSize() const {
    return shapeVectors_.cols();
  }

  size_t modelSize() const {
    return shapeVectors_.rows() / 3;
  }

 protected:
  MatrixXf shapeVectors_;
};

} // namespace momentum
