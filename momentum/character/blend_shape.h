/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/blend_shape_base.h>

namespace momentum {

// skinning class including both blendshape vectors/offsets (given by BlendShapeBase) and mean
// template (e.g., PCA mean) it used to model identity-dependent shape deformations
struct BlendShape : public BlendShapeBase {
 public:
  BlendShape() : factorizationValid_(false) {}
  BlendShape(gsl::span<const Vector3f> baseShape, size_t numShapes);

  void setBaseShape(gsl::span<const Vector3f> baseShape) {
    baseShape_.assign(baseShape.begin(), baseShape.end());
  };

  const std::vector<Vector3f>& getBaseShape() const {
    return baseShape_;
  };

  bool getFactorizationValid() const {
    return factorizationValid_;
  };

  template <typename T>
  std::vector<Eigen::Vector3<T>> computeShape(const BlendWeightsT<T>& coefficients) const;

  template <typename T>
  void computeShape(const BlendWeightsT<T>& coefficients, std::vector<Eigen::Vector3<T>>& result)
      const;

  VectorXf estimateCoefficients(
      gsl::span<const Vector3f> vertices,
      float regularization = 1.0f,
      const VectorXf& weights = VectorXf()) const;

  void setShapeVector(size_t index, gsl::span<const Vector3f> shapeVector);

  bool isApprox(const BlendShape& blendShape) const;

 private:
  std::vector<Vector3f> baseShape_;
  mutable Eigen::JacobiSVD<MatrixXf> factorization_;
  mutable bool factorizationValid_;
};

} // namespace momentum
