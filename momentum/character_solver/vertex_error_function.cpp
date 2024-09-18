/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_solver/vertex_error_function.h"

#include "momentum/character/blend_shape.h"
#include "momentum/character/blend_shape_skinning.h"
#include "momentum/character/character.h"
#include "momentum/character/linear_skinning.h"
#include "momentum/character/skeleton.h"
#include "momentum/character/skeleton_state.h"
#include "momentum/character/skin_weights.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"
#include "momentum/math/mesh.h"
#include "momentum/math/utility.h"

namespace momentum {

std::string_view toString(VertexConstraintType type) {
  static const std::array<std::string_view, 4> strings = {
      "Position", "Plane", "Normal", "SymmetricNormal"};
  return strings[static_cast<size_t>(type)];
}

template <typename T>
VertexErrorFunctionT<T>::VertexErrorFunctionT(
    const Character& character_in,
    VertexConstraintType type)
    : SkeletonErrorFunctionT<T>(character_in.skeleton, character_in.parameterTransform),
      character_(character_in),
      constraintType_(type) {
  MT_CHECK(static_cast<bool>(character_in.mesh));
  MT_CHECK(static_cast<bool>(character_in.skinWeights));
  MT_THROW_IF(
      character_in.faceExpressionBlendShape && (type != VertexConstraintType::Position),
      "Constraint type {} not implemented yet for face. Only Position type is supported.",
      toString(type));
  this->neutralMesh_ = std::make_unique<MeshT<T>>(character_in.mesh->template cast<T>());
  this->restMesh_ = std::make_unique<MeshT<T>>(character_in.mesh->template cast<T>());
  this->posedMesh_ = std::make_unique<MeshT<T>>(character_in.mesh->template cast<T>());
}

template <typename T>
VertexErrorFunctionT<T>::~VertexErrorFunctionT() {}

template <typename T>
void VertexErrorFunctionT<T>::clearConstraints() {
  constraints_.clear();
}

template <typename T>
void VertexErrorFunctionT<T>::addConstraint(
    int vertexIndex,
    T weight,
    const Eigen::Vector3<T>& targetPosition,
    const Eigen::Vector3<T>& targetNormal) {
  MT_CHECK(vertexIndex >= 0 && ((size_t)vertexIndex) < character_.mesh->vertices.size());
  constraints_.push_back(VertexConstraintT<T>{vertexIndex, weight, targetPosition, targetNormal});
}

template <typename T>
double VertexErrorFunctionT<T>::getError(
    const ModelParametersT<T>& modelParameters,
    const SkeletonStateT<T>& state) {
  MT_PROFILE_EVENT("VertexErrorFunction - getError");

  updateMeshes(modelParameters, state);

  // loop over all constraints and calculate the error
  double error = 0.0;

  if (constraintType_ == VertexConstraintType::Position) {
    for (size_t i = 0; i < constraints_.size(); ++i) {
      const VertexConstraintT<T>& constr = constraints_[i];

      const Eigen::Vector3<T> diff =
          this->posedMesh_->vertices[constr.vertexIndex] - constr.targetPosition;
      error += constr.weight * diff.squaredNorm() * kPositionWeight;
    }
  } else {
    const auto [sourceNormalWeight, targetNormalWeight] = computeNormalWeights();
    for (size_t i = 0; i < constraints_.size(); ++i) {
      const VertexConstraintT<T>& constr = constraints_[i];

      const Eigen::Vector3<T> sourceNormal = this->posedMesh_->normals[constr.vertexIndex];
      Eigen::Vector3<T> targetNormal = constr.targetNormal;
      if (sourceNormal.dot(constr.targetNormal) < 0) {
        targetNormal *= -1;
      }
      const Eigen::Vector3<T> normal =
          sourceNormalWeight * sourceNormal + targetNormalWeight * targetNormal;

      // calculate point->plane distance and error
      const Eigen::Vector3<T> diff =
          this->posedMesh_->vertices[constr.vertexIndex] - constr.targetPosition;
      const T dist = normal.dot(diff);
      error += constr.weight * dist * dist * kPlaneWeight;
    }
  }

  // return error
  return error * this->weight_;
}

template <typename T>
struct BoneWeightT {
  size_t parentBone;
  float weight;
  Eigen::Vector3<T> weightedWorldSpacePoint;

  bool operator<(const BoneWeightT<T>& rhs) const {
    return parentBone < rhs.parentBone;
  }

  bool operator>(const BoneWeightT<T>& rhs) const {
    return parentBone > rhs.parentBone;
  }

  BoneWeightT<T>& operator+=(const BoneWeightT<T>& rhs) {
    this->weight += rhs.weight;
    this->weightedWorldSpacePoint += rhs.weightedWorldSpacePoint;
    return *this;
  }
};

// When we compute the derivatives of skinned points, we usually
// end up with a loop like this:
//   for each skinned point i
//     for each skinning bone and weight B_ij, w_ij
//       for each ancestor k
//          compute the weighted change in the skinned point wrt the ancestor
// The redundancy here is that most points are skinned to a set of bones in the
// same hierarchy (for example, two parts of the arm) and so it's redundant to
// walk up the whole hierarchy from scratch for each skinned bone.
//
// This class instead does a single pass up the tree from leaf-most to root-most
// skinned bones, accumulating skinning weights along the way.  By packaging it
// in this class we hide the complexity and simplify the jacobian/gradient
// calculation.
template <typename T>
class SkinningWeightIteratorT {
 public:
  SkinningWeightIteratorT(
      const Character& character,
      const MeshT<T>& restMesh,
      const SkeletonStateT<T>& skelState,
      int vertexIndex)
      : character(character), restMesh(restMesh) {
    const auto& skinWeights = *character.skinWeights;
    nBoneWeights = 0;
    {
      for (uint32_t i = 0; i < kMaxSkinJoints; ++i) {
        const auto w = skinWeights.weight(vertexIndex, i);
        const auto parentBone = skinWeights.index(vertexIndex, i);
        if (w > 0) {
          boneWeights[nBoneWeights++] = {
              parentBone,
              w,
              T(w) *
                  (skelState.jointState[parentBone].transformation *
                   (character.inverseBindPose[parentBone].template cast<T>() *
                    restMesh.vertices[vertexIndex]))};
        }
      }
      std::sort(
          boneWeights.begin(), boneWeights.begin() + nBoneWeights, std::greater<BoneWeightT<T>>());
    }
    checkInvariants();
  }

  bool finished() const {
    return nBoneWeights == 0;
  }

  // Returns the tuple <parent bone index, bone weight, vertex position in world space wrt the
  // current bone>
  std::tuple<size_t, T, Eigen::Vector3<T>> next() {
    MT_CHECK(nBoneWeights != 0);
    const BoneWeightT<T> result = boneWeights[0];

    boneWeights[0].parentBone = this->character.skeleton.joints[result.parentBone].parent;
    if (boneWeights[0].parentBone == kInvalidIndex) {
      // Reached the root, so we're done with this one:
      for (int i = 1; i < nBoneWeights; ++i) {
        boneWeights[i - 1] = boneWeights[i];
      }
      --nBoneWeights;
    } else {
      // We decreased the bone index by moving up to its parent; now figure out
      // where the bone weight should be relocated in the list.
      int i = 1;
      while (i < nBoneWeights && boneWeights[i - 1].parentBone < boneWeights[i].parentBone) {
        std::swap(boneWeights[i - 1], boneWeights[i]);
        ++i;
      }

      if (i < nBoneWeights && boneWeights[i - 1].parentBone == boneWeights[i].parentBone) {
        // Merge them:
        boneWeights[i - 1] += boneWeights[i];

        // strip out the duplicate:
        ++i;
        while (i < nBoneWeights) {
          boneWeights[i - 1] = boneWeights[i];
          ++i;
        }
        --nBoneWeights;
      }
    }

    checkInvariants();
    return {result.parentBone, result.weight, result.weightedWorldSpacePoint / result.weight};
  }

 private:
  void checkInvariants() {
#ifndef NDEBUG
    if (nBoneWeights > 0) {
      // Check the invariants:
      float sum = 0;
      for (int i = 0; i < nBoneWeights; ++i) {
        MT_CHECK(boneWeights[i].parentBone < this->character.skeleton.joints.size());
        MT_CHECK(boneWeights[i].weight > 0 && boneWeights[i].weight <= 1.001);
        MT_CHECK(i == 0 || boneWeights[i - 1].parentBone > boneWeights[i].parentBone);
        sum += boneWeights[i].weight;
      }
      MT_CHECK(sum > 0.99f && sum < 1.01f);
    }
#endif
  }

  std::array<BoneWeightT<T>, kMaxSkinJoints> boneWeights;
  int nBoneWeights;
  const Character& character;
  const MeshT<T>& restMesh;
};

template <typename T>
void gradient_jointParams_to_modelParams(
    const T grad_jointParams,
    const Eigen::Index iJointParam,
    const ParameterTransform& parameterTransform,
    Eigen::Ref<Eigen::VectorX<T>> gradient) {
  // explicitly multiply with the parameter transform to generate parameter space gradients
  for (auto index = parameterTransform.transform.outerIndexPtr()[iJointParam];
       index < parameterTransform.transform.outerIndexPtr()[iJointParam + 1];
       ++index) {
    gradient[parameterTransform.transform.innerIndexPtr()[index]] +=
        grad_jointParams * parameterTransform.transform.valuePtr()[index];
  }
}

template <typename T>
void jacobian_jointParams_to_modelParams(
    const Eigen::Ref<const Eigen::VectorX<T>> jacobian_jointParams,
    const Eigen::Index iJointParam,
    const ParameterTransform& parameterTransform,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian) {
  // explicitly multiply with the parameter transform to generate parameter space gradients
  for (auto index = parameterTransform.transform.outerIndexPtr()[iJointParam];
       index < parameterTransform.transform.outerIndexPtr()[iJointParam + 1];
       ++index) {
    jacobian.col(parameterTransform.transform.innerIndexPtr()[index]) +=
        jacobian_jointParams * parameterTransform.transform.valuePtr()[index];
  }
}

template <typename T>
void jacobian_jointParams_to_modelParams(
    const T jacobian_jointParams,
    const Eigen::Index iJointParam,
    const ParameterTransform& parameterTransform,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian) {
  // explicitly multiply with the parameter transform to generate parameter space gradients
  for (auto index = parameterTransform.transform.outerIndexPtr()[iJointParam];
       index < parameterTransform.transform.outerIndexPtr()[iJointParam + 1];
       ++index) {
    jacobian(0, parameterTransform.transform.innerIndexPtr()[index]) +=
        jacobian_jointParams * parameterTransform.transform.valuePtr()[index];
  }
}

template <typename T>
void VertexErrorFunctionT<T>::calculateDWorldPos(
    const SkeletonStateT<T>& state,
    const VertexConstraintT<T>& constr,
    const Eigen::Vector3<T>& d_restPos,
    Eigen::Vector3<T>& d_worldPos) const {
  const auto& skinWeights = *character_.skinWeights;

  for (uint32_t i = 0; i < kMaxSkinJoints; ++i) {
    const auto w = skinWeights.weight(constr.vertexIndex, i);
    const auto parentBone = skinWeights.index(constr.vertexIndex, i);
    if (w > 0) {
      d_worldPos += w *
          (state.jointState[parentBone].transformation.linear() *
           (character_.inverseBindPose[parentBone].linear().template cast<T>() * d_restPos));
    }
  }
}

template <typename T>
double VertexErrorFunctionT<T>::calculatePositionGradient(
    const ModelParametersT<T>& /* modelParameters */,
    const SkeletonStateT<T>& state,
    const VertexConstraintT<T>& constr,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  const Eigen::Vector3<T> diff =
      this->posedMesh_->vertices[constr.vertexIndex] - constr.targetPosition;

  // calculate the difference between target and position and error
  const T wgt = constr.weight * 2.0f * kPositionWeight * this->weight_;

  SkinningWeightIteratorT<T> skinningIter(
      this->character_, *this->restMesh_, state, constr.vertexIndex);

  // IN handle derivatives wrt jointParameters
  while (!skinningIter.finished()) {
    size_t jointIndex;
    T boneWeight;
    Eigen::Vector3<T> pos;
    std::tie(jointIndex, boneWeight, pos) = skinningIter.next();

    // check for valid index
    MT_CHECK(jointIndex < this->skeleton_.joints.size());

    const auto& jointState = state.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Eigen::Vector3<T> posd = pos - jointState.translation();

    // calculate derivatives based on active joints
    for (size_t d = 0; d < 3; d++) {
      if (this->activeJointParams_[paramIndex + d]) {
        // Gradient wrt translation:
        gradient_jointParams_to_modelParams(
            boneWeight * diff.dot(jointState.getTranslationDerivative(d)) * wgt,
            paramIndex + d,
            this->parameterTransform_,
            gradient);
      }
      if (this->activeJointParams_[paramIndex + 3 + d]) {
        // Gradient wrt rotation:
        gradient_jointParams_to_modelParams(
            boneWeight * diff.dot(jointState.getRotationDerivative(d, posd)) * wgt,
            paramIndex + 3 + d,
            this->parameterTransform_,
            gradient);
      }
    }
    if (this->activeJointParams_[paramIndex + 6]) {
      // Gradient wrt scale:
      gradient_jointParams_to_modelParams(
          boneWeight * diff.dot(jointState.getScaleDerivative(posd)) * wgt,
          paramIndex + 6,
          this->parameterTransform_,
          gradient);
    }
  } // OUT handle derivatives wrt jointParameters

  // IN handle derivatives wrt blend shape parameters
  if (this->character_.blendShape) {
    for (Eigen::Index iBlendShape = 0;
         iBlendShape < this->parameterTransform_.blendShapeParameters.size();
         ++iBlendShape) {
      const auto paramIdx = this->parameterTransform_.blendShapeParameters[iBlendShape];
      if (paramIdx < 0) {
        continue;
      }

      const Eigen::Vector3<T> d_restPos =
          this->character_.blendShape->getShapeVectors()
              .template block<3, 1>(3 * constr.vertexIndex, iBlendShape, 3, 1)
              .template cast<T>();
      Eigen::Vector3<T> d_worldPos = Eigen::Vector3<T>::Zero();
      calculateDWorldPos(state, constr, d_restPos, d_worldPos);

      gradient[paramIdx] += wgt * diff.dot(d_worldPos);
    }
  }
  // OUT handle derivatives wrt blend shape parameters

  // IN handle derivatives wrt face expression blend shape parameters
  if (this->character_.faceExpressionBlendShape) {
    for (Eigen::Index iBlendShape = 0;
         iBlendShape < this->parameterTransform_.faceExpressionParameters.size();
         ++iBlendShape) {
      const auto paramIdx = this->parameterTransform_.faceExpressionParameters[iBlendShape];
      if (paramIdx < 0) {
        continue;
      }

      const Eigen::Vector3<T> d_restPos =
          this->character_.faceExpressionBlendShape->getShapeVectors()
              .template block<3, 1>(3 * constr.vertexIndex, iBlendShape, 3, 1)
              .template cast<T>();
      Eigen::Vector3<T> d_worldPos = Eigen::Vector3<T>::Zero();
      calculateDWorldPos(state, constr, d_restPos, d_worldPos);

      gradient[paramIdx] += wgt * diff.dot(d_worldPos);
    }
  }
  // OUT handle derivatives wrt face expression blend shape parameters

  return constr.weight * diff.squaredNorm() * kPositionWeight;
}

template <typename T>
double VertexErrorFunctionT<T>::calculatePositionJacobian(
    const ModelParametersT<T>& /* modelParameters */,
    const SkeletonStateT<T>& state,
    const VertexConstraintT<T>& constr,
    Ref<Eigen::MatrixX<T>> jac,
    Ref<Eigen::VectorX<T>> res) const {
  // calculate the difference between target and position and error
  const Eigen::Vector3<T> diff =
      this->posedMesh_->vertices[constr.vertexIndex] - constr.targetPosition;

  const T wgt = std::sqrt(constr.weight * kPositionWeight * this->weight_);

  SkinningWeightIteratorT<T> skinningIter(
      this->character_, *this->restMesh_, state, constr.vertexIndex);

  // IN handle derivatives wrt jointParameters
  while (!skinningIter.finished()) {
    size_t jointIndex;
    T boneWeight;
    Eigen::Vector3<T> pos;
    std::tie(jointIndex, boneWeight, pos) = skinningIter.next();

    // check for valid index
    MT_CHECK(jointIndex < this->skeleton_.joints.size());

    const auto& jointState = state.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Eigen::Vector3<T> posd = pos - jointState.translation();

    // calculate derivatives based on active joints
    for (size_t d = 0; d < 3; d++) {
      if (this->activeJointParams_[paramIndex + d]) {
        jacobian_jointParams_to_modelParams<T>(
            boneWeight * wgt * jointState.getTranslationDerivative(d),
            paramIndex + d,
            this->parameterTransform_,
            jac);
      }
      if (this->activeJointParams_[paramIndex + 3 + d]) {
        jacobian_jointParams_to_modelParams<T>(
            boneWeight * wgt * jointState.getRotationDerivative(d, posd),
            paramIndex + d + 3,
            this->parameterTransform_,
            jac);
      }
    }
    if (this->activeJointParams_[paramIndex + 6]) {
      jacobian_jointParams_to_modelParams<T>(
          boneWeight * wgt * jointState.getScaleDerivative(posd),
          paramIndex + 6,
          this->parameterTransform_,
          jac);
    }
  }
  // OUT handle derivatives wrt jointParameters

  // IN handle derivatives wrt blend shape parameters
  if (this->character_.blendShape) {
    for (Eigen::Index iBlendShape = 0;
         iBlendShape < this->parameterTransform_.blendShapeParameters.size();
         ++iBlendShape) {
      const auto paramIdx = this->parameterTransform_.blendShapeParameters[iBlendShape];
      if (paramIdx < 0) {
        continue;
      }

      const Eigen::Vector3<T> d_restPos =
          this->character_.blendShape->getShapeVectors()
              .template block<3, 1>(3 * constr.vertexIndex, iBlendShape, 3, 1)
              .template cast<T>();
      Eigen::Vector3<T> d_worldPos = Eigen::Vector3<T>::Zero();
      calculateDWorldPos(state, constr, d_restPos, d_worldPos);

      jac.col(paramIdx) += wgt * d_worldPos;
    }
  }
  // OUT handle derivatives wrt blend shape parameters

  // IN handle derivatives wrt face expression blend shape parameters
  if (this->character_.faceExpressionBlendShape) {
    for (Eigen::Index iBlendShape = 0;
         iBlendShape < this->parameterTransform_.faceExpressionParameters.size();
         ++iBlendShape) {
      const auto paramIdx = this->parameterTransform_.faceExpressionParameters[iBlendShape];
      if (paramIdx < 0) {
        continue;
      }

      const Eigen::Vector3<T> d_restPos =
          this->character_.faceExpressionBlendShape->getShapeVectors()
              .template block<3, 1>(3 * constr.vertexIndex, iBlendShape, 3, 1)
              .template cast<T>();
      Eigen::Vector3<T> d_worldPos = Eigen::Vector3<T>::Zero();
      calculateDWorldPos(state, constr, d_restPos, d_worldPos);

      jac.col(paramIdx) += wgt * d_worldPos;
    }
  }
  // OUT handle derivatives wrt face expression blend shape parameters

  res = diff * wgt;

  return wgt * wgt * diff.squaredNorm();
}

template <typename T>
double VertexErrorFunctionT<T>::calculateNormalGradient(
    const ModelParametersT<T>& /* modelParameters */,
    const SkeletonStateT<T>& state,
    const VertexConstraintT<T>& constr,
    const T sourceNormalWeight,
    const T targetNormalWeight,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  const auto& skinWeights = *character_.skinWeights;

  const Eigen::Vector3<T> sourceNormal = posedMesh_->normals[constr.vertexIndex];
  Eigen::Vector3<T> targetNormal = constr.targetNormal;
  if (sourceNormal.dot(constr.targetNormal) < 0) {
    targetNormal *= -1;
  }
  const Eigen::Vector3<T> normal =
      sourceNormalWeight * sourceNormal + targetNormalWeight * targetNormal;

  const Eigen::Vector3<T> diff =
      this->posedMesh_->vertices[constr.vertexIndex] - constr.targetPosition;
  const T dist = diff.dot(normal);

  // calculate the difference between target and position and error
  const T wgt = constr.weight * 2.0f * kPositionWeight * this->weight_;

  SkinningWeightIteratorT<T> skinningIter(
      this->character_, *this->restMesh_, state, constr.vertexIndex);

  // IN handle derivatives wrt jointParameters
  while (!skinningIter.finished()) {
    const auto [jointIndex, boneWeight, pos] = skinningIter.next();

    // check for valid index
    MT_CHECK(jointIndex < this->skeleton_.joints.size());

    const auto& jointState = state.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Eigen::Vector3<T> posd = pos - jointState.translation();

    // calculate derivatives based on active joints
    for (size_t d = 0; d < 3; d++) {
      if (this->activeJointParams_[paramIndex + d]) {
        // Gradient wrt translation:
        gradient_jointParams_to_modelParams(
            boneWeight * wgt * dist * normal.dot(jointState.getTranslationDerivative(d)),
            paramIndex + d,
            this->parameterTransform_,
            gradient);
      }
      if (this->activeJointParams_[paramIndex + 3 + d]) {
        // Gradient wrt rotation:
        const Eigen::Vector3<T> cross =
            -sourceNormal.cross(constr.targetPosition - jointState.translation());
        const T diff_src = jointState.rotationAxis.col(d).dot(cross);
        const T diff_tgt = targetNormal.dot(jointState.getRotationDerivative(d, posd));

        gradient_jointParams_to_modelParams(
            boneWeight * wgt * dist *
                (sourceNormalWeight * diff_src + targetNormalWeight * diff_tgt),
            paramIndex + 3 + d,
            this->parameterTransform_,
            gradient);
      }
    }
    if (this->activeJointParams_[paramIndex + 6]) {
      // Gradient wrt scale:
      gradient_jointParams_to_modelParams(
          boneWeight * wgt * dist * normal.dot(jointState.getScaleDerivative(posd)),
          paramIndex + 6,
          this->parameterTransform_,
          gradient);
    }
  } // OUT handle derivatives wrt jointParameters

  // IN handle derivatives wrt blend shape parameters:
  if (this->character_.blendShape) {
    for (Eigen::Index iBlendShape = 0;
         iBlendShape < this->parameterTransform_.blendShapeParameters.size();
         ++iBlendShape) {
      const auto paramIdx = this->parameterTransform_.blendShapeParameters[iBlendShape];
      if (paramIdx < 0) {
        continue;
      }

      const Eigen::Vector3<T> d_restPos =
          this->character_.blendShape->getShapeVectors()
              .template block<3, 1>(3 * constr.vertexIndex, iBlendShape, 3, 1)
              .template cast<T>();
      Eigen::Vector3<T> d_worldPos = Eigen::Vector3<T>::Zero();

      for (uint32_t jWeight = 0; jWeight < kMaxSkinJoints; ++jWeight) {
        const auto w = skinWeights.weight(constr.vertexIndex, jWeight);
        const auto parentBone = skinWeights.index(constr.vertexIndex, jWeight);
        if (w > 0) {
          d_worldPos += w *
              (state.jointState[parentBone].transformation.linear() *
               (character_.inverseBindPose[parentBone].linear().template cast<T>() * d_restPos));
        }
      }

      gradient[paramIdx] += wgt * dist * normal.dot(d_worldPos);
    }
  }
  // OUT handle derivatives wrt blend shape parameters:

  return constr.weight * dist * dist * kPlaneWeight;
}

template <typename T>
double VertexErrorFunctionT<T>::calculateNormalJacobian(
    const ModelParametersT<T>& /* modelParameters */,
    const SkeletonStateT<T>& state,
    const VertexConstraintT<T>& constr,
    const T sourceNormalWeight,
    const T targetNormalWeight,
    Ref<Eigen::MatrixX<T>> jac,
    T& res) const {
  const auto& skinWeights = *character_.skinWeights;

  const Eigen::Vector3<T> sourceNormal = posedMesh_->normals[constr.vertexIndex];
  Eigen::Vector3<T> targetNormal = constr.targetNormal;
  if (sourceNormal.dot(constr.targetNormal) < 0) {
    targetNormal *= -1;
  }
  const Eigen::Vector3<T> normal =
      sourceNormalWeight * sourceNormal + targetNormalWeight * targetNormal;

  // calculate the difference between target and position and error
  const Eigen::Vector3<T> diff =
      this->posedMesh_->vertices[constr.vertexIndex] - constr.targetPosition;
  const T dist = diff.dot(normal);

  const T wgt = std::sqrt(constr.weight * kPositionWeight * this->weight_);

  SkinningWeightIteratorT<T> skinningIter(
      this->character_, *this->restMesh_, state, constr.vertexIndex);

  // IN handle derivatives wrt jointParameters
  while (!skinningIter.finished()) {
    const auto [jointIndex, boneWeight, pos] = skinningIter.next();

    // check for valid index
    MT_CHECK(jointIndex < this->skeleton_.joints.size());

    const auto& jointState = state.jointState[jointIndex];
    const size_t paramIndex = jointIndex * kParametersPerJoint;
    const Eigen::Vector3<T> posd = pos - jointState.translation();

    // calculate derivatives based on active joints
    for (size_t d = 0; d < 3; d++) {
      if (this->activeJointParams_[paramIndex + d]) {
        // Jacobian wrt translation:
        jacobian_jointParams_to_modelParams<T>(
            boneWeight * wgt * normal.dot(jointState.getTranslationDerivative(d)),
            paramIndex + d,
            this->parameterTransform_,
            jac);
      }
      if (this->activeJointParams_[paramIndex + 3 + d]) {
        // Jacobian wrt rotation:
        const Eigen::Vector3<T> cross =
            -sourceNormal.cross(constr.targetPosition - jointState.translation());
        const T diff_src = jointState.rotationAxis.col(d).dot(cross);
        const T diff_tgt = targetNormal.dot(jointState.getRotationDerivative(d, posd));

        jacobian_jointParams_to_modelParams<T>(
            boneWeight * wgt * (sourceNormalWeight * diff_src + targetNormalWeight * diff_tgt),
            paramIndex + d + 3,
            this->parameterTransform_,
            jac);
      }
    }
    if (this->activeJointParams_[paramIndex + 6]) {
      // Jacobian wrt scale:
      jacobian_jointParams_to_modelParams<T>(
          boneWeight * wgt * normal.dot(jointState.getScaleDerivative(posd)),
          paramIndex + 6,
          this->parameterTransform_,
          jac);
    }
  }
  // OUT handle derivatives wrt jointParameters

  // IN handle derivatives wrt blend shape parameters:
  for (Eigen::Index iBlendShape = 0;
       iBlendShape < this->parameterTransform_.blendShapeParameters.size();
       ++iBlendShape) {
    const auto paramIdx = this->parameterTransform_.blendShapeParameters[iBlendShape];
    if (paramIdx < 0) {
      continue;
    }

    const Eigen::Vector3<T> d_restPos =
        this->character_.blendShape->getShapeVectors()
            .template block<3, 1>(3 * constr.vertexIndex, iBlendShape, 3, 1)
            .template cast<T>();
    Eigen::Vector3<T> d_worldPos = Eigen::Vector3<T>::Zero();

    for (uint32_t i = 0; i < kMaxSkinJoints; ++i) {
      const auto w = skinWeights.weight(constr.vertexIndex, i);
      const auto parentBone = skinWeights.index(constr.vertexIndex, i);
      if (w > 0) {
        d_worldPos += w *
            (state.jointState[parentBone].transformation.linear() *
             (character_.inverseBindPose[parentBone].linear().template cast<T>() * d_restPos));
      }
    }

    jac(0, paramIdx) += wgt * d_worldPos.dot(normal);
  }
  // OUT handle derivatives wrt blend shape parameters:

  res = dist * wgt;

  return wgt * wgt * dist * dist;
}

template <typename T>
std::pair<T, T> VertexErrorFunctionT<T>::computeNormalWeights() const {
  switch (constraintType_) {
    case VertexConstraintType::Plane:
      return {T(0), T(1)};
    case VertexConstraintType::Normal:
      return {T(1), T(0)};
    case VertexConstraintType::SymmetricNormal:
    default:
      return {T(0.5), T(0.5)};
  }
}

template <typename T>
double VertexErrorFunctionT<T>::getGradient(
    const ModelParametersT<T>& modelParameters,
    const SkeletonStateT<T>& state,
    Eigen::Ref<Eigen::VectorX<T>> gradient) {
  updateMeshes(modelParameters, state);

  double error = 0;

  if (constraintType_ == VertexConstraintType::Position) {
    for (size_t iCons = 0; iCons < constraints_.size(); ++iCons) {
      error += calculatePositionGradient(modelParameters, state, constraints_[iCons], gradient);
    }
  } else {
    const auto [sourceNormalWeight, targetNormalWeight] = computeNormalWeights();

    for (size_t iCons = 0; iCons < constraints_.size(); ++iCons) {
      error += calculateNormalGradient(
          modelParameters,
          state,
          constraints_[iCons],
          sourceNormalWeight,
          targetNormalWeight,
          gradient);
    }
  }
  return this->weight_ * error;
}

template <typename T>
double VertexErrorFunctionT<T>::getJacobian(
    const ModelParametersT<T>& modelParameters,
    const SkeletonStateT<T>& state,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    Eigen::Ref<Eigen::VectorX<T>> residual,
    int& usedRows) {
  MT_CHECK(
      jacobian.cols() == static_cast<Eigen::Index>(this->parameterTransform_.transform.cols()));
  MT_CHECK(jacobian.rows() >= (Eigen::Index)(1 * constraints_.size()));
  MT_CHECK(residual.rows() >= (Eigen::Index)(1 * constraints_.size()));

  updateMeshes(modelParameters, state);

  double error = 0;
  if (constraintType_ == VertexConstraintType::Position) {
    MT_PROFILE_EVENT("VertexErrorFunction - position jacobians");

    for (size_t iCons = 0; iCons < constraints_.size(); ++iCons) {
      error += calculatePositionJacobian(
          modelParameters,
          state,
          constraints_[iCons],
          jacobian.block(3 * iCons, 0, 3, modelParameters.size()),
          residual.middleRows(3 * iCons, 3));
    }
    usedRows = 3 * constraints_.size();
  } else {
    MT_PROFILE_EVENT("VertexErrorFunction - normal jacobians");
    const auto [sourceNormalWeight, targetNormalWeight] = computeNormalWeights();

    for (size_t iCons = 0; iCons < constraints_.size(); ++iCons) {
      error += calculateNormalJacobian(
          modelParameters,
          state,
          constraints_[iCons],
          sourceNormalWeight,
          targetNormalWeight,
          jacobian.block(iCons, 0, 1, modelParameters.size()),
          residual(iCons));
    }
    usedRows = constraints_.size();
  }

  return error;
}

template <typename T>
size_t VertexErrorFunctionT<T>::getJacobianSize() const {
  switch (constraintType_) {
    case VertexConstraintType::Position:
      return 3 * constraints_.size();
    case VertexConstraintType::Normal:
    case VertexConstraintType::Plane:
    case VertexConstraintType::SymmetricNormal:
    default:
      return constraints_.size();
  }
}

template <typename T>
void VertexErrorFunctionT<T>::updateMeshes(
    const ModelParametersT<T>& modelParameters,
    const SkeletonStateT<T>& state) {
  MT_PROFILE_FUNCTION();

  bool doUpdateNormals = false;
  if (this->character_.blendShape) {
    const BlendWeightsT<T> blendWeights =
        extractBlendWeights(this->parameterTransform_, modelParameters);
    this->character_.blendShape->computeShape(blendWeights, this->restMesh_->vertices);
    doUpdateNormals = true;
  }
  if (this->character_.faceExpressionBlendShape) {
    if (!this->character_.blendShape) {
      // Set restMesh back to neutral, removing potential previous expressions.
      // Note that if the character comes with (shape) blendShape, the previous if block already
      // takes care of this step.
      Eigen::Map<Eigen::VectorX<T>> outputVec(
          &this->restMesh_->vertices[0][0], this->restMesh_->vertices.size() * 3);
      const Eigen::Map<Eigen::VectorX<T>> baseVec(
          &this->neutralMesh_->vertices[0][0], this->neutralMesh_->vertices.size() * 3);
      outputVec = baseVec.template cast<T>();
    }
    const BlendWeightsT<T> faceExpressionBlendWeights =
        extractFaceExpressionBlendWeights(this->parameterTransform_, modelParameters);
    this->character_.faceExpressionBlendShape->applyDeltas(
        faceExpressionBlendWeights, this->restMesh_->vertices);
    doUpdateNormals = true;
  }
  if (doUpdateNormals) {
    this->restMesh_->updateNormals();
  }

  applySSD(
      cast<T>(character_.inverseBindPose),
      *this->character_.skinWeights,
      *this->restMesh_,
      state,
      *this->posedMesh_);
  // TODO should we call updateNormals() here too or trust the ones from skinning?
}

template class VertexErrorFunctionT<float>;
template class VertexErrorFunctionT<double>;

} // namespace momentum
