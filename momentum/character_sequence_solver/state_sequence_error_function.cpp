/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character_sequence_solver/state_sequence_error_function.h"

#include "momentum/character/character.h"
#include "momentum/character/skeleton.h"
#include "momentum/common/checks.h"
#include "momentum/common/profile.h"
#include "momentum/math/utility.h"

namespace momentum {

template <typename T>
StateSequenceErrorFunctionT<T>::StateSequenceErrorFunctionT(
    const Skeleton& skel,
    const ParameterTransform& pt)
    : SequenceErrorFunctionT<T>(skel, pt) {
  targetPositionWeights_.setOnes(this->skeleton_.joints.size());
  targetRotationWeights_.setOnes(this->skeleton_.joints.size());
  posWgt_ = T(1);
  rotWgt_ = T(1);
}

template <typename T>
StateSequenceErrorFunctionT<T>::StateSequenceErrorFunctionT(const Character& character)
    : StateSequenceErrorFunctionT<T>(character.skeleton, character.parameterTransform) {}

template <typename T>
void StateSequenceErrorFunctionT<T>::reset() {
  targetPositionWeights_.setOnes(this->skeleton_.joints.size());
  targetRotationWeights_.setOnes(this->skeleton_.joints.size());
}

template <typename T>
void StateSequenceErrorFunctionT<T>::setTargetWeights(
    const Eigen::VectorX<T>& posWeight,
    const Eigen::VectorX<T>& rotWeight) {
  MT_CHECK(posWeight.size() == static_cast<Eigen::Index>(this->skeleton_.joints.size()));
  MT_CHECK(rotWeight.size() == static_cast<Eigen::Index>(this->skeleton_.joints.size()));

  targetPositionWeights_ = posWeight;
  targetRotationWeights_ = rotWeight;
}

template <typename T>
double StateSequenceErrorFunctionT<T>::getError(
    gsl::span<const ModelParametersT<T>> /* modelParameters */,
    gsl::span<const SkeletonStateT<T>> skelStates) const {
  MT_PROFILE_EVENT("StateSequenceError: getError");

  // loop over all joints and check for smoothness
  double error = 0.0;

  MT_CHECK(skelStates.size() == 2);

  const auto& prevState = skelStates[0];
  const auto& nextState = skelStates[1];

  // go over the joint states and calculate the rotation difference contribution on joints that are
  // at the end
  for (size_t i = 0; i < this->skeleton_.joints.size(); ++i) {
    // calculate orientation error
    // Note: |R0 - RT|² is a valid norm on SO3, it doesn't have the same slope as the squared angle
    // difference
    //       but it's derivative doesn't have a singularity at the minimum, so is more stable
    const Eigen::Quaternion<T>& prevRot = prevState.jointState[i].rotation;
    const Eigen::Quaternion<T>& nextRot = nextState.jointState[i].rotation;
    const Eigen::Matrix3<T> rotDiff = nextRot.toRotationMatrix() - prevRot.toRotationMatrix();
    error += rotDiff.squaredNorm() * kOrientationWeight * rotWgt_ * targetRotationWeights_[i];

    // calculate position error
    const Eigen::Vector3<T> diff =
        nextState.jointState[i].translation - prevState.jointState[i].translation;
    error += diff.squaredNorm() * kPositionWeight * posWgt_ * targetPositionWeights_[i];
  }

  // return error
  return error * this->weight_;
}

template <typename T>
double StateSequenceErrorFunctionT<T>::getGradient(
    gsl::span<const ModelParametersT<T>> /* modelParameters */,
    gsl::span<const SkeletonStateT<T>> skelStates,
    Eigen::Ref<Eigen::VectorX<T>> gradient) const {
  MT_PROFILE_EVENT("StateSequenceError: getGradient");

  MT_CHECK(skelStates.size() == 2);
  const auto& prevState = skelStates[0];
  const auto& nextState = skelStates[1];

  const Eigen::Index nParam = this->parameterTransform_.numAllModelParameters();
  MT_CHECK(gradient.size() == 2 * nParam);

  // loop over all joints and check for smoothness
  double error = 0.0;

  // also go over the joint states and calculate the rotation difference contribution on joints that
  // are at the end
  for (size_t i = 0; i < this->skeleton_.joints.size(); ++i) {
    if (targetRotationWeights_[i] == 0 && targetPositionWeights_[i] == 0) {
      continue;
    }

    // calculate orientation gradient
    const Eigen::Quaternion<T>& prevRot = prevState.jointState[i].rotation;
    const Eigen::Quaternion<T>& nextRot = nextState.jointState[i].rotation;
    const Eigen::Matrix3<T> rotDiff = nextRot.toRotationMatrix() - prevRot.toRotationMatrix();
    const T rwgt = kOrientationWeight * rotWgt_ * this->weight_ * targetRotationWeights_[i];
    error += rotDiff.squaredNorm() * rwgt;

    // calculate position gradient
    const Eigen::Vector3<T> diff =
        nextState.jointState[i].translation - prevState.jointState[i].translation;
    const T pwgt = kPositionWeight * posWgt_ * this->weight_ * targetPositionWeights_[i];
    error += diff.squaredNorm() * pwgt;

    auto addGradient = [&](const SkeletonStateT<T>& state,
                           T sign,
                           Eigen::Ref<Eigen::VectorX<T>> grad) {
      const Eigen::Quaternion<T>& rot = state.jointState[i].rotation;

      // loop over all joints the constraint is attached to and calculate gradient
      size_t jointIndex = i;
      while (jointIndex != kInvalidIndex) {
        // check for valid index
        MT_CHECK(jointIndex < this->skeleton_.joints.size());

        const size_t paramIndex = jointIndex * kParametersPerJoint;

        // precalculate some more data for position gradient
        const Eigen::Vector3<T> posd =
            state.jointState[i].translation - state.jointState[jointIndex].translation;

        for (size_t d = 0; d < 3; d++) {
          // position gradient
          if (this->activeJointParams_[paramIndex + d]) {
            // calculate joint gradient
            const T val = sign * T(2) *
                diff.dot(state.jointState[jointIndex].getTranslationDerivative(d)) * pwgt;
            // explicitly multiply with the parameter transform to generate parameter space
            // gradients
            for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d];
                 index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 1];
                 ++index) {
              grad[this->parameterTransform_.transform.innerIndexPtr()[index]] +=
                  val * this->parameterTransform_.transform.valuePtr()[index];
            }
          }
          if (this->activeJointParams_[paramIndex + 3 + d]) {
            // calculate joint gradient consisting of position gradient and orientation gradient
            const Eigen::Vector3<T> axis = state.jointState[jointIndex].rotationAxis.col(d);
            const auto rotD = crossProductMatrix(axis) * rot;
            const T val = sign * T(2) *
                (diff.dot(state.jointState[jointIndex].getRotationDerivative(d, posd)) * pwgt +
                 rwgt * rotD.cwiseProduct(rotDiff).sum());
            // explicitly multiply with the parameter transform to generate parameter space
            // gradients
            for (auto index =
                     this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d];
                 index <
                 this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 3 + d + 1];
                 ++index) {
              grad[this->parameterTransform_.transform.innerIndexPtr()[index]] +=
                  val * this->parameterTransform_.transform.valuePtr()[index];
            }
          }
        }
        if (this->activeJointParams_[paramIndex + 6]) {
          // calculate joint gradient
          const T val =
              sign * T(2) * diff.dot(state.jointState[jointIndex].getScaleDerivative(posd)) * pwgt;
          // explicitly multiply with the parameter transform to generate parameter space gradients
          for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 6];
               index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 6 + 1];
               ++index) {
            grad[this->parameterTransform_.transform.innerIndexPtr()[index]] +=
                val * this->parameterTransform_.transform.valuePtr()[index];
          }
        }

        // go to the next joint
        jointIndex = this->skeleton_.joints[jointIndex].parent;
      }
    };

    addGradient(prevState, T(-1), gradient.segment(0, nParam));
    addGradient(nextState, T(1), gradient.segment(nParam, nParam));
  }

  // return error
  return error;
}

template <typename T>
size_t StateSequenceErrorFunctionT<T>::getJacobianSize() const {
  auto result =
      (targetPositionWeights_.array() != 0 || targetRotationWeights_.array() != 0).count() * 12;
  return result;
}

template <typename T>
double StateSequenceErrorFunctionT<T>::getJacobian(
    gsl::span<const ModelParametersT<T>> /* modelParameters */,
    gsl::span<const SkeletonStateT<T>> skelStates,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian_full,
    Eigen::Ref<Eigen::VectorX<T>> residual_full,
    int& usedRows) const {
  MT_PROFILE_EVENT("StateSequenceError: getJacobian");

  MT_CHECK(skelStates.size() == 2);
  const auto& prevState = skelStates[0];
  const auto& nextState = skelStates[1];

  const Eigen::Index nParam = this->parameterTransform_.numAllModelParameters();
  MT_CHECK(jacobian_full.cols() == 2 * nParam);

  double error = 0.0;

  Eigen::Index offset = 0;

  // calculate state difference jacobians
  for (size_t i = 0; i < this->skeleton_.joints.size(); ++i) {
    if (targetRotationWeights_[i] == 0 && targetPositionWeights_[i] == 0) {
      continue;
    }

    // calculate translation gradient
    const auto transDiff =
        (nextState.jointState[i].translation - prevState.jointState[i].translation).eval();

    // calculate orientation gradient
    const Eigen::Quaternion<T>& prevRot = prevState.jointState[i].rotation;
    const Eigen::Quaternion<T>& nextRot = nextState.jointState[i].rotation;
    const Eigen::Matrix3<T> rotDiff = nextRot.toRotationMatrix() - prevRot.toRotationMatrix();
    const T rwgt = kOrientationWeight * rotWgt_ * this->weight_ * targetRotationWeights_[i];
    error += rotDiff.squaredNorm() * rwgt;
    const T awgt = std::sqrt(rwgt);

    // calculate the difference between target and position and error
    const T pwgt = kPositionWeight * posWgt_ * this->weight_ * targetPositionWeights_[i];
    const T wgt = std::sqrt(pwgt);
    error += transDiff.squaredNorm() * pwgt;

    // update the residue
    Ref<Eigen::VectorX<T>> res = residual_full.middleRows(offset, 12);
    res.template topRows<3>() = transDiff * wgt;
    res.template bottomRows<9>() =
        Map<const Eigen::VectorX<T>>(rotDiff.data(), rotDiff.size()) * awgt;

    auto addJacobian = [&](const SkeletonStateT<T>& state, T sign, Ref<Eigen::MatrixX<T>> jac) {
      const Eigen::Quaternion<T>& rot = state.jointState[i].rotation;

      // loop over all joints the constraint is attached to and calculate jacobian
      size_t jointIndex = i;
      while (jointIndex != kInvalidIndex) {
        // check for valid index
        MT_CHECK(jointIndex < this->skeleton_.joints.size());

        const auto& jointState = state.jointState[jointIndex];
        const size_t paramIndex = jointIndex * kParametersPerJoint;

        // precalculate some more data for position gradient
        const Eigen::Vector3<T> posd = state.jointState[i].translation - jointState.translation;

        // calculate derivatives based on active joints
        for (size_t d = 0; d < 3; d++) {
          if (this->activeJointParams_[paramIndex + d]) {
            const Eigen::Vector3<T> jc = sign * jointState.getTranslationDerivative(d) * wgt;
            for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d];
                 index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 1];
                 ++index) {
              jac.col(this->parameterTransform_.transform.innerIndexPtr()[index])
                  .template topRows<3>()
                  .noalias() += jc * this->parameterTransform_.transform.valuePtr()[index];
            }
          }

          if (this->activeJointParams_[paramIndex + 3 + d]) {
            const Eigen::Vector3<T> jc = sign * jointState.getRotationDerivative(d, posd) * wgt;

            const Eigen::Vector3<T> axis = state.jointState[jointIndex].rotationAxis.col(d);
            const Eigen::Matrix3<T> rotD = sign * crossProductMatrix(axis) * rot * awgt;
            const auto ja = Map<const Eigen::VectorX<T>>(rotD.data(), rotD.size());

            for (auto index =
                     this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 3];
                 index <
                 this->parameterTransform_.transform.outerIndexPtr()[paramIndex + d + 3 + 1];
                 ++index) {
              jac.col(this->parameterTransform_.transform.innerIndexPtr()[index])
                  .template topRows<3>()
                  .noalias() += jc * this->parameterTransform_.transform.valuePtr()[index];
              jac.col(this->parameterTransform_.transform.innerIndexPtr()[index])
                  .template bottomRows<9>()
                  .noalias() += ja * this->parameterTransform_.transform.valuePtr()[index];
            }
          }
        }

        if (this->activeJointParams_[paramIndex + 6]) {
          const Eigen::Vector3<T> jc = sign * jointState.getScaleDerivative(posd) * wgt;
          for (auto index = this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 6];
               index < this->parameterTransform_.transform.outerIndexPtr()[paramIndex + 6 + 1];
               ++index) {
            jac.col(this->parameterTransform_.transform.innerIndexPtr()[index])
                .template topRows<3>()
                .noalias() += jc * this->parameterTransform_.transform.valuePtr()[index];
          }
        }

        // go to the next joint
        jointIndex = this->skeleton_.joints[jointIndex].parent;
      }
    };

    addJacobian(prevState, T(-1), jacobian_full.block(offset, 0, 12, nParam));
    addJacobian(nextState, T(1), jacobian_full.block(offset, nParam, 12, nParam));
    offset += 12;
  }

  usedRows = gsl::narrow_cast<int>(offset);

  // return error
  return error;
}

template class StateSequenceErrorFunctionT<float>;
template class StateSequenceErrorFunctionT<double>;

} // namespace momentum
