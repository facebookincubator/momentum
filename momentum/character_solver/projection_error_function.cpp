/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/character_solver/projection_error_function.h>

#include <momentum/character/skeleton.h>
#include <momentum/character/skeleton_state.h>

namespace momentum {

template <typename T>
ProjectionErrorFunctionT<T>::ProjectionErrorFunctionT(
    const momentum::Skeleton& skel,
    const momentum::ParameterTransform& pt,
    T nearClip)
    : SkeletonErrorFunctionT<T>(skel, pt), _nearClip(nearClip) {}

template <typename T>
double ProjectionErrorFunctionT<T>::getError(
    const momentum::ModelParametersT<T>& params,
    const momentum::SkeletonStateT<T>& skeletonState) {
  double error = 0;

  const auto& jointState = skeletonState.jointState;

  const auto nCons = _constraints.size();
  for (size_t iCons = 0; iCons < nCons; ++iCons) {
    const auto& cons = _constraints[iCons];
    const auto& js = jointState[cons.parent];
    const Eigen::Vector3<T> p_world_cm = js.transformation * cons.offset;
    const Eigen::Vector3<T> p_projected_cm =
        _constraints[iCons].projection * p_world_cm.homogeneous();

    // Behind camera:
    if (p_projected_cm.z() < _nearClip) {
      continue;
    }

    const Eigen::Vector2<T> diff = p_projected_cm.hnormalized().template head<2>() - cons.target;
    error += cons.weight * kProjectionWeight * this->weight_ * diff.squaredNorm();
  }

  return error;
}

template <typename T>
Eigen::Matrix<T, 3, 4> makeProjectionRadians(
    const Affine3<T>& worldToCamera,
    const Eigen::Vector3<T>& targetDirection) {
  // find the eye-space ray corresponding to the target point, then set up
  // our matrix as an isometry which carries the target ray to the z axis.
  //
  // This gives us units of the resulting constraint in tan(angle), which is
  // just radians in the neighborhood of the target ray.
  //
  // Since we're using a rotation rather than a shear, the resulting transform
  // is better-behaved for target points near the corners of the camera frame.
  // This code copied from FullProjectionConstraint.h:

  Eigen::Matrix<T, 3, 4> result =
      Eigen::Quaternion<T>::FromTwoVectors(targetDirection, Eigen::Vector3<T>::UnitZ())
          .toRotationMatrix() *
      worldToCamera.matrix().template topRows<3>();

  return result;
}

template <typename T>
void ProjectionErrorFunctionT<T>::addConstraint(const ProjectionConstraintDataT<T>& data) {
  _constraints.push_back(data);
}

template <typename T>
double ProjectionErrorFunctionT<T>::getGradient(
    const momentum::ModelParametersT<T>& params,
    const momentum::SkeletonStateT<T>& skeletonState,
    Eigen::Ref<Eigen::VectorX<T>> gradient) {
  double error = 0;

  for (size_t iCons = 0; iCons < _constraints.size(); ++iCons) {
    const auto& cons = _constraints[iCons];

    const auto& jsCons = skeletonState.jointState[cons.parent];
    const Eigen::Vector3<T> p_world_cm = jsCons.transformation * cons.offset;
    const Eigen::Vector3<T> p_projected_cm = cons.projection * p_world_cm.homogeneous();

    // Behind camera:
    if (p_projected_cm.z() < _nearClip) {
      continue;
    }

    const Eigen::Vector2<T> p_res = p_projected_cm.hnormalized() - cons.target;

    error += cons.weight * kProjectionWeight * this->weight_ * p_res.squaredNorm();
    const T wgt = 2.0f * cons.weight * kProjectionWeight * this->weight_;

    const T z = p_projected_cm(2);
    const T z_sqr = sqr(z);
    const T x_zz = p_projected_cm(0) / z_sqr;
    const T y_zz = p_projected_cm(1) / z_sqr;

    auto addGradient = [&](const size_t jFullBodyDOF, const Eigen::Vector3<T> d_p_world_cm) {
      const Eigen::Vector3<T> d_p_projected =
          cons.projection.template topLeftCorner<3, 3>() * d_p_world_cm;
      const T dx = d_p_projected(0);
      const T dy = d_p_projected(1);
      const T dz = d_p_projected(2);

      const Eigen::Vector2<T> d_p_res(dx / z - x_zz * dz, dy / z - y_zz * dz);
      const T gradFull = d_p_res.dot(p_res);

      // multiply by the parameter transform:
      for (momentum::SparseRowMatrixf::InnerIterator it(
               this->parameterTransform_.transform, (int)jFullBodyDOF);
           it;
           ++it) {
        assert(static_cast<size_t>(it.row()) == jFullBodyDOF);
        const auto jReducedDOF = it.col();
        gradient(jReducedDOF) += it.value() * wgt * gradFull;
      }
    };

    // loop over all joints the constraint is attached to and calculate gradient
    size_t jointIndex = cons.parent;
    while (jointIndex != kInvalidIndex) {
      // check for valid index
      assert(jointIndex < this->skeleton_.joints.size());

      const auto& jointState = skeletonState.jointState[jointIndex];
      const size_t paramIndex = jointIndex * 7;
      const Eigen::Vector3<T> posd_cm = p_world_cm - jointState.translation();

      // calculate derivatives based on active joints
      for (size_t d = 0; d < 3; d++) {
        if (this->activeJointParams_[paramIndex + d] > 0) {
          addGradient(paramIndex + d, jointState.getTranslationDerivative(d));
        }
        if (this->activeJointParams_[paramIndex + 3 + d] > 0) {
          addGradient(paramIndex + 3 + d, jointState.getRotationDerivative(d, posd_cm));
        }
      }
      if (this->activeJointParams_[paramIndex + 6] > 0) {
        addGradient(paramIndex + 6, jointState.getScaleDerivative(posd_cm));
      }

      // go to the next joint
      jointIndex = this->skeleton_.joints[jointIndex].parent;
    }
  }

  return error;
}

template <typename T>
double ProjectionErrorFunctionT<T>::getJacobian(
    const momentum::ModelParametersT<T>& params,
    const momentum::SkeletonStateT<T>& skeletonState,
    Eigen::Ref<Eigen::MatrixX<T>> jacobian,
    Eigen::Ref<Eigen::VectorX<T>> residual,
    int& usedRows) {
  double error = 0;

  for (size_t iCons = 0; iCons < _constraints.size(); ++iCons) {
    const auto& cons = _constraints[iCons];

    const auto& jsCons = skeletonState.jointState[cons.parent];
    const Eigen::Vector3<T> p_world_cm = jsCons.transformation * cons.offset;
    const Eigen::Vector3<T> p_projected_cm = cons.projection * p_world_cm.homogeneous();

    // Behind camera:
    if (p_projected_cm.z() < _nearClip) {
      continue;
    }

    const Eigen::Vector2<T> p_res = p_projected_cm.hnormalized() - cons.target;
    error += cons.weight * kProjectionWeight * this->weight_ * p_res.squaredNorm();

    const T z = p_projected_cm(2);
    const T z_sqr = sqr(z);
    const T x_zz = p_projected_cm(0) / z_sqr;
    const T y_zz = p_projected_cm(1) / z_sqr;

    const T wgt = std::sqrt(cons.weight * kProjectionWeight * this->weight_);
    residual.template segment<2>(2 * iCons) = (wgt * p_res).template cast<T>();

    auto addJacobian = [&](const size_t jFullBodyDOF, const Eigen::Vector3<T> d_p_world_cm) {
      const Eigen::Vector3<T> d_p_projected =
          cons.projection.template topLeftCorner<3, 3>() * d_p_world_cm;
      const T dx = d_p_projected(0);
      const T dy = d_p_projected(1);
      const T dz = d_p_projected(2);

      const Eigen::Vector2<T> d_p_res(dx / z - x_zz * dz, dy / z - y_zz * dz);

      // multiply by the parameter transform:
      for (momentum::SparseRowMatrixf::InnerIterator it(
               this->parameterTransform_.transform, (int)jFullBodyDOF);
           it;
           ++it) {
        assert(static_cast<size_t>(it.row()) == jFullBodyDOF);
        const auto jReducedDOF = it.col();
        jacobian.template block<2, 1>(2 * iCons, jReducedDOF) += it.value() * wgt * d_p_res;
      }
    };

    // loop over all joints the constraint is attached to and calculate gradient
    size_t jointIndex = cons.parent;
    while (jointIndex != kInvalidIndex) {
      // check for valid index
      assert(jointIndex < this->skeleton_.joints.size());

      const auto& jointState = skeletonState.jointState[jointIndex];
      const size_t paramIndex = jointIndex * 7;
      const Eigen::Vector3<T> posd_cm = p_world_cm - jointState.translation();

      // calculate derivatives based on active joints
      for (size_t d = 0; d < 3; d++) {
        if (this->activeJointParams_[paramIndex + d] > 0) {
          addJacobian(paramIndex + d, jointState.getTranslationDerivative(d));
        }
        if (this->activeJointParams_[paramIndex + 3 + d] > 0) {
          addJacobian(paramIndex + 3 + d, jointState.getRotationDerivative(d, posd_cm));
        }
      }
      if (this->activeJointParams_[paramIndex + 6] > 0) {
        addJacobian(paramIndex + 6, jointState.getScaleDerivative(posd_cm));
      }

      // go to the next joint
      jointIndex = this->skeleton_.joints[jointIndex].parent;
    }
  }

  usedRows = static_cast<int>(jacobian.rows());

  return error;
}

template <typename T>
size_t ProjectionErrorFunctionT<T>::getJacobianSize() const {
  return 2 * _constraints.size();
}

template class ProjectionErrorFunctionT<float>;
template class ProjectionErrorFunctionT<double>;

} // namespace momentum
