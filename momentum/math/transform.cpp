/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/math/transform.h"

namespace momentum {

template <typename T>
TransformT<T> TransformT<T>::makeRotation(const Eigen::Quaternion<T>& rotation_in) {
  return TransformT<T>(Eigen::Vector3<T>::Zero(), rotation_in, T(1));
}

template <typename T>
TransformT<T> TransformT<T>::makeTranslation(const Eigen::Vector3<T>& translation_in) {
  return TransformT<T>(translation_in, Eigen::Quaternion<T>::Identity(), T(1));
}

template <typename T>
TransformT<T> TransformT<T>::makeScale(
    const T scale_in) // NOLINT(facebook-hte-ConstantArgumentPassByValue)
{
  return TransformT<T>(Eigen::Vector3<T>::Zero(), Eigen::Quaternion<T>::Identity(), scale_in);
}

template <typename T>
typename TransformT<T>::Affine3 TransformT<T>::matrix() const {
  Affine3 xf = Affine3::Identity();
  xf.linear() = scale * rotation.toRotationMatrix();
  xf.translation() = translation;
  return xf;
}

template <typename T>
Eigen::Vector3<T> TransformT<T>::transformPoint(const Eigen::Vector3<T>& pt) const {
  return translation + rotation * (scale * pt).eval();
}

template <typename T>
Eigen::Vector3<T> TransformT<T>::rotate(const Eigen::Vector3<T>& vec) const {
  return rotation * vec;
}

template <typename T>
TransformT<T> TransformT<T>::inverse() const {
  // (translate(t) * rotation(R) * scale(s)).inv() =
  //     scale(s).inv() * rotation(R).inv() * translate(t).inv()
  //   = scale(1/s) * rotation(invR) * translate(-t)
  //   = translate(-invR*s*t) * rotation(invR) * scale(1/s)
  const Eigen::Quaternion<T> invRot = rotation.inverse();
  const double invScale = T(1) / scale;
  return TransformT<T>(-invScale * (invRot * translation), invRot, invScale);
}

template <typename T>
TransformT<T> operator*(const TransformT<T>& lhs, const TransformT<T>& rhs) {
  // [ s_1*R_1 t_1 ] * [ s_2*R_2 t_2 ] = [ s_1*s_2*R_1*R_2  s_1*R_1*t_2 + t_1 ]
  // [     0    1  ]   [     0    1  ]   [        0                     1     ]
  const Eigen::Vector3<T> trans =
      lhs.translation + lhs.rotation * (lhs.scale * rhs.translation).eval();
  const Eigen::Quaternion<T> rot = lhs.rotation * rhs.rotation;
  const T scale = lhs.scale * rhs.scale;
  return TransformT<T>(trans, rot, scale);
}

template struct TransformT<float>;
template struct TransformT<double>;

template TransformT<float> operator*(const TransformT<float>& lhs, const TransformT<float>& rhs);
template TransformT<double> operator*(const TransformT<double>& lhs, const TransformT<double>& rhs);

} // namespace momentum
