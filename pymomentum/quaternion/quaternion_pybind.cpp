/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/tensor_momentum/tensor_quaternion.h"

#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/python.h>

namespace pymomentum {

PYBIND11_MODULE(quaternion, m) {
  // TODO more explanation
  m.doc() = "Handling of quaternions.  ";
  m.attr("__name__") = "pymomentum.quaternion";

  pybind11::module_::import("torch"); // @dep=//caffe2:torch

  m.def(
      "identity",
      &quaternionIdentity,
      R"(:return: the identity quaternion (0, 0, 0, 1))");

  m.def(
      "multiply",
      &quaternionMultiply,
      R"(Multiplies two quaternions together.

:param q1: A quaternion ((x, y, z), w)).
:param q2: A quaternion ((x, y, z), w)).
:return: The product q1*q2.
)",
      py::arg("q1"),
      py::arg("q2"));

  m.def(
      "inverse",
      &quaternionInverse,
      R"(Compute the inverse of a quaternion.

:param q: A quaternion ((x, y, z), w)).
:return: The inverse.
)",
      py::arg("q"));

  m.def(
      "conjugate",
      &quaternionConjugate,
      R"(Conjugate a quaternion.

:param q: A quaternion ((x, y, z), w)).
:return: The conjugate.
)",
      py::arg("q"));

  m.def(
      "normalize",
      &quaternionNormalize,
      R"(Normalize a quaternion.

:param q: A quaternion ((x, y, z), w)).
:return: The normalized quaternion.
)",
      py::arg("q"));

  m.def(
      "euler_xyz_to_quaternion",
      &xyzEulerToQuaternion,
      R"(Convert XYZ Euler rotations to quaternions.

:param xyzEuler: (nBatch x k x 3) tensor with the 3-dof xyz Euler rotations.
:return: A (nBatch x k x 4) tensor containing quaternions in the ((x, y, z), w) format.)",
      py::arg("xyz_euler"));

  m.def(
      "quaternion_to_xyz_euler",
      &quaternionToXYZEuler,
      R"(Convert quaternions to XYZ Euler rotations.

:param quat: (nBatch x k x 4) tensor with the quaternions in ((x, y, z), w) format.
:return: A (nBatch x k x 3) tensor containing (x, y, z) Euler angles.)",
      py::arg("xyz_euler"));

  m.def(
      "to_rotation_matrix",
      &quaternionToRotationMatrix,
      R"(Convert quaternions to 3x3 rotation matrices.

:param q: (nBatch x k x 4) tensor with the quaternions in ((x, y, z), w) format.
:return: (nBatch x k x 3 x 3) tensor with 3x3 rotation matrices.)",
      py::arg("q"));

  m.def(
      "from_rotation_matrix",
      &rotationMatrixToQuaternion,
      R"(Convert 3x3 rotation matrices to quaternions.

:param matrices: (nBatch x k x 4) tensor with the quaternions in ((x, y, z), w) format.
:return: (nBatch x k x 3 x 3) tensor with 3x3 rotation matrices.)",
      py::arg("matrices"));

  m.def(
      "rotate_vector",
      &quaternionRotateVector,
      R"(Rotate a vector by a quaternion.

:param q: (nBatch x k x 4) tensor with the quaternions in ((x, y, z), w) format.
:param v: (nBatch x k x 3) vector.
:return: (nBatch x k x 3) rotated vectors.)",
      py::arg("q"),
      py::arg("v"));

  m.def(
      "blend",
      &blendQuaternions,
      R"(Blend k quaternions with the passed-in weights.

:param quaternions: (nBatch x k x 4) tensor with the quaternions in ((x, y, z), w) format.
:param weights: Optional (nBatch x k) tensor.
:return: (nBatch x 4) blended quaternion.)",
      py::arg("quaternions"),
      py::arg("weights") = std::optional<at::Tensor>{});
}

} // namespace pymomentum
