/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/tensor_momentum/tensor_skeleton_state.h"
#include "pymomentum/tensor_momentum/tensor_transforms.h"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/python.h>
#include <Eigen/Core>

namespace pymomentum {

PYBIND11_MODULE(skel_state, m) {
  // TODO more explanation
  m.doc() =
      R"(Handling of skeleton states, including converting to and from transform matrices.

A skeleton state is an (n x 8) tensor representing an array of transforms, where each transform is represented as a translation,
rotation, and (uniform) scale.  The 8 components are: (tx, ty, tz, rx, ry, rz, rw, s).

Skeleton states are used extensively in momentum to represent transforms from joint-local space to world space, hence the name.
They are convenient because it is easy to separate out the different components of the transform; a matrix representation would
require using the polar decomposition to split scale from rotation.
       )";
  m.attr("__name__") = "pymomentum.skel_state";

  pybind11::module_::import("torch"); // @dep=//caffe2:torch

  m.def(
      "to_matrix",
      &skeletonStateToTransforms,
      R"(Convert skeleton state to a tensor of 4x4 matrices. The matrix represents the transform from a local joint space to the world space.

:param skeletonState: (nBatch x nJoints x 8) tensor with the skeleton state.
:return: A (nBatch x nJoints x 4 x 4) tensor containing 4x4 matrix transforms.)",
      py::arg("skeleton_state"));

  m.def(
      "from_matrix",
      &matricesToSkeletonStates,
      R"(Convert 4x4 matrices to skeleton states. The matrix represents the transform from a local joint space to the world space.

:param matrix: (nBatch x nJoints x 4 x 4) tensor with the transforms.
:return: A (nBatch x nJoints x 8) tensor containing the skeleton states.)",
      py::arg("skeleton_state"));

  m.def(
      "multiply",
      &multiplySkeletonStates,
      R"(Multiply two skeleton states.

:param s1: (nBatch x nJoints x 8) tensor with the skeleton states.
:param s2: (nBatch x nJoints x 8) tensor with the skeleton states.
:return: A (nBatch x nJoints x 8) tensor containing the (x, y, z, rx, ry, rz, rw, s) skeleton states corresponding to the product.)",
      py::arg("s1"),
      py::arg("s2"));

  m.def(
      "inverse",
      &inverseSkeletonStates,
      R"(Compute the inverse of a skeleton state.

:param skeleton_states: (nBatch x nJoints x 8) tensor with the skeleton states.
:return: A (nBatch x nJoints x 8) tensor containing the (x, y, z, rx, ry, rz, rw, s) skeleton states corresponding to the inverse transform.)",
      py::arg("skeleton_states"));

  m.def(
      "transform_points",
      &transformPointsWithSkeletonState,
      R"(Transform 3d points by the transform represented by the skeleton state.

:param skeleton_states: (nBatch x nJoints x 8) tensor with the skeleton states.
:param points: (nBatch x nJoints x 3) tensor with the points.
:return: A (nBatch x nJoints x 3) tensor containing the points transformed by the skeleton states.)",
      py::arg("skeleton_states"),
      py::arg("points"));

  m.def(
      "from_quaternion",
      &quaternionToSkeletonState,
      R"(Convert quaternions to a skeleton state.

:param q: (nBatch x 4) tensor with the quaternion rotations in the form ((x, y, z), w).
:return: A (nBatch x 8) tensor containing skeleton states (tx, ty, tz, rx, ry, rz, rw, s).)",
      py::arg("skeleton_state"));

  m.def(
      "from_translation",
      &translationToSkeletonState,
      R"(Convert translations to a skeleton state.

:param q: (nBatch x 3) tensor with the translations (x, y, z).
:return: A (nBatch x 8) tensor containing skeleton states (tx, ty, tz, rx, ry, rz, rw, s).)",
      py::arg("skeleton_state"));

  m.def(
      "from_scale",
      &scaleToSkeletonState,
      R"(Convert scales to a skeleton state.

:param q: (nBatch x 1) tensor with uniform scales.
:return: A (nBatch x 8) tensor containing skeleton states (tx, ty, tz, rx, ry, rz, rw, s).)",
      py::arg("skeleton_state"));

  m.def(
      "blend",
      &blendSkeletonStates,
      R"(Blend k skeleton states with the passed-in weights.

:param skeleton_states: (nBatch x k x 8) tensor containing skeleton states (tx, ty, tz, rx, ry, rz, rw, s)..
:param weights: Optional (nBatch x k) tensor.
:return: (nBatch x 4) blended quaternion.)",
      py::arg("skeleton_states"),
      py::arg("weights") = std::optional<at::Tensor>{});

  m.def(
      "identity",
      &identitySkeletonState,
      R"(Returns a skeleton state representing the identity transform.)");

  m.def(
      "split",
      &splitSkeletonState,
      R"(Splits a skeleton state into translation, rotation, and scale components.

:param skeleton_state: (nBatch x 8)-dimension tensor with skeletons states.
:return: A tuple of tensors (translation, rotation, scale).
      )",
      py::arg("skeleton_state"));
}

} // namespace pymomentum
