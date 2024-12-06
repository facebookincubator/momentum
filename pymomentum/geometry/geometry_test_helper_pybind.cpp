/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/test/character/character_helpers.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;
namespace mm = momentum;

PYBIND11_MODULE(geometry_test_helper, m) {
  m.doc() = "Geometry test helper.  ";
  m.attr("__name__") = "pymomentum.geometry_test_helper";

  pybind11::module_::import(
      "pymomentum.geometry"); // @dep=fbcode//pymomentum:geometry

  // createTestCharacter()
  m.def(
      "test_character",
      &momentum::createTestCharacter<float>,
      R"(Create a simple 3-joint test character.  This is useful for writing confidence tests that
execute quickly and don't rely on outside files.

The mesh is made by a few vertices on the line segment from (1,0,0) to (1,1,0) and a few dummy
faces. The skeleton has three joints: root at (0,0,0), joint1 parented by root, at world-space
(0,1,0), and joint2 parented by joint1, at world-space (0,2,0).
The character has only one parameter limit: min-max type [-0.1, 0.1] for root.

:parameter numJoints: The number of joints in the resulting character.
:return: A simple character with 3 joints and 10 model parameters.
      )",
      py::arg("num_joints") = 3);

  // createTestPosePrior()
  m.def(
      "create_test_mppca",
      &momentum::createDefaultPosePrior<float>,
      R"(Create a pose prior that acts on the simple 3-joint test character.

:return: A simple pose prior.)");
}
