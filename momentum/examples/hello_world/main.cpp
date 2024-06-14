/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <momentum/math/mesh.h>

using namespace momentum;

int main() {
  auto mesh = Mesh();
  mesh.updateNormals();
  return EXIT_SUCCESS;
}
