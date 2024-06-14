/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/motion/joint_params_binary_io.h"

#include <fstream>

namespace momentum {

std::vector<JointParameters> loadJointParamsBinary(const std::string& filename) {
  std::ifstream jpFile(filename, std::ios::in | std::ios::binary);
  int64_t numFrames;
  int64_t numParams;

  // first two parameters (int64) are number of frames and size of jointParameters
  jpFile.read(reinterpret_cast<char*>(&numFrames), sizeof(numFrames));
  jpFile.read(reinterpret_cast<char*>(&numParams), sizeof(numParams));

  std::vector<JointParameters> jp =
      std::vector<JointParameters>(numFrames, JointParameters::Zero(numParams));

  // joint parameters are stored as floats
  for (int frame_i = 0; frame_i < numFrames; ++frame_i) {
    JointParameters jpFrame(numParams);
    jpFile.read((char*)jpFrame.v.data(), numParams * sizeof(float));
    jp[frame_i] = jpFrame;
  }

  return jp;
}

} // namespace momentum
