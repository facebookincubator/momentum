/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/skeleton/parameter_limits_io.h"

#include "momentum/character/parameter_transform.h"
#include "momentum/character/skeleton.h"
#include "momentum/common/log.h"
#include "momentum/math/constants.h"
#include "momentum/math/utility.h"

#include <re2/re2.h>

namespace momentum {

namespace {

void parseMinmaxWithParameterIndex(
    ParameterLimits& pl,
    const std::string& valueStr,
    int parameterIndex) {
  // create new parameterlimit
  ParameterLimit p;
  p.weight = 1.0f;
  p.type = MINMAX;
  static const re2::RE2 minmaxRegex(
      "\\[\\s*([-+]?[0-9]*\\.?[0-9]+)\\s*,\\s*([-+]?[0-9]*\\.?[0-9]+)\\s*\\](\\s*[-+]?[0-9]*\\.?[0-9]+)?"); // matches
                                                                                                            // [<float>
                                                                                                            // ,
                                                                                                            // <float>]
                                                                                                            // <optional
                                                                                                            // weight>
  std::array<std::string, 3> values;
  std::string valueX, valueY, weight;
  MT_THROW_IF(
      !re2::RE2::FullMatch(valueStr, minmaxRegex, &valueX, &valueY, &weight),
      "Unrecognized minmax limit token in parameter configuration : {}",
      valueStr);
  p.data.minMax.parameterIndex = parameterIndex;
  p.data.minMax.limits = Vector2f(std::stof(valueX), std::stof(valueY));
  if (!weight.empty())
    p.weight = std::stof(weight);
  pl.push_back(std::move(p));
}

void parseMinmaxWithJointIndex(
    ParameterLimits& pl,
    const std::string& valueStr,
    size_t jointIndex,
    size_t jointParameter) {
  // create new parameterlimit
  ParameterLimit p;
  p.weight = 1.0f;
  p.type = MINMAX_JOINT;
  static const re2::RE2 minmaxRegex(
      "\\[\\s*([-+]?[0-9]*\\.?[0-9]+)\\s*,\\s*([-+]?[0-9]*\\.?[0-9]+)\\s*\\](\\s*[-+]?[0-9]*\\.?[0-9]+)?"); // matches
                                                                                                            // [<float>
                                                                                                            // ,
                                                                                                            // <float>]
                                                                                                            // <optional
                                                                                                            // weight>
  std::string valueX, valueY, weight;
  MT_THROW_IF(
      !re2::RE2::FullMatch(valueStr, minmaxRegex, &valueX, &valueY, &weight),
      "Unrecognized minmax limit token in parameter configuration : {}",
      valueStr);
  p.data.minMaxJoint.jointIndex = jointIndex;
  p.data.minMaxJoint.jointParameter = jointParameter;
  p.data.minMaxJoint.limits = Vector2f(std::stof(valueX), std::stof(valueY));
  if (!weight.empty())
    p.weight = std::stof(weight);
  pl.push_back(std::move(p));
}

void parseMinmaxPassive(
    ParameterLimits& pl,
    const std::string& valueStr,
    size_t jointIndex,
    size_t jointParameter) {
  // create new parameterlimit
  ParameterLimit p;
  p.weight = 1.0f;
  p.type = MINMAX_JOINT_PASSIVE;
  static const re2::RE2 minmaxRegex(
      "\\[\\s*([-+]?[0-9]*\\.?[0-9]+)\\s*,\\s*([-+]?[0-9]*\\.?[0-9]+)\\s*\\](\\s*[-+]?[0-9]*\\.?[0-9]+)?"); // matches
                                                                                                            // [<float>
                                                                                                            // ,
                                                                                                            // <float>]
                                                                                                            // <optional
                                                                                                            // weight>
  std::string valueX, valueY, weight;
  MT_THROW_IF(
      !re2::RE2::FullMatch(valueStr, minmaxRegex, &valueX, &valueY, &weight),
      "Unrecognized minmax limit token in parameter configuration : {}",
      valueStr);
  p.data.minMaxJoint.jointIndex = jointIndex;
  p.data.minMaxJoint.jointParameter = jointParameter;
  p.data.minMaxJoint.limits = Vector2f(std::stof(valueX), std::stof(valueY));
  if (!weight.empty())
    p.weight = std::stof(weight);
  pl.push_back(std::move(p));
}

void parseLinear(
    ParameterLimits& pl,
    const std::string& valueStr,
    const ParameterTransform& parameterTransform,
    int parameterIndex) {
  // create new parameterlimit
  ParameterLimit p;
  p.weight = 1.0f;
  p.type = LINEAR;
  static const re2::RE2 linearRegex(
      "(\\w+)\\s*\\[\\s*([-+]?[0-9]*\\.?[0-9]+)\\s*,\\s*([-+]?[0-9]*\\.?[0-9]+)\\s*\\](\\s*[-+]?[0-9]*\\.?[0-9]+)?"); // matches <parametername> [<float> , <float>] <optional weight>
  std::array<std::string, 4> values;
  std::string name, scaleAndOffset, notUsed,
      weight; // TODO: Check if the second and third parameters are being correctly parsed
  MT_THROW_IF(
      !re2::RE2::FullMatch(valueStr, linearRegex, &name, &scaleAndOffset, &notUsed, &weight),
      "Unrecognized linear limit token in parameter configuration : {}",
      valueStr);

  int otherParameterIndex = -1;
  for (size_t d = 0; d < parameterTransform.name.size(); d++) {
    if (parameterTransform.name[d] == name) {
      otherParameterIndex = static_cast<int>(d);
      break;
    }
  }
  MT_THROW_IF(
      otherParameterIndex == -1,
      "Unrecognized parameter name for linear limit in parameter configuration : {}",
      valueStr);

  p.data.linear.referenceIndex = parameterIndex;
  p.data.linear.targetIndex = otherParameterIndex;
  p.data.linear.scale = std::stof(scaleAndOffset);
  p.data.linear.offset = std::stof(scaleAndOffset);
  if (!weight.empty())
    p.weight = std::stof(weight);
  pl.push_back(std::move(p));
}

void parseEllipsoid(
    ParameterLimits& pl,
    const std::string& valueStr,
    const Skeleton& skeleton,
    int jointIndex) {
  // create new parameterlimit
  ParameterLimit p;
  p.weight = 1.0f;
  p.type = ELLIPSOID;
  // format is is [offset] parent [translation] [rotation] [scale] <optional weight>
  static const re2::RE2 ellipsoidRegex(
      "\\[\\s*([-+]?[0-9]*\\.?[0-9]+)\\s*,\\s*([-+]?[0-9]*\\.?[0-9]+)\\s*,\\s*([-+]?[0-9]*\\.?[0-9]+)\\s*\\]\\s*"
      "(\\w+)\\s*"
      "\\[\\s*([-+]?[0-9]*\\.?[0-9]+)\\s*,\\s*([-+]?[0-9]*\\.?[0-9]+)\\s*,\\s*([-+]?[0-9]*\\.?[0-9]+)\\s*\\]\\s*"
      "\\[\\s*([-+]?[0-9]*\\.?[0-9]+)\\s*,\\s*([-+]?[0-9]*\\.?[0-9]+)\\s*,\\s*([-+]?[0-9]*\\.?[0-9]+)\\s*\\]\\s*"
      "\\[\\s*([-+]?[0-9]*\\.?[0-9]+)\\s*,\\s*([-+]?[0-9]*\\.?[0-9]+)\\s*,\\s*([-+]?[0-9]*\\.?[0-9]+)\\s*\\]\\s*"
      "(\\s*[-+]?[0-9]*\\.?[0-9]+)?");
  std::array<std::string, 14> values;
  std::string offsetX, offsetY, offsetZ, jointName, transX, transY, transZ, eulerZ, eulerY, eulerX,
      scaleX, scaleY, scaleZ, weight;
  MT_THROW_IF(
      !re2::RE2::FullMatch(
          valueStr,
          ellipsoidRegex,
          &offsetX,
          &offsetY,
          &offsetZ,
          &jointName,
          &transX,
          &transY,
          &transZ,
          &eulerZ,
          &eulerY,
          &eulerX,
          &scaleX,
          &scaleY,
          &scaleZ,
          &weight),
      "Unrecognized ellipsoid limit token in parameter configuration : {}",
      valueStr);

  const size_t ellipsoidJointIndex = skeleton.getJointIdByName(jointName);
  MT_THROW_IF(
      ellipsoidJointIndex == kInvalidIndex,
      "Unrecognized joint name for ellipsoid limit in parameter configuration : {}",
      valueStr);

  // parse ellipsoid number data
  p.data.ellipsoid.parent = jointIndex;
  p.data.ellipsoid.ellipsoidParent = ellipsoidJointIndex;
  p.data.ellipsoid.offset = Vector3f(std::stof(offsetX), std::stof(offsetY), std::stof(offsetZ));
  p.data.ellipsoid.ellipsoid = Affine3f::Identity();
  p.data.ellipsoid.ellipsoid.translation() =
      Vector3f(std::stof(transX), std::stof(transY), std::stof(transZ));
  const Vector3f angles =
      Vector3f(toRad(std::stof(eulerX)), toRad(std::stof(eulerY)), toRad(std::stof(eulerZ)));
  p.data.ellipsoid.ellipsoid.linear() =
      eulerXYZToRotationMatrix(angles, EulerConvention::EXTRINSIC) *
      Eigen::Scaling(std::stof(scaleX), std::stof(scaleY), std::stof(scaleZ));
  p.data.ellipsoid.ellipsoidInv = p.data.ellipsoid.ellipsoid.inverse();

  if (!weight.empty())
    p.weight = std::stof(weight);
  pl.push_back(std::move(p));
}

} // namespace

ParameterLimits parseParameterLimits(
    const std::string& data,
    const Skeleton& skeleton,
    const ParameterTransform& parameterTransform) {
  ParameterLimits pl;

  std::istringstream f(data);
  std::string line;
  while (std::getline(f, line)) {
    // erase all comments
    line = line.substr(0, line.find_first_of('#'));

    // ignore everything but limits
    if (line.find("limit") != 0)
      continue;

    // parse limits
    // token 1 = parameter name, token 2 = type, token 3 = value
    static const re2::RE2 reg("limit ([\\w.]+) (\\w+) (.*)");
    std::string parameterName;
    std::string type;
    std::string valueStr;
    MT_THROW_IF(
        !re2::RE2::FullMatch(line, reg, &parameterName, &type, &valueStr),
        "Could not parse limit line in parameter configuration : {}",
        line);

    int parameterIndex = -1;
    for (size_t d = 0; d < parameterTransform.name.size(); d++) {
      if (parameterTransform.name[d] == parameterName) {
        parameterIndex = static_cast<int>(d);
        break;
      }
    }

    // check if the first token is actually a joint name as well
    const size_t jointIndex =
        skeleton.getJointIdByName(parameterName.substr(0, parameterName.find_first_of(".")));
    size_t jointParameter = kInvalidIndex;
    if (jointIndex != kInvalidIndex) {
      const std::string jpString = parameterName.substr(parameterName.find_first_of(".") + 1);
      for (size_t j = 0; j < kParametersPerJoint; j++) {
        if (jpString == kJointParameterNames[j]) {
          jointParameter = j;
          break;
        }
      }
    }

    // create a new ParameterLimit by the type and add it to pl
    if (type == "minmax" && parameterIndex != -1) {
      parseMinmaxWithParameterIndex(pl, valueStr, parameterIndex);
    } else if (type == "minmax" && jointIndex != kInvalidIndex && jointParameter != kInvalidIndex) {
      parseMinmaxWithJointIndex(pl, valueStr, jointIndex, jointParameter);
    } else if (
        type == "minmax_passive" && jointIndex != kInvalidIndex &&
        jointParameter != kInvalidIndex) {
      parseMinmaxPassive(pl, valueStr, jointIndex, jointParameter);
    } else if (type == "linear" && parameterIndex != -1) {
      parseLinear(pl, valueStr, parameterTransform, parameterIndex);
    } else if (type == "ellipsoid" && jointIndex != kInvalidIndex) {
      parseEllipsoid(pl, valueStr, skeleton, jointIndex);
    } else if (type == "elipsoid" && jointIndex != kInvalidIndex) {
      MT_LOGW_ONCE(
          "Deprecated parameter limit type: {} (typo). Please use 'ellipsoid' instead.", type);
      parseEllipsoid(pl, valueStr, skeleton, jointIndex);
    } else {
      MT_THROW("Unexpected limit type '{}' in parameter configuration : {}", type, line);
    }
  }
  return pl;
}

} // namespace momentum
