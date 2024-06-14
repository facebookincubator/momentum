/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/character/parameter_limits.h"

#include "momentum/character/parameter_transform.h"
#include "momentum/character/skeleton.h"
#include "momentum/common/checks.h"
#include "momentum/math/utility.h"

namespace momentum {

std::string_view toString(const LimitType type) {
  switch (type) {
    case MINMAX:
      return "MINMAX";
    case MINMAX_JOINT:
      return "MINMAX_JOINT";
    case MINMAX_JOINT_PASSIVE:
      return "MINMAX_JOINT_PASSIVE";
    case LINEAR:
      return "LINEAR";
    case ELLIPSOID:
      return "ELLIPSOID";
    default:
      return "UNKNOWN";
  }
}

JointParameters applyJointParameterLimits(
    const ParameterLimits& limits,
    const JointParameters& jointParams) {
  JointParameters res = jointParams;
  for (const auto& limit : limits) {
    if (limit.type != MINMAX_JOINT_PASSIVE)
      continue;

    const auto& data = limit.data.minMaxJoint;
    const size_t parameterIndex = data.jointIndex * kParametersPerJoint + data.jointParameter;
    MT_CHECK(
        parameterIndex <= gsl::narrow<size_t>(jointParams.size()),
        "{} vs {}",
        parameterIndex,
        jointParams.size());
    if (res(parameterIndex) < data.limits[0])
      res(parameterIndex) = data.limits[0];
    if (res(parameterIndex) > data.limits[1])
      res(parameterIndex) = data.limits[1];
  }
  return res;
}

ParameterLimits getPoseConstraintParameterLimits(
    const std::string& name,
    const ParameterTransform& pt,
    const float weight) {
  ParameterLimits res;

  if (pt.poseConstraints.count(name) > 0) {
    for (const auto& pc : pt.poseConstraints.at(name).parameterIdValue) {
      ParameterLimit p;
      p.type = MINMAX;
      p.data.minMax.parameterIndex = pc.first;
      p.data.minMax.limits = Vector2f::Constant(pc.second);
      p.weight = weight;
      res.emplace_back(p);
    }
  }

  return res;
}

LimitData::LimitData() {
  std::fill_n(rawData, sizeof(rawData), 0);
}

LimitData::LimitData(const LimitData& rhs) {
  std::copy_n(rhs.rawData, sizeof(rawData), rawData);
}

LimitData& LimitData::operator=(const LimitData& rhs) {
  if (&rhs != this)
    std::copy_n(rhs.rawData, sizeof(rawData), rawData);
  return *this;
}

bool LimitData::operator==(const LimitData& limitData) const {
  return (std::memcmp(this, &limitData, sizeof(LimitData)) == 0);
}

} // namespace momentum
