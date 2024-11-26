/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/character/types.h>
#include <momentum/common/memory.h>
#include <momentum/math/utility.h>

#include <cstring>

namespace momentum {

// structure for applying limits to parameter sets
// the simplest kind of limits could be just min/max limits on parameters, but we also have the
// option of modeling more complicated limits that depend on each other

// limit type
enum LimitType { MinMax, MinMaxJoint, MinMaxJointPassive, Linear, Ellipsoid };

[[nodiscard]] std::string_view toString(LimitType type);

struct LimitMinMax {
  size_t parameterIndex; // index of parameter influenced
  Vector2f limits; // min and max values of the parameter
};

struct LimitMinMaxJoint {
  size_t jointIndex; // index of joint influenced
  size_t jointParameter; // parameter of joint influenced [ tx, ty, tz, rx, ry, rz, sc ]
  Vector2f limits; // min and max values of the parameter
};

struct LimitLinear { // set joints to be similar by a linear relation i.e. p_0 = p_1 * x - o
  size_t referenceIndex; // index of reference parameter (p_0)
  size_t targetIndex; // index of target parameter (p_1)
  float scale; // linear scale of parameter (x)
  float offset; // offset (positive and negative) of acceptable parameter zone

  // Range where limit is applied (in target parameter values p1).  This can be used to construct
  // piecewise linear limits.  Note that the minimum value of the range is inclusive but the maximum
  // value is noninclusive, this is to allow constructing overlapping limits without
  // double-counting.
  float rangeMin;
  float rangeMax;
};

struct LimitEllipsoid {
  alignas(32) Affine3f ellipsoid;
  alignas(32) Affine3f ellipsoidInv;
  alignas(32) Vector3f offset;
  size_t ellipsoidParent;
  size_t parent;
};

union LimitData {
  LimitMinMax minMax;
  LimitMinMaxJoint minMaxJoint;
  LimitLinear linear;
  LimitEllipsoid ellipsoid;
  unsigned char rawData[512];

  // Need to explicitly write these constructors to just copy the raw memory
  // since otherwise the compiler doesn't know which copy constructor to call.
  LimitData(const LimitData& rhs);
  LimitData();
  LimitData& operator=(const LimitData& rhs);
  bool operator==(const LimitData& limitData) const;
};

struct ParameterLimit {
  LimitData data; // limit data depending on the type
  LimitType type = LimitType::MinMax; // type of limit
  float weight = 1.0f; // limit weight

  inline bool operator==(const ParameterLimit& parameterLimit) const {
    return (
        (data == parameterLimit.data) && (type == parameterLimit.type) &&
        isApprox(weight, parameterLimit.weight));
  };
};

using ParameterLimits = std::vector<ParameterLimit>; // a list of limits

JointParameters applyPassiveJointParameterLimits(
    const ParameterLimits& limits,
    const JointParameters& jointParams);

ParameterLimits getPoseConstraintParameterLimits(
    const std::string& name,
    const ParameterTransform& pt,
    float weight = 1.0f);

bool isInRange(const LimitLinear& limit, float value);

MOMENTUM_DEFINE_POINTERS(ParameterLimits)
} // namespace momentum
