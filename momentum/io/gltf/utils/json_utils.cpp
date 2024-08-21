/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/gltf/utils/json_utils.h"

#include "momentum/character/character.h"
#include "momentum/common/log.h"
#include "momentum/math/constants.h"
#include "momentum/math/utility.h"

#include <nlohmann/json.hpp>

#include <stdexcept>

namespace momentum {

void parameterSetsToJson(const Character& character, nlohmann::json& j) {
  j = nlohmann::json::object();
  for (const auto& set : character.parameterTransform.parameterSets) {
    j[set.first] = nlohmann::json::array();
    for (size_t i = 0; i < character.parameterTransform.name.size(); i++) {
      if (set.second.test(i)) {
        j[set.first].push_back(character.parameterTransform.name[i]);
      }
    }
  }
}

ParameterSets parameterSetsFromJson(const Character& character, const nlohmann::json& j) {
  ParameterSets result;

  for (const auto& parameterSet : j.items()) {
    auto& set = result[parameterSet.key()];
    for (const auto& parameter : parameterSet.value().items()) {
      const auto parameterIndex =
          character.parameterTransform.getParameterIdByName(parameter.value());

      if (parameterIndex != kInvalidIndex)
        set.set(parameterIndex);
    }
  }

  return result;
}

void poseConstraintsToJson(const Character& character, nlohmann::json& j) {
  j = nlohmann::json::object();
  for (const auto& csts : character.parameterTransform.poseConstraints) {
    for (const auto& pc : csts.second.parameterIdValue) {
      j[csts.first][character.parameterTransform.name[pc.first]] = pc.second;
    }
  }
}

PoseConstraints poseConstraintsFromJson(const Character& character, const nlohmann::json& j) {
  PoseConstraints result;

  for (const auto& constraint : j.items()) {
    auto& pose = result[constraint.key()];
    for (const auto& constraintPair : constraint.value().items()) {
      const auto parameterIndex =
          character.parameterTransform.getParameterIdByName(constraintPair.key());

      const float value = constraintPair.value();
      if (parameterIndex != kInvalidIndex)
        pose.parameterIdValue.emplace_back(parameterIndex, value);
    }
  }

  return result;
}

void parameterTransformToJson(const Character& character, nlohmann::json& j) {
  j["parameters"] = character.parameterTransform.name;

  for (size_t jt = 0; jt < character.skeleton.joints.size(); jt++) {
    const auto& jointName = character.skeleton.joints[jt].name;
    const size_t paramIndex = jt * kParametersPerJoint;
    for (size_t d = 0; d < kParametersPerJoint; d++) {
      for (auto index = character.parameterTransform.transform.outerIndexPtr()[paramIndex + d];
           index < character.parameterTransform.transform.outerIndexPtr()[paramIndex + d + 1];
           ++index) {
        const auto& parameterIndex = character.parameterTransform.transform.innerIndexPtr()[index];
        const auto& parameterValue = character.parameterTransform.transform.valuePtr()[index];

        j["joints"][jointName][kJointParameterNames[d]]
         [character.parameterTransform.name[parameterIndex]] = parameterValue;
      }
    }
  }
}

ParameterTransform parameterTransformFromJson(const Character& character, const nlohmann::json& j) {
  ParameterTransform pt;
  try {
    pt.activeJointParams =
        VectorX<bool>::Constant(character.skeleton.joints.size() * kParametersPerJoint, false);
    pt.offsets = VectorXf::Zero(character.skeleton.joints.size() * kParametersPerJoint);

    // load all the parameter names
    auto parameterItr = j.find("parameters");
    if (parameterItr == j.end()) {
      throw std::runtime_error("No 'parameters' found in parameter transform.");
    }

    for (const auto& p : *parameterItr)
      pt.name.push_back(p.get<std::string>());

    // triplet list
    std::vector<Eigen::Triplet<float>> triplets;

    auto jointItr = j.find("joints");
    if (jointItr == j.end()) {
      throw std::runtime_error("No 'joints' found in parameter transform.");
    }

    for (const auto& joint : jointItr->items()) {
      const std::string jointName = joint.key();

      // get the right joint to modify
      const size_t jointIndex = character.skeleton.getJointIdByName(jointName);
      if (jointIndex == kInvalidIndex)
        throw std::runtime_error(std::string("Unknown joint name in expression : ") + jointName);

      for (const auto& jointParameter : joint.value().items()) {
        const std::string jointParameterName = jointParameter.key();

        // the first pToken is the name of the joint and it's attribute type
        size_t attributeIndex = kInvalidIndex;
        for (size_t jt = 0; jt < kParametersPerJoint; jt++) {
          if (jointParameterName == kJointParameterNames[jt]) {
            attributeIndex = jt;
            break;
          }
        }

        // if we didn't find a right name exit with an error
        if (attributeIndex == kInvalidIndex)
          throw std::runtime_error(
              std::string("Unknown channel name in expression : ") + jointParameterName);

        // enable the attribute in the skeleton if we have a parameter controlling it
        pt.activeJointParams[jointIndex * kParametersPerJoint + attributeIndex] = 1;

        // split the parameter names
        for (const auto& parameter : jointParameter.value().items()) {
          // get weight and name
          const float weight = parameter.value();
          const std::string& parameterName = parameter.key();

          auto parameterIndex = pt.getParameterIdByName(parameterName);

          if (parameterIndex == kInvalidIndex)
            throw std::runtime_error(
                std::string("Unknown parameter name in expression : ") + parameterName);

          // add triplet
          triplets.push_back(Eigen::Triplet<float>(
              static_cast<int>(jointIndex) * kParametersPerJoint + static_cast<int>(attributeIndex),
              static_cast<int>(parameterIndex),
              weight));
        }
      }
    }

    // resize the Transform matrix to the correct size
    pt.transform.resize(
        static_cast<int>(character.skeleton.joints.size()) * kParametersPerJoint,
        static_cast<int>(pt.name.size()));

    // finish parameter setup by creating sparse matrix
    pt.transform.setFromTriplets(triplets.begin(), triplets.end());
  } catch (...) {
    // create an identity parameter transform
    pt = pt.identity(character.skeleton.getJointNames());
  }

  return pt;
}

void parameterLimitsToJson(const Character& character, nlohmann::json& j) {
  j = nlohmann::json::array();
  for (const auto& lim : character.parameterLimits) {
    j.push_back(nlohmann::json::object());
    auto& li = j.back();
    li["weight"] = lim.weight;

    switch (lim.type) {
      case MINMAX:
        li["type"] = "minmax";
        li["parameter"] = character.parameterTransform.name[lim.data.minMax.parameterIndex];
        li["limits"] = lim.data.minMax.limits;
        break;
      case MINMAX_JOINT:
        li["type"] = "minmax_joint";
        li["jointIndex"] = character.skeleton.joints[lim.data.minMaxJoint.jointIndex].name;
        li["jointParameter"] = kJointParameterNames[lim.data.minMaxJoint.jointParameter];
        li["limits"] = lim.data.minMaxJoint.limits;
        break;
      case MINMAX_JOINT_PASSIVE:
        li["type"] = "minmax_joint_passive";
        li["jointIndex"] = character.skeleton.joints[lim.data.minMaxJoint.jointIndex].name;
        li["jointParameter"] = kJointParameterNames[lim.data.minMaxJoint.jointParameter];
        li["limits"] = lim.data.minMaxJoint.limits;
        break;
      case LINEAR:
        li["type"] = "limit";
        li["referenceParameter"] =
            character.parameterTransform.name[lim.data.linear.referenceIndex];
        li["targetParameter"] = character.parameterTransform.name[lim.data.linear.targetIndex];
        li["scale"] = lim.data.linear.scale;
        li["offset"] = lim.data.linear.offset;
        break;
      case ELLIPSOID: {
        li["type"] = "ellipsoid";
        li["parent"] = character.skeleton.joints[lim.data.ellipsoid.parent].name;
        li["ellipsoidParent"] = character.skeleton.joints[lim.data.ellipsoid.ellipsoidParent].name;
        li["offset"] = lim.data.ellipsoid.offset * toM();
        auto eli = lim.data.ellipsoid.ellipsoid;
        eli.translation() *= toM();
        toJson(eli.matrix(), li["ellipsoid"]);
        break;
      }
      default: {
        MT_LOGE(
            "Unknown parameter limit type '{}' from character name '{}'",
            toString(lim.type),
            character.name);
        break;
      }
    }
  }
}

ParameterLimits parameterLimitsFromJson(const Character& character, const nlohmann::json& j) {
  ParameterLimits result;

  for (const auto& element : j) {
    const std::string type = element.value("type", "");
    ParameterLimit l;
    l.weight = element.value("weight", 0.0f);
    if (type == "minmax") {
      l.type = MINMAX;
      l.data.minMax.parameterIndex =
          character.parameterTransform.getParameterIdByName(element.value("parameter", ""));
      l.data.minMax.limits = fromJson<Vector2f>(element["limits"]);
    } else if (type == "minmax_joint") {
      l.type = MINMAX_JOINT;
      l.data.minMaxJoint.jointIndex =
          character.skeleton.getJointIdByName(element.value("jointIndex", ""));
      const std::string attribute = element.value("jointParameter", "");
      size_t attributeIndex = kInvalidIndex;
      for (size_t t = 0; t < kParametersPerJoint; t++) {
        if (attribute == kJointParameterNames[t]) {
          attributeIndex = t;
          break;
        }
      }
      l.data.minMaxJoint.jointParameter = attributeIndex;
      l.data.minMaxJoint.limits = fromJson<Vector2f>(element["limits"]);
    } else if (type == "minmax_joint_passive") {
      l.type = MINMAX_JOINT_PASSIVE;
      l.data.minMaxJoint.jointIndex =
          character.skeleton.getJointIdByName(element.value("jointIndex", ""));
      const std::string attribute = element.value("jointParameter", "");
      size_t attributeIndex = kInvalidIndex;
      for (size_t t = 0; t < kParametersPerJoint; t++) {
        if (attribute == kJointParameterNames[t]) {
          attributeIndex = t;
          break;
        }
      }
      l.data.minMaxJoint.jointParameter = attributeIndex;
      l.data.minMaxJoint.limits = fromJson<Vector2f>(element["limits"]);
    } else if (type == "linear") {
      l.type = LINEAR;
      l.data.linear.referenceIndex = character.parameterTransform.getParameterIdByName(
          element.value("referenceParameter", ""));
      l.data.linear.targetIndex =
          character.parameterTransform.getParameterIdByName(element.value("targetParameter", ""));
      l.data.linear.scale = element["scale"];
      l.data.linear.offset = element["offset"];
    } else if (type == "ellipsoid") {
      l.type = ELLIPSOID;
      l.data.ellipsoid.parent = character.skeleton.getJointIdByName(element.value("parent", ""));
      l.data.ellipsoid.ellipsoidParent =
          character.skeleton.getJointIdByName(element.value("ellipsoidParent", ""));
      l.data.ellipsoid.offset = fromJson<Vector3f>(element["offset"]);
      l.data.ellipsoid.offset /= toM();
      l.data.ellipsoid.ellipsoid.matrix() = fromJson<Matrix4f>(element["ellipsoid"]);
      l.data.ellipsoid.ellipsoid.translation() /= toM();
      l.data.ellipsoid.ellipsoidInv = l.data.ellipsoid.ellipsoid.inverse();
    } else if (type == "elipsoid") {
      // TODO: Remove once all the model files are migrated to ellipsoid
      MT_LOGW_ONCE(
          "Deprecated parameter limit type: {} (typo). Please use 'ellipsoid' instead.", type);
      l.type = ELLIPSOID;
      l.data.ellipsoid.parent = character.skeleton.getJointIdByName(element.value("parent", ""));
      l.data.ellipsoid.ellipsoidParent =
          character.skeleton.getJointIdByName(element.value("elipsoidParent", ""));
      l.data.ellipsoid.offset = fromJson<Vector3f>(element["offset"]);
      l.data.ellipsoid.offset /= toM();
      l.data.ellipsoid.ellipsoid.matrix() = fromJson<Matrix4f>(element["elipsoid"]);
      l.data.ellipsoid.ellipsoid.translation() /= toM();
      l.data.ellipsoid.ellipsoidInv = l.data.ellipsoid.ellipsoid.inverse();
    } else {
      throw std::runtime_error(
          "Unknown parameter limit type '" + type + "' from character name '" + character.name +
          "'.");
    }
    result.push_back(l);
  }

  return result;
}

} // namespace momentum
