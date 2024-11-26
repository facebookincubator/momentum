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

namespace momentum {

namespace {

enum class Token {
  OpenBracket,
  CloseBracket,
  Number,
  Comma,
  Identifier,
  Eof,
};

std::string tokenStr(const Token& t) {
  switch (t) {
    case Token::OpenBracket:
      return "[";
    case Token::CloseBracket:
      return "]";
    case Token::Number:
      return "float value";
    case Token::Comma:
      return ",";
    case Token::Eof:
      return "EOF";
    case Token::Identifier:
      return "Identifier";
  }
  return "Unknown";
}

class Tokenizer {
 public:
  Tokenizer(
      const std::string& str,
      size_t lineIndex,
      const ParameterTransform& parameterTransform,
      const Skeleton& skeleton)
      : _str(str),
        _lineIndex(lineIndex),
        _parameterTransform(parameterTransform),
        _skeleton(skeleton) {
    _curPos = _str.begin();
    _tokenStart = _curPos;
    advance();
  }

  [[nodiscard]] bool isEOF() const {
    return _curToken == Token::Eof;
  }

  void eatToken(Token expectedToken) {
    verifyToken(expectedToken);
    advance();
  }

  [[nodiscard]] std::string getIdentifier() {
    verifyToken(Token::Identifier);
    std::string result(_tokenStart, _curPos);
    advance();
    return result;
  }

  [[nodiscard]] float getNumber() {
    verifyToken(Token::Number);
    float result = std::stof(std::string(_tokenStart, _curPos));
    advance();
    return result;
  }

  template <int N>
  [[nodiscard]] Vector<float, N> getVec() {
    Vector<float, N> result;

    eatToken(Token::OpenBracket);
    for (Eigen::Index i = 0; i < N; ++i) {
      if (i > 0) {
        eatToken(Token::Comma);
      }
      result[i] = getNumber();
    }
    eatToken(Token::CloseBracket);

    return result;
  }

  [[nodiscard]] std::pair<float, float> getPair() {
    Vector2<float> result = getVec<2>();
    return {result.x(), result.y()};
  }

  [[nodiscard]] size_t modelParameterIndexFromName(const std::string& modelParameterName) const {
    auto itr = std::find(
        _parameterTransform.name.begin(), _parameterTransform.name.end(), modelParameterName);
    MT_THROW_IF(
        itr == _parameterTransform.name.end(),
        "Parameter {} not found in transform at line {}: {}",
        modelParameterName,
        _lineIndex,
        _str);
    return std::distance(_parameterTransform.name.begin(), itr);
  }

  [[nodiscard]] size_t getModelParameterIndex() {
    return modelParameterIndexFromName(getIdentifier());
  }

  [[nodiscard]] std::pair<size_t, size_t> jointParameterIndexFromName(
      const std::string& jointParameterName) const {
    auto dotPos = jointParameterName.find('.');
    MT_THROW_IF(
        dotPos == std::string::npos,
        "Could not find '.' in joint parameter name {}; expected 'joint_name.tx/tz/tz/rx/ry/rz/s' at line {}: {}",
        jointParameterName,
        _lineIndex,
        _str);

    auto jointName = jointParameterName.substr(0, dotPos);
    const auto jointIndex = jointIndexFromName(jointName);

    const auto subParamName = jointParameterName.substr(dotPos + 1);
    const auto* kJointParameterNamesEnd = kJointParameterNames + kParametersPerJoint;
    const auto* subParamItr =
        std::find_if(kJointParameterNames, kJointParameterNamesEnd, [&](const char* name) {
          return subParamName == name;
        });
    MT_THROW_IF(
        subParamItr == kJointParameterNamesEnd,
        "Invalid parameter name {}, expected tx/ty/tz/rx/ry/rz/s at line {}: {}",
        subParamName,
        _lineIndex,
        _str);
    return {jointIndex, std::distance(kJointParameterNames, subParamItr)};
  }

  [[nodiscard]] size_t jointIndexFromName(const std::string& jointName) const {
    const auto jointIndex = _skeleton.getJointIdByName(jointName);
    MT_THROW_IF(
        jointIndex == kInvalidIndex,
        "Could not find joint {} in skeleton at line {}: {}",
        jointName,
        _lineIndex,
        _str);

    return jointIndex;
  }

  [[nodiscard]] size_t getJointIndex() {
    return jointIndexFromName(getIdentifier());
  }

  [[nodiscard]] size_t currentPosition() const {
    return std::distance(_str.begin(), _curPos);
  }

 private:
  void verifyToken(Token expectedToken) const {
    MT_THROW_IF(
        _curToken != expectedToken,
        "Expected {} at character {} of line {}: {}",
        tokenStr(expectedToken),
        std::distance(_str.begin(), _tokenStart),
        _lineIndex,
        _str);
  }

  void advance() {
    // Eat all spaces:
    while (_curPos != _str.end() && std::isspace(*_curPos)) {
      ++_curPos;
    }

    _tokenStart = _curPos;
    if (_curPos == _str.end()) {
      _curToken = Token::Eof;
      return;
    }

    if (*_curPos == '[') {
      _curToken = Token::OpenBracket;
      ++_curPos;
      return;
    }

    if (*_curPos == ']') {
      _curToken = Token::CloseBracket;
      ++_curPos;
      return;
    }

    if (*_curPos == ',') {
      _curToken = Token::Comma;
      ++_curPos;
      return;
    }

    // We won't allow numbers to start with a period so you'll need to say 0.1 instead of .1,
    // this will help to avoid ambiguous cases whre the dot is part of a joint parameter definition.
    if (std::isdigit(*_curPos) || *_curPos == '-' || *_curPos == '+') {
      _curToken = Token::Number;
      while (_curPos != _str.end() &&
             (std::isdigit(*_curPos) || *_curPos == '-' || *_curPos == '+' || *_curPos == '.' ||
              *_curPos == 'e')) {
        ++_curPos;
      }
      return;
    }

    if (_curPos != _str.end() && (std::isalpha(*_curPos) || *_curPos == '_')) {
      _curToken = Token::Identifier;
      // period in identifier is okay, we consider "joint.rx" as a single identifier
      while (std::isalnum(*_curPos) || *_curPos == '_' || *_curPos == '.') {
        ++_curPos;
      }
      return;
    }

    MT_THROW(
        "Unexpected token at character {} of line {}: {}",
        std::distance(_str.begin(), _tokenStart),
        _lineIndex,
        _str);
  }

  const std::string& _str;
  const size_t _lineIndex;
  const ParameterTransform& _parameterTransform;
  const Skeleton& _skeleton;
  std::string::const_iterator _curPos;
  std::string::const_iterator _tokenStart;
  Token _curToken = Token::Eof;
};

void parseMinmax(const std::string& parameterName, ParameterLimits& pl, Tokenizer& tokenizer) {
  if (parameterName.find('.') == std::string::npos) {
    // "[<min>, <max>] <optional weight>"
    ParameterLimit p;
    p.weight = 1.0f;
    p.type = MinMax;
    p.data.minMax.parameterIndex = tokenizer.modelParameterIndexFromName(parameterName);

    std::tie(p.data.minMax.limits.x(), p.data.minMax.limits.y()) = tokenizer.getPair();
    if (!tokenizer.isEOF()) {
      p.weight = tokenizer.getNumber();
    }
    pl.push_back(p);
  } else {
    // "[<min>, <max>] <optional weight>"
    ParameterLimit p;
    p.weight = 1.0f;
    p.type = MinMaxJoint;
    std::tie(p.data.minMaxJoint.jointIndex, p.data.minMaxJoint.jointParameter) =
        tokenizer.jointParameterIndexFromName(parameterName);

    std::tie(p.data.minMaxJoint.limits.x(), p.data.minMaxJoint.limits.y()) = tokenizer.getPair();
    if (!tokenizer.isEOF()) {
      p.weight = tokenizer.getNumber();
    }
    pl.push_back(p);
  }
}

void parseMinmaxPassive(
    const std::string& parameterName,
    ParameterLimits& pl,
    Tokenizer& tokenizer) {
  // "[<min>, <max>] <optional weight>"
  ParameterLimit p;
  p.weight = 1.0f;
  p.type = MinMaxJointPassive;
  std::tie(p.data.minMaxJoint.jointIndex, p.data.minMaxJoint.jointParameter) =
      tokenizer.jointParameterIndexFromName(parameterName);

  std::tie(p.data.minMaxJoint.limits.x(), p.data.minMaxJoint.limits.y()) = tokenizer.getPair();
  if (!tokenizer.isEOF()) {
    p.weight = tokenizer.getNumber();
  }
  pl.push_back(p);
}

void parseLinear(const std::string& parameterName, ParameterLimits& pl, Tokenizer& tokenizer) {
  // create new parameterlimit
  ParameterLimit p;
  p.weight = 1.0f;
  p.type = Linear;
  p.data.linear.referenceIndex = tokenizer.modelParameterIndexFromName(parameterName);

  // "<model parameter name> [<scale>, <offset>] <optional weight> [<opt. range min>, <opt. range
  // max>]"
  p.data.linear.targetIndex = tokenizer.getModelParameterIndex();

  std::tie(p.data.linear.scale, p.data.linear.offset) = tokenizer.getPair();

  if (!tokenizer.isEOF()) {
    p.weight = tokenizer.getNumber();
  }
  pl.push_back(std::move(p));
}

void parseEllipsoid(const std::string& jointName, ParameterLimits& pl, Tokenizer& tokenizer) {
  // create new parameterlimit
  ParameterLimit p;
  p.weight = 1.0f;
  p.type = Ellipsoid;

  // "[offset] parent [translation] [rotation] [scale] <optional weight>"
  p.data.ellipsoid.parent = tokenizer.jointIndexFromName(jointName);
  p.data.ellipsoid.offset = tokenizer.getVec<3>();

  p.data.ellipsoid.ellipsoidParent = tokenizer.getJointIndex();

  const Eigen::Vector3f translation = tokenizer.getVec<3>();
  const Eigen::Vector3f eulerZYX = tokenizer.getVec<3>();
  const Eigen::Vector3f scale = tokenizer.getVec<3>();

  if (!tokenizer.isEOF()) {
    p.weight = tokenizer.getNumber();
  }

  // parse ellipsoid number data
  p.data.ellipsoid.ellipsoid = Affine3f::Identity();
  p.data.ellipsoid.ellipsoid.translation() = translation;
  const Vector3f eulerXYZ = Vector3f(toRad(eulerZYX.z()), toRad(eulerZYX.y()), toRad(eulerZYX.x()));
  p.data.ellipsoid.ellipsoid.linear() =
      eulerXYZToRotationMatrix(eulerXYZ, EulerConvention::Extrinsic) *
      Eigen::Scaling(scale.x(), scale.y(), scale.z());
  p.data.ellipsoid.ellipsoidInv = p.data.ellipsoid.ellipsoid.inverse();

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
  size_t lineIndex = 0;
  while (std::getline(f, line)) {
    ++lineIndex;

    // erase all comments
    line = line.substr(0, line.find_first_of('#'));

    // ignore everything but limits
    if (line.find("limit") != 0) {
      continue;
    }

    // parse limits
    // token 1 = parameter name, token 2 = type, token 3 = value
    Tokenizer tokenizer(line, lineIndex, parameterTransform, skeleton);

    // Eat the initial "limit":
    std::ignore = tokenizer.getIdentifier();

    const std::string parameterName = tokenizer.getIdentifier();

    const std::string type = tokenizer.getIdentifier();
    // create a new ParameterLimit by the type and add it to pl
    if (type == "minmax") {
      parseMinmax(parameterName, pl, tokenizer);
    } else if (type == "minmax_passive") {
      parseMinmaxPassive(parameterName, pl, tokenizer);
    } else if (type == "linear") {
      parseLinear(parameterName, pl, tokenizer);
    } else if (type == "ellipsoid") {
      parseEllipsoid(parameterName, pl, tokenizer);
    } else if (type == "elipsoid") {
      MT_LOGW_ONCE(
          "Deprecated parameter limit type: {} (typo). Please use 'ellipsoid' instead.", type);
      parseEllipsoid(parameterName, pl, tokenizer);
    } else {
      MT_THROW("Unexpected parameter limit type {} in line {}: {}", type, lineIndex, line);
    }

    MT_THROW_IF(
        !tokenizer.isEOF(),
        "Unexpected token at character {} in line {}: {}",
        tokenizer.currentPosition(),
        lineIndex,
        line);
  }
  return pl;
}

namespace {

std::string vecToString(Eigen::Ref<const Eigen::VectorXf> vec) {
  std::ostringstream oss;
  oss << "[";
  for (Eigen::Index i = 0; i < vec.size(); ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << vec[i];
  }
  oss << "]";

  return oss.str();
}

} // namespace

std::string writeParameterLimits(
    const ParameterLimits& parameterLimits,
    const Skeleton& skeleton,
    const ParameterTransform& parameterTransform) {
  std::ostringstream oss;

  auto jointParameterToName = [&](size_t jointIndex, size_t jointParameterIndex) {
    MT_CHECK_LT(jointParameterIndex, kParametersPerJoint);
    MT_CHECK_LT(jointIndex, skeleton.joints.size());
    return (skeleton.joints.at(jointIndex).name + "." + kJointParameterNames[jointParameterIndex]);
  };

  for (const auto& limit : parameterLimits) {
    oss << "limit ";
    switch (limit.type) {
      case LimitType::MinMax:
        oss << parameterTransform.name.at(limit.data.minMax.parameterIndex) << " minmax "
            << vecToString(limit.data.minMax.limits);
        break;
      case LimitType::MinMaxJoint:
        oss << jointParameterToName(
                   limit.data.minMaxJoint.jointIndex, limit.data.minMaxJoint.jointParameter)
            << " minmax " << vecToString(limit.data.minMaxJoint.limits);
        break;
      case LimitType::MinMaxJointPassive:
        oss << jointParameterToName(
                   limit.data.minMaxJoint.jointIndex, limit.data.minMaxJoint.jointParameter)
            << " minmax_passive " << vecToString(limit.data.minMaxJoint.limits);
        break;
      case LimitType::Linear:
        oss << parameterTransform.name.at(limit.data.linear.referenceIndex) << " linear "
            << parameterTransform.name.at(limit.data.linear.targetIndex) << " "
            << vecToString(Eigen::Vector2f(limit.data.linear.scale, limit.data.linear.offset));
        break;

      case LimitType::Ellipsoid: {
        const Eigen::Affine3f ellipsoid = limit.data.ellipsoid.ellipsoid;
        const Eigen::Vector3f translation = ellipsoid.translation();
        Eigen::Matrix3f rotationMat;
        Eigen::Matrix3f scalingMat;
        ellipsoid.computeRotationScaling(&rotationMat, &scalingMat);

        const Eigen::Vector3f eulerXYZVecRad =
            rotationMatrixToEulerXYZ(rotationMat, EulerConvention::Extrinsic);
        // Not sure why this is reversed but it's implemented this way in the parser above.
        const Eigen::Vector3f eulerZYXVecDeg(
            toDeg(eulerXYZVecRad.z()), toDeg(eulerXYZVecRad.y()), toDeg(eulerXYZVecRad.x()));

        MT_CHECK_LT(limit.data.ellipsoid.parent, skeleton.joints.size());
        MT_CHECK_LT(limit.data.ellipsoid.ellipsoidParent, skeleton.joints.size());
        oss << skeleton.joints.at(limit.data.ellipsoid.parent).name << " ellipsoid "
            << vecToString(limit.data.ellipsoid.offset) << " "
            << skeleton.joints.at(limit.data.ellipsoid.ellipsoidParent).name << " "
            << vecToString(translation) << " " << vecToString(eulerZYXVecDeg) << " "
            << vecToString(scalingMat.diagonal());
      }
    }

    if (limit.weight != 1.0f) {
      oss << " " << limit.weight;
    }
    oss << "\n";
  }

  return oss.str();
}

} // namespace momentum
