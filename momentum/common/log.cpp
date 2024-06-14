/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/common/log.h"

#include <algorithm>
#include <stdexcept>

#if !defined(MOMENTUM_WITH_XR_LOGGER)
#include <iostream>
#endif

namespace momentum {

namespace {

#if defined(MOMENTUM_WITH_XR_LOGGER)

[[nodiscard]] arvr::logging::Level toArvrLogLevel(LogLevel level) {
  switch (level) {
    case LogLevel::Disabled:
      return arvr::logging::Level::Disabled;
    case LogLevel::Error:
      return arvr::logging::Level::Error;
    case LogLevel::Warning:
      return arvr::logging::Level::Warning;
    case LogLevel::Info:
      return arvr::logging::Level::Info;
    case LogLevel::Debug:
      return arvr::logging::Level::Debug;
    case LogLevel::Trace:
      return arvr::logging::Level::Trace;
    default:
      throw std::runtime_error("Unknown log level");
  }
}

#endif

[[nodiscard]] LogLevel stringToLogLevel(const std::string& logLevelStr) {
  std::string upperCaseLogLevelStr = logLevelStr;
  std::transform(
      upperCaseLogLevelStr.begin(),
      upperCaseLogLevelStr.end(),
      upperCaseLogLevelStr.begin(),
      ::toupper);
  if (upperCaseLogLevelStr == "TRACE") {
    return LogLevel::Trace;
  } else if (upperCaseLogLevelStr == "DEBUG") {
    return LogLevel::Debug;
  } else if (upperCaseLogLevelStr == "INFO") {
    return LogLevel::Info;
  } else if (upperCaseLogLevelStr == "WARNING") {
    return LogLevel::Warning;
  } else if (upperCaseLogLevelStr == "ERROR") {
    return LogLevel::Error;
  } else if (upperCaseLogLevelStr == "DISABLED") {
    return LogLevel::Disabled;
  } else {
    throw std::invalid_argument("Invalid log level: " + logLevelStr);
  }
}

} // namespace

std::map<std::string, LogLevel> logLevelMap() {
  return {
      {"Disabled", LogLevel::Disabled},
      {"Error", LogLevel::Error},
      {"Warning", LogLevel::Warning},
      {"Info", LogLevel::Info},
      {"Debug", LogLevel::Debug},
      {"Trace", LogLevel::Trace}};
}

void setLogLevel(LogLevel level) {
#if defined(MOMENTUM_WITH_XR_LOGGER)
  arvr::logging::getChannel(DEFAULT_LOG_CHANNEL).setLevel(toArvrLogLevel(level));
#else
  (void)level;
  std::cout << "Momentum was not built with XR_LOGGER, so setting log level has no effect.\n";
#endif
}

void setLogLevel(const std::string& levelStr) {
  setLogLevel(stringToLogLevel(levelStr));
}

} // namespace momentum
