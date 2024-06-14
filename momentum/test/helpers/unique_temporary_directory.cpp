/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/test/helpers/unique_temporary_directory.h"

#include "momentum/common/checks.h"

#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <fmt/format.h>

#include <sstream>

namespace momentum::test {

namespace {

std::string getUuidString() {
  std::stringstream sstream;
  boost::uuids::random_generator generator;
  sstream << generator();
  return sstream.str();
}

} // namespace

filesystem::path getUniqueTemporaryPath(
    const filesystem::path& tmp_dir,
    const std::function<bool(const filesystem::path&)>& validationFunc,
    const std::string& prefix /* = "" */,
    const std::string& extension /* = "" */) {
  filesystem::path uniqueDirectoryPath;
  int counter = 0;
  do {
    filesystem::path name(fmt::format("{}{}", prefix, getUuidString()));
    if (!extension.empty()) {
      // can't use filesystem::replace_extension since prefix can contain dots
      name = fmt::format("{}.{}", name.string(), extension);
    }
    uniqueDirectoryPath = tmp_dir / name;
    MT_CHECK_LT(++counter, 10, "Failed to create unique temporary path in {} tries", counter);
  } while (!validationFunc(uniqueDirectoryPath));

  return uniqueDirectoryPath;
}

/// Returns a path to an unique temporary directory. The directory is already created when this
/// function returns.
filesystem::path getUniqueTemporaryDirectory(
    const std::string& prefix = "",
    const filesystem::path& tmp_dir = filesystem::temp_directory_path()) {
  return getUniqueTemporaryPath(
      tmp_dir,
      [](const filesystem::path& path) { return filesystem::create_directory(path); },
      prefix);
}

UniqueTemporaryDirectory::UniqueTemporaryDirectory(
    const std::string& prefix /* = "" */,
    const filesystem::path& tmp_dir /* = filesystem::temp_directory_path() */)
    : path_{getUniqueTemporaryDirectory(prefix, tmp_dir)} {}

UniqueTemporaryDirectory::UniqueTemporaryDirectory(UniqueTemporaryDirectory&& other) noexcept
    : path_(std::move(other.path_)), isPermanent_(other.isPermanent_) {}

UniqueTemporaryDirectory& UniqueTemporaryDirectory::operator=(
    UniqueTemporaryDirectory&& other) noexcept {
  filesystem::remove_all(path_);
  path_ = std::move(other.path_);
  return *this;
}

UniqueTemporaryDirectory::~UniqueTemporaryDirectory() {
  if (!isPermanent_) {
    std::error_code ec;
    filesystem::remove_all(path_, ec);
  }
}

const filesystem::path& UniqueTemporaryDirectory::path() const {
  return path_;
}

std::string UniqueTemporaryDirectory::string() const {
  return path_.string();
}

filesystem::path UniqueTemporaryDirectory::operator/(const filesystem::path& path) const {
  return path_ / path;
}

void UniqueTemporaryDirectory::makePermanent() {
  isPermanent_ = true;
}

void UniqueTemporaryDirectory::makeTemporary() {
  isPermanent_ = false;
}

bool UniqueTemporaryDirectory::isPermanent() const {
  return isPermanent_;
}

} // namespace momentum::test
