/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/test/helpers/unique_temporary_file.h"

#include "momentum/test/helpers/unique_temporary_directory.h"

namespace momentum::test {

UniqueTemporaryFile::UniqueTemporaryFile(
    const std::string& prefix /* = "" */,
    const std::string& extension /* = "" */,
    const filesystem::path& tmp_dir /* = filesystem::temp_directory_path() */)
    : path_(getUniqueTemporaryPath(
          tmp_dir,
          [](const filesystem::path& path) { return !filesystem::exists(path); },
          prefix,
          extension)) {}

UniqueTemporaryFile::UniqueTemporaryFile(UniqueTemporaryFile&& other) noexcept
    : path_(std::move(other.path_)) {}

UniqueTemporaryFile& UniqueTemporaryFile::operator=(UniqueTemporaryFile&& other) noexcept {
  filesystem::remove_all(path_);
  path_ = std::move(other.path_);
  return *this;
}

UniqueTemporaryFile::~UniqueTemporaryFile() {
  std::error_code ec;
  filesystem::remove_all(path_, ec);
}

const filesystem::path& UniqueTemporaryFile::path() const {
  return path_;
}

std::string UniqueTemporaryFile::string() const {
  return path_.string();
}

} // namespace momentum::test
