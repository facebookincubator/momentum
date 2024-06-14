/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/common/filesystem.h>

#include <functional>

namespace momentum::test {

/// Returns a path to an unique temporary path.
filesystem::path getUniqueTemporaryPath(
    const filesystem::path& tmp_dir,
    const std::function<bool(const filesystem::path&)>& validationFunc,
    const std::string& prefix = "",
    const std::string& extension = "");

/// Manages a unique temporary directory using RAII.
///
/// When instantiated, this class will create a temporary directory that may be accessed using the
/// path() method. When the object is destroyed again, the directory and all its contents will be
/// removed recursively.
class UniqueTemporaryDirectory {
 public:
  explicit UniqueTemporaryDirectory(
      const std::string& prefix = "",
      const filesystem::path& tmp_dir = filesystem::temp_directory_path());

  UniqueTemporaryDirectory(const UniqueTemporaryDirectory&) = delete;
  UniqueTemporaryDirectory(UniqueTemporaryDirectory&& /*other*/) noexcept;
  UniqueTemporaryDirectory& operator=(const UniqueTemporaryDirectory&) = delete;
  UniqueTemporaryDirectory& operator=(UniqueTemporaryDirectory&& /*other*/) noexcept;

  ~UniqueTemporaryDirectory();

  [[nodiscard]] const filesystem::path& path() const;

  [[nodiscard]] std::string string() const;

  filesystem::path operator/(const filesystem::path& path) const;

  void makePermanent();
  void makeTemporary();
  [[nodiscard]] bool isPermanent() const;

 private:
  filesystem::path path_;
  bool isPermanent_{false};
};
} // namespace momentum::test
