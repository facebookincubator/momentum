/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/common/filesystem.h>

namespace momentum::test {

/// Manages a unique temporary file.
///
/// When instantiated, this class will create a temporary file that may be accessed using the
/// path() method. When the object is destroyed again, the file will be removed.
class UniqueTemporaryFile {
 public:
  explicit UniqueTemporaryFile(
      const std::string& prefix = "",
      const std::string& extension = "",
      const filesystem::path& tmp_dir = filesystem::temp_directory_path());

  UniqueTemporaryFile(const UniqueTemporaryFile&) = delete;
  UniqueTemporaryFile(UniqueTemporaryFile&& other) noexcept;
  UniqueTemporaryFile& operator=(const UniqueTemporaryFile&) = delete;
  UniqueTemporaryFile& operator=(UniqueTemporaryFile&& /*other*/) noexcept;

  ~UniqueTemporaryFile();

  [[nodiscard]] const filesystem::path& path() const;

  [[nodiscard]] std::string string() const;

 private:
  filesystem::path path_;
};

} // namespace momentum::test
