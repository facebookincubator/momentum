/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/test/helpers/unique_temporary_directory.h>
#include <momentum/test/helpers/unique_temporary_file.h>

#include <optional>
#include <string>
#include <string_view>

namespace momentum {

using TemporaryFile = test::UniqueTemporaryFile;
using TemporaryDirectory = test::UniqueTemporaryDirectory;

/// Returns the process identifier (PID) of the current process.
[[nodiscard]] std::int64_t GetPID();

/// This is a portable function to read an environment variable.
///
/// It returns std::nullopt if the environment variable is not set.
[[nodiscard]] std::optional<std::string> GetEnvVar(std::string_view name);

/// Sets the value of the specified environment variable.
bool SetEnvVar(std::string_view name, std::string_view value);

/// Returns the temporary directory all temporary files will be created in
[[nodiscard]] filesystem::path temporaryDirectory();

/// Create a temporary file name with the given prefix and suffix
/// returns the path to a file in the system designated temporary directory where the
/// filename is of the form prefix_pid_n.suffix where pid is the process id, and n is
/// the number of times this function has been invoked.
/// The file will not be created automatically.
/// If it exists, the file will be deleted after TemporaryFile goes out of scope.
[[nodiscard]] TemporaryFile temporaryFile(const std::string& prefix, const std::string& extension);

/// Create a temporary directory with the given prefix
/// returns the path to a directory in the system designated temporary directory where the
/// filename is of the form prefix_pid_n where pid is the process id, and n is
/// the number of times this function has been invoked.
/// The directory is created automatically and will be deleted after the return value goes out of
/// scope.
[[nodiscard]] TemporaryDirectory temporaryDirectory(const std::string& prefix);

} // namespace momentum
