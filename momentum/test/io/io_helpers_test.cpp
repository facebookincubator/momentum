/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/test/io/io_helpers.h"

#include <gtest/gtest.h>

using namespace momentum;

TEST(IoHelpersTest, EnvVarNonexistent) {
  std::optional<std::string> value = GetEnvVar("NONEXISTENT_VAR");
  EXPECT_FALSE(value.has_value());
}

TEST(IoHelpersTest, EnvVarExisting) {
  std::string expected_value = "test_value";
  SetEnvVar("TEST_VAR", expected_value);
  std::optional<std::string> value = GetEnvVar("TEST_VAR");
  ASSERT_TRUE(value.has_value());
  EXPECT_EQ(*value, expected_value);
}

TEST(IoHelpersTest, DISABLED_EnvVarEmpty) {
  // Windows handle empty environment variable differently and will fail the test.
  std::string expected_value;
  SetEnvVar("TEST_VAR", expected_value);
  std::optional<std::string> value = GetEnvVar("TEST_VAR");
  ASSERT_TRUE(value.has_value());
  EXPECT_EQ(*value, expected_value);
}

TEST(IoHelpersTest, EnvVarVeryLongValue) {
  // This tests an environment variable with a very long value
  std::string expected_value(10000, 'x');
  SetEnvVar("TEST_VAR", expected_value);
  std::optional<std::string> value = GetEnvVar("TEST_VAR");
  ASSERT_TRUE(value.has_value());
  EXPECT_EQ(*value, expected_value);
}

TEST(IoHelpersTest, EnvVarMultipleEqualSigns) {
  // This tests an environment variable with multiple equal signs
  std::string expected_value = "test=value=test";
  SetEnvVar("TEST_VAR", expected_value);
  std::optional<std::string> value = GetEnvVar("TEST_VAR");
  ASSERT_TRUE(value.has_value());
  EXPECT_EQ(*value, expected_value);
}

#if defined(_WIN32)
TEST(IoHelpersTest, EnvVarWindowsPath) {
  // This tests an environment variable with a funky Windows file path
  std::string expected_value = R"(C:\Program Files\Test\test.exe)";
  SetEnvVar("TEST_VAR", expected_value);
  std::optional<std::string> value = GetEnvVar("TEST_VAR");
  ASSERT_TRUE(value.has_value());
  EXPECT_EQ(*value, expected_value);
}

TEST(IoHelpersTest, EnvVarRemoteMount) {
  // This tests an environment variable with a remote mount path
  std::string expected_value = R"(\\remote\path\to\file)";
  SetEnvVar("TEST_VAR", expected_value);
  std::optional<std::string> value = GetEnvVar("TEST_VAR");
  ASSERT_TRUE(value.has_value());
  EXPECT_EQ(*value, expected_value);
}

TEST(IoHelpersTest, EnvVarCornerCases) {
  // This tests various corner cases of file paths on Windows
  std::string expected_value1 = R"(C:\Program Files\Test\test.exe)";
  std::string expected_value2 = R"(C:\Program Files\Test)";
  std::string expected_value3 = R"(\test.exe)";
  std::string expected_value4 = R"(\)";
  SetEnvVar("TEST_VAR1", expected_value1);
  SetEnvVar("TEST_VAR2", expected_value2);
  SetEnvVar("TEST_VAR3", expected_value3);
  SetEnvVar("TEST_VAR4", expected_value4);
  std::optional<std::string> value1 = GetEnvVar("TEST_VAR1");
  std::optional<std::string> value2 = GetEnvVar("TEST_VAR2");
  std::optional<std::string> value3 = GetEnvVar("TEST_VAR3");
  std::optional<std::string> value4 = GetEnvVar("TEST_VAR4");
  ASSERT_TRUE(value1.has_value());
  ASSERT_TRUE(value2.has_value());
  ASSERT_TRUE(value3.has_value());
  ASSERT_TRUE(value4.has_value());
  EXPECT_EQ(*value1, expected_value1);
  EXPECT_EQ(*value2, expected_value2);
  EXPECT_EQ(*value3, expected_value3);
  EXPECT_EQ(*value4, expected_value4);
}
#endif

TEST(IoHelpersTest, GetPIDReturnsNonZero) {
  ASSERT_NE(GetPID(), 0);
}

TEST(IoHelpersTest, TemporaryDirectoryExists) {
  auto temp_dir = temporaryDirectory();
  ASSERT_TRUE(filesystem::exists(temp_dir));
}

TEST(IoHelpersTest, TemporaryFileCreation) {
  std::string prefix = "test_prefix_";
  std::string extension = ".txt";

  auto temp_file = temporaryFile(prefix, extension);
  auto temp_file_path = temp_file.path();

  EXPECT_TRUE(temp_file_path.string().find(prefix) != std::string::npos);
  EXPECT_TRUE(temp_file_path.string().find(std::to_string(GetPID())) != std::string::npos);
  EXPECT_TRUE(temp_file_path.string().find(extension) != std::string::npos);
}

TEST(IoHelpersTest, TemporaryDirectoryCreation) {
  std::string prefix = "test_prefix_";

  auto temp_dir = temporaryDirectory(prefix);
  auto temp_dir_path = temp_dir.path();

  EXPECT_TRUE(temp_dir_path.string().find(prefix) != std::string::npos);
  EXPECT_TRUE(temp_dir_path.string().find(std::to_string(GetPID())) != std::string::npos);
  EXPECT_TRUE(filesystem::exists(temp_dir_path));
}
