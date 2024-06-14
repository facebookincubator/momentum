/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace momentum {

/// Returns a copy of the string from which all the characters in whiteChars
/// at the beginning or at the end of the string have been removed.
/// @tparam StringType: The type of the string to trim (e.g., std::string, std::string_view)
/// @param text: The string to trim
/// @param whiteChars: A series of 1-byte characters to remove
/// @return The trimmed string
template <typename StringType>
[[nodiscard]] StringType trim(const StringType& text, const char* whiteChars = " \t");

/// Returns a copy of the string from which all the characters in whiteChars
/// at the beginning or at the end of the string have been removed.
/// @param text: The string to trim (null-terminated)
/// @param whiteChars: A series of 1-byte characters to remove
/// @return The trimmed string
[[nodiscard]] std::string trim(const char* text, const char* whiteChars = " \t");

/// Tokenize a string using the specified delimiters.
///
/// @param inputString The string to tokenize.
/// @param delimiters The string containing delimiter characters.
/// @param trim Whether to trim spaces around the tokens (default: true).
/// @return A vector of tokenized strings.
std::vector<std::string> tokenize(
    const std::string& inputString,
    const std::string& delimiters = " \t\r\n",
    bool trim = true);

/// Tokenize a string_view using the specified delimiters.
///
/// @param inputString The string_view to tokenize.
/// @param delimiters The string_view containing delimiter characters.
/// @param trim Whether to trim spaces around the tokens (default: true).
/// @return A vector of tokenized string_views.
std::vector<std::string_view>
tokenize(std::string_view inputString, std::string_view delimiters = " \t\r\n", bool trim = true);

} // namespace momentum
