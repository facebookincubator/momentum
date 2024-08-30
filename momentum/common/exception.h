/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <fmt/format.h>

#include <stdexcept>

/// Throws an exception of a specified type with a formatted message.
/// @param Exception The type of exception to throw.
/// @param ... The message and format specifiers.
/// @code
///   MT_THROW_T(std::invalid_argument, "Invalid argument: {}", arg);
///   MT_THROW_T(std::bad_array_new_length); // No message
/// @endcode
#define MT_THROW_T(Exception, ...) ::momentum::detail::throwImpl<Exception>(__VA_ARGS__)

/// Throws a std::runtime_error with a formatted message.
/// @param ... The message and format specifiers.
/// @code
///   MT_THROW("Error occurred: {}", errorDetail);
/// @endcode
#define MT_THROW(...) MT_THROW_T(::momentum::detail::DefaultException, ##__VA_ARGS__)

/// Conditionally throws an exception of a specified type with a formatted message if the condition
/// is true.
/// @param Condition The condition to evaluate.
/// @param Exception The type of exception to throw.
/// @param ... The message and format specifiers.
/// @code
///   MT_THROW_IF_T(x > y, std::out_of_range, "x ({}) is greater than y ({})", x, y);
///   MT_THROW_IF_T(x > y, std::bad_array_new_length); // No message
/// @endcode
#define MT_THROW_IF_T(Condition, Exception, ...) \
  if (Condition) {                               \
    MT_THROW_T(Exception, ##__VA_ARGS__);        \
  }

/// Conditionally throws a std::runtime_error with a formatted message if the condition is true.
/// @param Condition The condition to evaluate.
/// @param ... The message and format specifiers.
/// @code
///   MT_THROW_IF(x == nullptr, "x cannot be nullptr");
/// @endcode
#define MT_THROW_IF(Condition, ...) \
  MT_THROW_IF_T(Condition, ::momentum::detail::DefaultException, ##__VA_ARGS__)

namespace momentum::detail {

using DefaultException = std::runtime_error;

// Helper function template to throw with formatted message
template <typename Exception = DefaultException, typename... Args>
[[noreturn]] void throwImpl(fmt::format_string<Args...> format, Args&&... args) {
  throw Exception{fmt::format(format, std::forward<Args>(args)...)};
}

// Overload for throwing exceptions that do not require any message or whose constructors do not
// take any arguments.
template <typename Exception>
[[noreturn]] void throwImpl() {
  throw Exception{};
}

} // namespace momentum::detail
