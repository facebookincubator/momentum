/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <fmt/format.h>

#include <stdexcept>

#define MT_THROW_(Exception, Expression, ...) \
  throw Exception {                           \
    ::fmt::format(Expression, ##__VA_ARGS__)  \
  }

#define MT_THROW(Expression, ...) MT_THROW_(::std::runtime_error, Expression, ##__VA_ARGS__)

#define MT_THROW_IF_(Condition, Exception, Expression, ...)    \
  if (Condition) {                                             \
    throw Exception{::fmt::format(Expression, ##__VA_ARGS__)}; \
  }

#define MT_THROW_IF(Condition, Expression, ...) \
  MT_THROW_IF_(Condition, ::std::runtime_error, Expression, ##__VA_ARGS__)
