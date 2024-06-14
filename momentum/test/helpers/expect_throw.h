/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <gtest/gtest.h>

#include <string>

#define EXPECT_THROW_WITH_MESSAGE(callable, ExceptionType, matcher)            \
  {                                                                            \
    EXPECT_THROW(                                                              \
        try { callable(); } catch (const ExceptionType& e) {                   \
          EXPECT_THAT(std::string{e.what()}, matcher);                         \
          /* re-throw the current exception to use the gtest provided macro */ \
          throw;                                                               \
        },                                                                     \
        ExceptionType);                                                        \
  }
