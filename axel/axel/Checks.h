/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if defined(AXEL_WITH_XR_LOGGER)

#include <logging/Checks.h>

#else

#include <cassert>

// TODO: Support asserts with messages as XR_CHECK does
#define XR_CHECK(condition, ...) assert(condition)

#endif
