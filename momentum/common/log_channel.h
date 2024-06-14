/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if defined(MOMENTUM_WITH_XR_LOGGER)

// The file to include in header files. Used as XR_LOGCI(MOMENTUM_LOG_CHANNEL, "message");

#include <logging/Log.h>
#define MOMENTUM_LOG_CHANNEL "MOMENTUM"

#endif
