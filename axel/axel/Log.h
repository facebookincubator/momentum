/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if defined(AXEL_WITH_XR_LOGGER)

#include <logging/Log.h>

#else

#define XR_LOGT(...)
#define XR_LOGD(...)
#define XR_LOGI(...)
#define XR_LOGW(...)
#define XR_LOGE(...)

#endif
