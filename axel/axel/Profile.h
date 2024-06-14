/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if defined(AXEL_WITH_XR_PROFILE)

#include <arvr/libraries/profile_redirect/annotate.hpp>

#else

#define XR_PROFILE_FUNCTION()
#define XR_PROFILE_FUNCTION_CATEGORY(CATEGORY)
#define XR_PROFILE_EVENT(NAME)
#define XR_PROFILE_EVENT_DYNAMIC(NAME)
#define XR_PROFILE_CATEGORY(NAME, CATEGORY)
#define XR_PROFILE_PUSH(NAME)
#define XR_PROFILE_POP()
#define XR_PROFILE_THREAD(THREAD_NAME)
#define XR_PROFILE_UPDATE()
#define XR_PROFILE_BEGIN_FRAME()
#define XR_PROFILE_END_FRAME()
#define XR_PROFILE_METADATA(NAME, DATA)

#endif
