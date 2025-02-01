/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if defined(AXEL_WITH_XR_PROFILER)

#include <arvr/libraries/profile_redirect/annotate.hpp>

#elif defined(AXEL_WITH_TRACY_PROFILER)

#include <tracy/Tracy.hpp>
#include <tracy/TracyC.h>
#include <stack>

#define _XR_PROFILE_CONCATENATE_DETAIL(x, y) x##y
#define _XR_PROFILE_CONCATENATE(x, y) _XR_PROFILE_CONCATENATE_DETAIL(x, y)
#define _XR_PROFILE_MAKE_UNIQUE(x) _XR_PROFILE_CONCATENATE(x, __LINE__)

#define XR_PROFILE_FUNCTION() ZoneScoped
#define XR_PROFILE_FUNCTION_CATEGORY(CATEGORY)
#define XR_PROFILE_EVENT(NAME) ZoneNamedN(_XR_PROFILE_MAKE_UNIQUE(__tracy), NAME, true)
#define XR_PROFILE_EVENT_DYNAMIC(NAME) ZoneTransientN(_XR_PROFILE_MAKE_UNIQUE(__tracy), NAME, true)
#define XR_PROFILE_CATEGORY(NAME, CATEGORY)
#define XR_PROFILE_PREPARE_PUSH_POP() std::stack<TracyCZoneCtx> ___tracy_xr_stack
#define XR_PROFILE_PUSH(NAME)                                                                    \
  static const struct ___tracy_source_location_data TracyConcat(                                 \
      __tracy_source_location, TracyLine) = {NAME, __func__, TracyFile, (uint32_t)TracyLine, 0}; \
  ___tracy_xr_stack.push(                                                                        \
      ___tracy_emit_zone_begin(&TracyConcat(__tracy_source_location, TracyLine), true));
#define XR_PROFILE_POP()                  \
  TracyCZoneEnd(___tracy_xr_stack.top()); \
  ___tracy_xr_stack.pop()
#define XR_PROFILE_THREAD(THREAD_NAME)           \
  {                                              \
    /* support both std::string and c strings */ \
    std::string threadNameStr(THREAD_NAME);      \
    TracyCSetThreadName(threadNameStr.c_str());  \
  }
#define XR_PROFILE_UPDATE()
#define XR_PROFILE_BEGIN_FRAME() FrameMark
#define XR_PROFILE_END_FRAME()
#define XR_PROFILE_METADATA(NAME, DATA)

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
