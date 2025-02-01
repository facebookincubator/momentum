/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if defined(MOMENTUM_WITH_XR_PROFILER)

#include <arvr/libraries/profile_redirect/annotate.hpp>

#define MT_PROFILE_FUNCTION() XR_PROFILE_FUNCTION()
#define MT_PROFILE_FUNCTION_CATEGORY(CATEGORY) XR_PROFILE_FUNCTION_CATEGORY(CATEGORY)
#define MT_PROFILE_EVENT(NAME) XR_PROFILE_EVENT(NAME)
#define MT_PROFILE_EVENT_DYNAMIC(NAME) XR_PROFILE_EVENT_DYNAMIC(NAME)
#define MT_PROFILE_CATEGORY(NAME, CATEGORY) XR_PROFILE_CATEGORY(NAME, CATEGORY)
#define MT_PROFILE_PREPARE_PUSH_POP() XR_PROFILE_PREPARE_PUSH_POP()
#define MT_PROFILE_PUSH(NAME) XR_PROFILE_PUSH(NAME)
#define MT_PROFILE_POP() XR_PROFILE_POP()
#define MT_PROFILE_THREAD(THREAD_NAME) XR_PROFILE_THREAD(THREAD_NAME)
#define MT_PROFILE_UPDATE() XR_PROFILE_UPDATE()
#define MT_PROFILE_BEGIN_FRAME() XR_PROFILE_BEGIN_FRAME()
#define MT_PROFILE_END_FRAME() XR_PROFILE_END_FRAME()
#define MT_PROFILE_METADATA(NAME, DATA) XR_PROFILE_METADATA(NAME, DATA)

#elif defined(MOMENTUM_WITH_TRACY_PROFILER)

#include <tracy/Tracy.hpp>
#include <tracy/TracyC.h>
#include <stack>

#define _MT_PROFILE_CONCATENATE_DETAIL(x, y) x##y
#define _MT_PROFILE_CONCATENATE(x, y) _MT_PROFILE_CONCATENATE_DETAIL(x, y)
#define _MT_PROFILE_MAKE_UNIQUE(x) _MT_PROFILE_CONCATENATE(x, __LINE__)

#define MT_PROFILE_FUNCTION() ZoneScoped
#define MT_PROFILE_FUNCTION_CATEGORY(CATEGORY)
#define MT_PROFILE_EVENT(NAME) ZoneNamedN(_MT_PROFILE_MAKE_UNIQUE(__tracy), NAME, true)
#define MT_PROFILE_EVENT_DYNAMIC(NAME) ZoneTransientN(_MT_PROFILE_MAKE_UNIQUE(__tracy), NAME, true)
#define MT_PROFILE_CATEGORY(NAME, CATEGORY)
#define MT_PROFILE_PREPARE_PUSH_POP() std::stack<TracyCZoneCtx> ___tracy_xr_stack
#define MT_PROFILE_PUSH(NAME)                                                                    \
  static const struct ___tracy_source_location_data TracyConcat(                                 \
      __tracy_source_location, TracyLine) = {NAME, __func__, TracyFile, (uint32_t)TracyLine, 0}; \
  ___tracy_xr_stack.push(                                                                        \
      ___tracy_emit_zone_begin(&TracyConcat(__tracy_source_location, TracyLine), true));
#define MT_PROFILE_POP()                  \
  TracyCZoneEnd(___tracy_xr_stack.top()); \
  ___tracy_xr_stack.pop()
#define MT_PROFILE_THREAD(THREAD_NAME)           \
  {                                              \
    /* support both std::string and c strings */ \
    std::string threadNameStr(THREAD_NAME);      \
    TracyCSetThreadName(threadNameStr.c_str());  \
  }
#define MT_PROFILE_UPDATE()
#define MT_PROFILE_BEGIN_FRAME() FrameMark
#define MT_PROFILE_END_FRAME()
#define MT_PROFILE_METADATA(NAME, DATA)

#else

#define MT_PROFILE_FUNCTION()
#define MT_PROFILE_FUNCTION_CATEGORY(CATEGORY)
#define MT_PROFILE_EVENT(NAME)
#define MT_PROFILE_EVENT_DYNAMIC(NAME)
#define MT_PROFILE_CATEGORY(NAME, CATEGORY)
#define MT_PROFILE_PREPARE_PUSH_POP()
#define MT_PROFILE_PUSH(NAME)
#define MT_PROFILE_POP()
#define MT_PROFILE_THREAD(THREAD_NAME)
#define MT_PROFILE_UPDATE()
#define MT_PROFILE_BEGIN_FRAME()
#define MT_PROFILE_END_FRAME()
#define MT_PROFILE_METADATA(NAME, DATA)

#endif
