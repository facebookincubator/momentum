# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

find_package(re2 QUIET CONFIG)
if(re2_FOUND)
  message(STATUS "Found RE2 via CMake.")
  return()
endif()

find_package(PkgConfig QUIET)
if (PKG_CONFIG_FOUND)
  pkg_check_modules(PC_RE2 QUIET re2)
endif()

if(PC_RE2_FOUND)
  set(re2_FOUND true)
  add_library(re2::re2 INTERFACE IMPORTED)
  if(PC_RE2_INCLUDE_DIRS)
    set_property(
      TARGET re2::re2 PROPERTY
      INTERFACE_INCLUDE_DIRECTORIES "${PC_RE2_INCLUDE_DIRS}"
    )
  endif()
  if(PC_RE2_CFLAGS_OTHER)
    # Filter out the -std flag, which is handled by CMAKE_CXX_STANDARD.
    foreach(flag IN LISTS PC_RE2_CFLAGS_OTHER)
      if("${flag}" MATCHES "^-std=")
        list(REMOVE_ITEM PC_RE2_CFLAGS_OTHER "${flag}")
      endif()
    endforeach()
    set_property(
      TARGET re2::re2 PROPERTY
      INTERFACE_COMPILE_OPTIONS "${PC_RE2_CFLAGS_OTHER}"
    )
  endif()

  if(PC_RE2_LDFLAGS)
    set_property(
      TARGET re2::re2 PROPERTY
      INTERFACE_LINK_LIBRARIES "${PC_RE2_LDFLAGS}"
    )
  endif()

  message(STATUS "Found RE2 via pkg-config.")
  return()
endif()

if(re2_FIND_REQUIRED)
  message(FATAL_ERROR "Failed to find RE2.")
endif()
