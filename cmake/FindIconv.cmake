# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This module defines
#   Iconv_FOUND       : Variable indicating if the iconv support was found.
#   Iconv_INCLUDE_DIRS: The directories containing the iconv headers.
#   Iconv_LIBRARIES   : The iconv libraries to be linked.
#   Iconv::Iconv       :IMPORTED target

execute_process(
  COMMAND xcrun --sdk macosx --show-sdk-path
  OUTPUT_VARIABLE XCODE_SDK_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_path(
  Iconv_INCLUDE_DIR "iconv.h"
  PATHS "${XCODE_SDK_PATH}/usr/include"
  PATH_SUFFIXES "include"
)

find_library(Iconv_LIBRARY NAMES iconv
  PATHS "${XCODE_SDK_PATH}/usr/lib"
  NO_DEFAULT_PATH
)
set(Iconv_LIBRARIES ${Iconv_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Iconv
  FAIL_MESSAGE DEFAULT_MGS
  REQUIRED_VARS Iconv_INCLUDE_DIR Iconv_LIBRARIES
)

if(Iconv_FOUND)
  add_library(Iconv::Iconv INTERFACE IMPORTED)
  set_target_properties(Iconv::Iconv PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${Iconv_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${Iconv_LIBRARIES}"
  )
endif()
