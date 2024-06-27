# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.16.3)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

project(momentum)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

#===============================================================================
# Build dependencies
#===============================================================================

# Needed by mt_defs.cmake
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

include(GNUInstallDirs)
include(cmake/mt_defs.cmake)

#===============================================================================
# Build Options
#===============================================================================

option(BUILD_SHARED_LIBS "Build as shared libraries" ON)
option(MOMENTUM_BUILD_IO_FBX "Build with IO FBX" OFF)
option(MOMENTUM_BUILD_PYMOMENTUM "Build Python binding" OFF)
option(MOMENTUM_BUILD_TESTING "Enable building tests" OFF)
option(MOMENTUM_BUILD_EXAMPLES "Enable building examples" OFF)
option(MOMENTUM_USE_SYSTEM_RERUN_CPP_SDK "Use Rerun C++ SDK installed in system" OFF)

# TODO: Workaround for Pixi/conda packaging
option(MOMENTUM_BUILD_WITH_EZC3D "Build with ezc3d dependency" ON)
option(MOMENTUM_BUILD_WITH_OPENFBX "Build with OpenFBX dependency" ON)

#===============================================================================
# Find dependencies
#===============================================================================

# 3rd party
find_package(Ceres CONFIG REQUIRED)
find_package(CLI11 CONFIG REQUIRED)
find_package(Dispenso CONFIG REQUIRED)
find_package(drjit CONFIG REQUIRED)
find_package(Eigen3 3.4.0 CONFIG REQUIRED)
if(MOMENTUM_BUILD_WITH_EZC3D)
  find_package(ezc3d CONFIG REQUIRED)
endif()
find_package(Microsoft.GSL CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(fx-gltf CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
if(MOMENTUM_BUILD_WITH_OPENFBX)
  find_package(openfbx CONFIG REQUIRED)
endif()
find_package(re2 MODULE REQUIRED)
find_package(Sophus CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
if(MOMENTUM_USE_SYSTEM_RERUN_CPP_SDK)
  find_package(rerun_sdk CONFIG REQUIRED)
else()
  include(FetchContent)
  FetchContent_Declare(rerun_sdk
    URL https://github.com/rerun-io/rerun/releases/download/0.16.0/rerun_cpp_sdk.zip
  )
  FetchContent_MakeAvailable(rerun_sdk)
endif()

# 1st party
# TODO: Make Axel as a separate project and change to find_package(axel CONFIG REQUIRED)
add_subdirectory(axel)

#===============================================================================
# Compiler Settings
#===============================================================================

if(MOMENTUM_BUILD_TESTING)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

if(MSVC)
  add_compile_options(/bigobj)
endif()

#===============================================================================
# Build momentum
#===============================================================================

mt_library(
  NAME common
  HEADERS_VARS common_public_headers
  SOURCES_VARS common_sources
  PUBLIC_LINK_LIBRARIES
    fmt::fmt
    spdlog::spdlog
  PUBLIC_COMPILE_DEFINITIONS
    MOMENTUM_WITH_SPDLOG=1
)

mt_library(
  NAME simd
  HEADERS_VARS simd_public_headers
  PUBLIC_LINK_LIBRARIES
    Eigen3::Eigen
    drjit
)

mt_library(
  NAME fmt_eigen
  HEADERS_VARS fmt_eigen_public_headers
  PUBLIC_LINK_LIBRARIES
    fmt::fmt
    Eigen3::Eigen
)

mt_library(
  NAME online_qr
  HEADERS_VARS online_qr_public_headers
  SOURCES_VARS online_qr_sources
  PUBLIC_LINK_LIBRARIES
    common
    fmt_eigen
    Eigen3::Eigen
    Microsoft.GSL::GSL
)

mt_library(
  NAME math
  HEADERS_VARS math_public_headers
  SOURCES_VARS math_sources
  PUBLIC_LINK_LIBRARIES
    common
    fmt_eigen
    online_qr
    Eigen3::Eigen
    Microsoft.GSL::GSL
    Sophus::Sophus
)

mt_library(
  NAME simd_generalized_loss
  HEADERS_VARS simd_generalized_loss_public_headers
  SOURCES_VARS simd_generalized_loss_sources
  PUBLIC_LINK_LIBRARIES
    math
    simd
)

mt_library(
  NAME skeleton
  HEADERS_VARS skeleton_public_headers
  SOURCES_VARS skeleton_sources
  PUBLIC_LINK_LIBRARIES
    common
    math
    Eigen3::Eigen
    Microsoft.GSL::GSL
)

mt_library(
  NAME character
  HEADERS_VARS character_public_headers
  SOURCES_VARS character_sources
  PUBLIC_LINK_LIBRARIES
    common
    math
    skeleton
    axel
    Dispenso::dispenso
    Eigen3::Eigen
    Microsoft.GSL::GSL
)

mt_library(
  NAME solver
  HEADERS_VARS solver_public_headers
  SOURCES_VARS solver_sources
  PUBLIC_LINK_LIBRARIES
    common
    math
    Eigen3::Eigen
    Microsoft.GSL::GSL
)

mt_library(
  NAME character_solver
  HEADERS_VARS character_solver_public_headers
  SOURCES_VARS character_solver_sources
  PUBLIC_LINK_LIBRARIES
    character
    solver
    axel
    Eigen3::Eigen
    Microsoft.GSL::GSL
  PRIVATE_LINK_LIBRARIES
    Dispenso::dispenso
)

mt_library(
  NAME simd_constraints
  HEADERS_VARS simd_constraints_public_headers
  SOURCES_VARS simd_constraints_sources
  PUBLIC_LINK_LIBRARIES
    character_solver
    simd
)

mt_library(
  NAME character_sequence_solver
  HEADERS_VARS character_sequence_solver_public_headers
  SOURCES_VARS character_sequence_solver_sources
  PUBLIC_LINK_LIBRARIES
    character_solver
    Eigen3::Eigen
    Microsoft.GSL::GSL
  PRIVATE_LINK_LIBRARIES
    Dispenso::dispenso
)

mt_library(
  NAME momentum
  PUBLIC_LINK_LIBRARIES
    common
    simd
    online_qr
    math
    solver
    skeleton
    character
    character_solver
    simd_constraints
    character_sequence_solver
)

mt_library(
  NAME diff_ik
  HEADERS_VARS diff_ik_public_headers
  SOURCES_VARS diff_ik_sources
  PUBLIC_LINK_LIBRARIES
    character_solver
    Ceres::ceres
    Dispenso::dispenso
)

mt_library(
  NAME io_common
  HEADERS_VARS io_common_public_headers
  SOURCES_VARS io_common_sources
  PUBLIC_LINK_LIBRARIES
    Microsoft.GSL::GSL
)

mt_library(
  NAME io_skeleton
  HEADERS_VARS io_skeleton_public_headers
  SOURCES_VARS io_skeleton_sources
  PUBLIC_LINK_LIBRARIES
    character
  PRIVATE_LINK_LIBRARIES
    io_common
    re2::re2
)

mt_library(
  NAME io_shape
  HEADERS_VARS io_shape_public_headers
  SOURCES_VARS io_shape_sources
  PUBLIC_LINK_LIBRARIES
    character
)

if(TARGET OpenFBX)
  mt_library(
    NAME io_openfbx
    HEADERS_VARS
      io_openfbx_public_headers
    PRIVATE_HEADERS_VARS
      io_openfbx_private_headers
    SOURCES_VARS io_openfbx_sources
    PRIVATE_LINK_LIBRARIES
      io_common
      io_skeleton
      OpenFBX
  )
endif()

if(MOMENTUM_BUILD_IO_FBX)
  if(DEFINED ENV{FBXSDK_PATH})
    set(fbxsdk_path $ENV{FBXSDK_PATH})

    if(NOT EXISTS "${fbxsdk_path}")
      message(FATAL_ERROR "FBXSDK_PATH does not exist: ${fbxsdk_path}")
    endif()

    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
      set(io_fbx_sources_var io_fbx_sources_windows)
      set(fbxsdk_libs
        "${fbxsdk_path}/lib/vs2017/x64/release/libfbxsdk-md.lib"
        "${fbxsdk_path}/lib/vs2017/x64/release/libxml2-md.lib"
        "${fbxsdk_path}/lib/vs2017/x64/release/zlib-md.lib"
      )
    else()
      message(FATAL_ERROR "Unsupported platform")
    endif()

    mt_library(
      NAME fbxsdk
      PUBLIC_INCLUDE_DIRECTORIES
        "${fbxsdk_path}/include"
      PUBLIC_LINK_LIBRARIES
        ${fbxsdk_libs}
    )

    set(io_fbx_private_link_libraries fbxsdk)
  else()
    message(WARNING "MOMENTUM_BUILD_IO_FBX is enabled, but the FBXSDK_PATH environment variable is not defined. The build will proceed without the FBX SDK. Please download the required FBX SDK 2019.2 from the Autodesk Developer Network: https://aps.autodesk.com/developer/overview/fbx-sdk. After installation, set FBXSDK_PATH to the installation directory, for example, 'C:\\Program Files\\Autodesk\\FBX\\FBX SDK\\2019.2'.")
    set(io_fbx_sources_var io_fbx_sources_unsupported)
    set(io_fbx_private_link_libraries )
  endif()
else()
  set(io_fbx_sources_var io_fbx_sources_unsupported)
  set(io_fbx_private_link_libraries )
endif()

mt_library(
  NAME io_fbx
  HEADERS_VARS
    io_fbx_public_headers
  PRIVATE_HEADERS_VARS
    io_fbx_private_headers
  SOURCES_VARS
    ${io_fbx_sources_var}
  PRIVATE_LINK_LIBRARIES
    io_skeleton
    ${io_fbx_private_link_libraries}
)

mt_library(
  NAME io_gltf
  HEADERS_VARS
    io_gltf_public_headers
  PRIVATE_HEADERS_VARS
    io_gltf_private_headers
  SOURCES_VARS io_gltf_sources
  PUBLIC_LINK_LIBRARIES
    character
    io_common
    io_skeleton
    fx-gltf::fx-gltf
  PRIVATE_LINK_LIBRARIES
    nlohmann_json::nlohmann_json
)

mt_library(
  NAME io_motion
  HEADERS_VARS io_motion_public_headers
  SOURCES_VARS io_motion_sources
  PUBLIC_LINK_LIBRARIES
    character
)

if(TARGET ezc3d)
  mt_library(
    NAME io_marker
    HEADERS_VARS io_marker_public_headers
    SOURCES_VARS io_marker_sources
    PUBLIC_LINK_LIBRARIES
      character
    PRIVATE_LINK_LIBRARIES
      io_common
      io_gltf
      ezc3d
  )
endif()

if(TARGET io_marker AND TARGET io_openfbx)
  mt_library(
    NAME io
    HEADERS_VARS io_public_headers
    SOURCES_VARS io_sources
    PUBLIC_LINK_LIBRARIES
      io_common
      io_fbx
      io_gltf
      io_motion
      io_marker
      io_openfbx
      io_shape
      io_skeleton
  )
endif()

mt_library(
  NAME marker_tracker
  HEADERS_VARS
    marker_tracker_public_headers
  SOURCES_VARS
    marker_tracker_sources
  PUBLIC_LINK_LIBRARIES
    character_solver
  PRIVATE_LINK_LIBRARIES
    character_sequence_solver
)

if(TARGET io)
  mt_library(
    NAME app_utils
    HEADERS_VARS
      app_utils_public_headers
    SOURCES_VARS
      app_utils_sources
    PUBLIC_LINK_LIBRARIES
      character
      marker_tracker
      CLI11::CLI11
    PRIVATE_LINK_LIBRARIES
      io
  )
endif()

if(TARGET io_marker AND TARGET app_utils)
  mt_library(
    NAME process_markers
    HEADERS_VARS
      process_markers_public_headers
    SOURCES_VARS
      process_markers_sources
    PUBLIC_LINK_LIBRARIES
      app_utils
      marker_tracker
      skeleton
    PRIVATE_LINK_LIBRARIES
      io_marker
  )
endif()

mt_library(
  NAME rerun_eigen_adapters
  HEADERS_VARS
    rerun_eigen_adapters_public_headers
  PUBLIC_LINK_LIBRARIES
    Eigen3::Eigen
    rerun_sdk
  EXCLUDE_FROM_INSTALL
)

mt_library(
  NAME rerun
  HEADERS_VARS
    rerun_public_headers
  SOURCES_VARS
    rerun_sources
  PUBLIC_LINK_LIBRARIES
    character
    rerun_sdk
  PRIVATE_LINK_LIBRARIES
    rerun_eigen_adapters
    axel
  EXCLUDE_FROM_INSTALL
)

#===============================================================================
# Tests
#===============================================================================

if(MOMENTUM_BUILD_TESTING)
  find_package(Boost REQUIRED)

  enable_testing()
  mt_setup_gtest()

  mt_library(
    NAME test_helpers
    HEADERS_VARS test_helpers_public_headers
    SOURCES_VARS test_helpers_sources
    PUBLIC_LINK_LIBRARIES
      common
      gtest
    PRIVATE_LINK_LIBRARIES
      Boost::boost
    EXCLUDE_FROM_INSTALL
  )

  mt_library(
    NAME character_test_helper
    HEADERS_VARS character_test_helper_public_headers
    SOURCES_VARS character_test_helper_sources
    PUBLIC_LINK_LIBRARIES
      character
    PRIVATE_LINK_LIBRARIES
      gmock
      gtest
    EXCLUDE_FROM_INSTALL
  )

  mt_library(
    NAME solver_test_helper
    HEADERS_VARS solver_test_helper_public_headers
    SOURCES_VARS solver_test_helper_sources
    PUBLIC_LINK_LIBRARIES
      solver
    EXCLUDE_FROM_INSTALL
  )

  mt_library(
    NAME error_function_helper
    HEADERS_VARS error_function_helper_public_headers
    SOURCES_VARS error_function_helper_sources
    PUBLIC_LINK_LIBRARIES
      character
      character_solver
    PRIVATE_LINK_LIBRARIES
      gmock
      gtest
    EXCLUDE_FROM_INSTALL
  )

  # set(io_test_helper_libs )
  # if(APPLE)
  #   set(io_test_helper_sources io_test_helper_sources_macos)
  #   list(APPEND io_test_helper_libs "-framework Foundation")
  # elseif(UNIX)
  #   set(io_test_helper_sources io_test_helper_sources_linux)
  # elseif(WIN32)
  #   set(io_test_helper_sources io_test_helper_sources_windows)
  # else()
  #   message(FATAL_ERROR "Unsupported platform")
  # endif()

  # mt_library(
  #   NAME io_test_helper
  #   HEADERS_VARS io_test_helper_public_headers
  #   SOURCES_VARS ${io_test_helper_sources}
  #   PUBLIC_LINK_LIBRARIES
  #     test_helpers
  #     Microsoft.GSL::GSL
  #   PRIVATE_LINK_LIBRARIES
  #     ${io_test_helper_libs}
  #   EXCLUDE_FROM_INSTALL
  # )

  mt_test(
    NAME common_test
    SOURCES_VARS common_test_sources
    LINK_LIBRARIES common
  )

  mt_test(
    NAME simd_test
    SOURCES_VARS simd_test_sources
    LINK_LIBRARIES simd
  )

  mt_test(
    NAME online_qr_test
    SOURCES_VARS online_qr_test_sources
    LINK_LIBRARIES online_qr
  )

  mt_test(
    NAME math_test
    SOURCES_VARS math_test_sources
    LINK_LIBRARIES math
  )

  mt_test(
    NAME simd_generalized_loss_test
    SOURCES_VARS simd_generalized_loss_test_sources
    LINK_LIBRARIES simd_generalized_loss
  )

  # TODO: Add solver_test

  mt_test(
    NAME character_test
    SOURCES_VARS character_test_sources
    LINK_LIBRARIES
      character
      character_test_helper
      test_helpers
  )

  mt_test(
    NAME character_solver_test
    SOURCES_VARS character_solver_test_sources
    LINK_LIBRARIES
      character_solver
      character_test_helper
      error_function_helper
      solver_test_helper
  )

  if(NOT APPLE) # TODO: Fix
    mt_test(
      NAME simd_constraints_test
      SOURCES_VARS simd_constraints_test_sources
      LINK_LIBRARIES
        character_solver
        character_test_helper
        error_function_helper
        simd_constraints
    )
  endif()

  mt_test(
    NAME character_sequence_solver_test
    SOURCES_VARS character_sequence_solver_test_sources
    LINK_LIBRARIES
      character_sequence_solver
      character_test_helper
      solver_test_helper
  )

  # TODO: Fix test failures on macOS
  if(NOT APPLE)
    mt_test(
      NAME diff_ik_test
      HEADERS_VARS diff_ik_test_headers
      SOURCES_VARS diff_ik_test_sources
      LINK_LIBRARIES
        character_test_helper
        diff_ik
        io_gltf
        io_skeleton
    )
  endif()

  # mt_test(
  #   NAME io_common_test
  #   LINK_LIBRARIES io_common
  #   SOURCES_VARS io_common_test_sources
  # )

  # TODO: Add io_gltf_test

  # mt_test(
  #   NAME io_marker_test
  #   SOURCES_VARS io_marker_test_sources
  #   LINK_LIBRARIES
  #     io_marker
  #     io_test_helper
  #   ENV
  #     "TEST_C3D_PATH=${CMAKE_SOURCE_DIR}/momentum/test/resources"
  # )
endif()

#===============================================================================
# Examples
#===============================================================================

if(MOMENTUM_BUILD_EXAMPLES)
  mt_executable(
    NAME hello_world
    SOURCES_VARS hello_world_sources
    LINK_LIBRARIES
      momentum
  )

  if(TARGET io)
    mt_executable(
      NAME convert_model
      SOURCES_VARS convert_model_sources
      LINK_LIBRARIES
        momentum io CLI11::CLI11
    )
  endif()

  mt_executable(
    NAME glb_viewer
    SOURCES_VARS glb_viewer_sources
    LINK_LIBRARIES
      io_gltf
      rerun
      CLI11::CLI11
      rerun_sdk
  )

  if(TARGET io_openfbx)
    mt_executable(
      NAME fbx_viewer
      SOURCES_VARS fbx_viewer_sources
      LINK_LIBRARIES
        io_openfbx
        rerun
        CLI11::CLI11
        rerun_sdk
    )
  endif()

  if(TARGET io_marker)
    mt_executable(
      NAME c3d_viewer
      SOURCES_VARS c3d_viewer_sources
      LINK_LIBRARIES
        io_marker
        rerun
        CLI11::CLI11
        rerun_sdk
    )
  endif()

  if(TARGET process_markers)
    mt_executable(
      NAME process_markers_app
      SOURCES_VARS process_markers_app_sources
      LINK_LIBRARIES
        app_utils
        common
        process_markers
    )
  endif()
endif()

#===============================================================================
# Install momentum
#===============================================================================

get_property(targets GLOBAL PROPERTY MOMENTUM_TARGETS)

# Generate required install artifacts for CMake buildsystem
include(CMakePackageConfigHelpers)
set(MOMENTUM_CONFIG_INPUT cmake/momentum-config.cmake.in)
set(MOMENTUM_CONFIG_OUTPUT ${CMAKE_BINARY_DIR}/cmake/momentum-config.cmake)
set(MOMENTUM_CONFIG_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
configure_package_config_file(
  ${MOMENTUM_CONFIG_INPUT}
  ${MOMENTUM_CONFIG_OUTPUT}
  INSTALL_DESTINATION ${MOMENTUM_CONFIG_INSTALL_DIR}
)
install(
  FILES ${MOMENTUM_CONFIG_OUTPUT}
  DESTINATION ${MOMENTUM_CONFIG_INSTALL_DIR}
)

# Install CMake modules
install(
  FILES cmake/Findre2.cmake
  DESTINATION ${MOMENTUM_CONFIG_INSTALL_DIR}
)

# Install headers
install(
  DIRECTORY momentum
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT headers
  FILES_MATCHING PATTERN "*.h"
  PATTERN "*/examples/*" EXCLUDE
  PATTERN "*/test/*" EXCLUDE
  PATTERN "*/website/*" EXCLUDE
)

# Install targets (lib)
install(
  TARGETS ${targets}
  EXPORT momentumTargets
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)
install(
  EXPORT momentumTargets
  NAMESPACE momentum::
  DESTINATION ${MOMENTUM_CONFIG_INSTALL_DIR}
)

#===============================================================================
# PyMomentum
#===============================================================================

if(MOMENTUM_BUILD_PYMOMENTUM)
  add_subdirectory(pymomentum)
endif()
