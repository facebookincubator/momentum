# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

if(NOT Python3_EXECUTABLE)
  message(FATAL_ERROR "Python3_EXECUTABLE is needed")
endif()

function(mt_append_filelist name rootdir vars_path_in vars_path_out outputvar)
  # configure_file adds its input to the list of CMAKE_RERUN dependencies
  configure_file(
    ${vars_path_in}
    ${vars_path_out}
  )
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
      "exec(open('${vars_path_in}').read());print(';'.join(['${rootdir}' + x for x in ${name}]))"
    WORKING_DIRECTORY "${rootdir}"
    RESULT_VARIABLE _retval
    OUTPUT_VARIABLE _tempvar
  )
  if(NOT _retval EQUAL 0)
    message(FATAL_ERROR "Failed to fetch filelist ${name} from build_variables.bzl")
  endif()
  string(REPLACE "\n" "" _tempvar "${_tempvar}")
  list(APPEND ${outputvar} ${_tempvar})
  set(${outputvar} "${${outputvar}}" PARENT_SCOPE)
endfunction()

function(mt_append_momentum_filelist name outputvar)
  mt_append_filelist(
    ${name}
    "${PROJECT_SOURCE_DIR}/momentum/"
    ${PROJECT_SOURCE_DIR}/cmake/build_variables.bzl
    ${PROJECT_BINARY_DIR}/cmake/build_variables.bzl
    ${outputvar}
  )
  set(${outputvar} "${${outputvar}}" PARENT_SCOPE)
endfunction()

function(mt_append_pymomentum_filelist name outputvar)
  mt_append_filelist(
    ${name}
    "${PROJECT_SOURCE_DIR}/pymomentum/"
    ${PROJECT_SOURCE_DIR}/pymomentum/cmake/build_variables.bzl
    ${PROJECT_BINARY_DIR}/pymomentum/cmake/build_variables.bzl
    ${outputvar}
  )
  set(${outputvar} "${${outputvar}}" PARENT_SCOPE)
endfunction()

function(mt_library)
  set(prefix _ARG)
  set(options
    EXCLUDE_FROM_INSTALL
  )
  set(oneValueArgs
    NAME
  )
  set(multiValueArgs
    HEADERS
    HEADERS_VARS
    PYMOMENTUM_HEADERS_VARS
    PRIVATE_HEADERS
    PRIVATE_HEADERS_VARS
    PYMOMENTUM_PRIVATE_HEADERS_VARS
    SOURCES
    SOURCES_VARS
    PYMOMENTUM_SOURCES_VARS
    PUBLIC_INCLUDE_DIRECTORIES
    PRIVATE_INCLUDE_DIRECTORIES
    PUBLIC_LINK_LIBRARIES
    PRIVATE_LINK_LIBRARIES
    PUBLIC_COMPILE_DEFINITIONS
    PRIVATE_COMPILE_DEFINITIONS
    PUBLIC_COMPILE_OPTIONS
    PRIVATE_COMPILE_OPTIONS
  )
  cmake_parse_arguments(
    "${prefix}"
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
  )

  foreach(var ${_ARG_HEADERS_VARS})
    mt_append_momentum_filelist("${var}" _ARG_HEADERS)
  endforeach()

  foreach(var ${_ARG_PYMOMENTUM_HEADERS_VARS})
    mt_append_pymomentum_filelist("${var}" _ARG_HEADERS)
  endforeach()

  foreach(var ${_ARG_PRIVATE_HEADERS_VARS})
    mt_append_momentum_filelist("${var}" _ARG_PRIVATE_HEADERS)
  endforeach()

  foreach(var ${_ARG_PYMOMENTUM_PRIVATE_HEADERS_VARS})
    mt_append_pymomentum_filelist("${var}" _ARG_PRIVATE_HEADERS)
  endforeach()

  foreach(var ${_ARG_SOURCES_VARS})
    mt_append_momentum_filelist("${var}" _ARG_SOURCES)
  endforeach()

  foreach(var ${_ARG_PYMOMENTUM_SOURCES_VARS})
    mt_append_pymomentum_filelist("${var}" _ARG_SOURCES)
  endforeach()

  if("${_ARG_SOURCES}" STREQUAL "")
    set(library_type INTERFACE)
    set(type INTERFACE)
  else()
    set(library_type )
    set(type PUBLIC)
  endif()

  add_library(${_ARG_NAME}
    ${library_type}
    ${_ARG_HEADERS}
    ${_ARG_SOURCES}
  )
  target_sources(${_ARG_NAME} PRIVATE
    ${_ARG_PRIVATE_HEADERS}
  )
  target_include_directories(${_ARG_NAME}
    ${type}
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
      $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> # TODO: Remove once momentum/pymomentum/pymomentum is moved to momentum/pymomentum
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
      ${_ARG_PUBLIC_INCLUDE_DIRECTORIES}
  )
  target_include_directories(${_ARG_NAME} PRIVATE ${_ARG_PRIVATE_INCLUDE_DIRECTORIES})
  target_compile_features(${_ARG_NAME} ${type} cxx_std_17)
  target_link_libraries(${_ARG_NAME} ${type} ${_ARG_PUBLIC_LINK_LIBRARIES})
  target_link_libraries(${_ARG_NAME} PRIVATE ${_ARG_PRIVATE_LINK_LIBRARIES})
  set_target_properties(${_ARG_NAME} PROPERTIES
    OUTPUT_NAME momentum_${_ARG_NAME}
    POSITION_INDEPENDENT_CODE ON
  )
  target_compile_definitions(${_ARG_NAME}
    ${type} ${_ARG_PUBLIC_COMPILE_DEFINITIONS}
    PRIVATE ${_ARG_PRIVATE_COMPILE_DEFINITIONS}
  )
  if(MOMENTUM_ENABLE_SIMD)
    if(MSVC)
      target_compile_options(${_ARG_NAME} ${type} "/arch:SSE2")
      target_compile_options(${_ARG_NAME} ${type} "/arch:AVX")
      target_compile_options(${_ARG_NAME} ${type} "/arch:AVX2")
    elseif(APPLE)
      if(CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
        target_compile_options(${_ARG_NAME} ${type} -march=armv8-a)
      else()
        target_compile_options(${_ARG_NAME} ${type} -march=native)
      endif()
    else()
      target_compile_options(${_ARG_NAME} ${type} -march=native)
    endif()
  endif()
  target_compile_options(${_ARG_NAME}
    ${type} ${_ARG_PUBLIC_COMPILE_OPTIONS}
    PRIVATE ${_ARG_PRIVATE_COMPILE_OPTIONS}
  )

  if(NOT ${_ARG_EXCLUDE_FROM_INSTALL})
    set_property(GLOBAL APPEND PROPERTY MOMENTUM_TARGETS ${_ARG_NAME})
  endif()
endfunction()

function(mt_executable)
  set(prefix _ARG)
  set(options
  )
  set(oneValueArgs
    NAME
  )
  set(multiValueArgs
    HEADERS
    SOURCES
    SOURCES_VARS
    INCLUDE_DIRECTORIES
    LINK_LIBRARIES
  )
  cmake_parse_arguments(
    "${prefix}"
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
  )

  foreach(var ${_ARG_SOURCES_VARS})
    mt_append_momentum_filelist("${var}" _ARG_SOURCES)
  endforeach()

  add_executable(${_ARG_NAME}
    ${_ARG_HEADERS}
    ${_ARG_SOURCES}
  )
  target_include_directories(${_ARG_NAME}
    PRIVATE
      ${_ARG_INCLUDE_DIRECTORIES}
  )
  target_link_libraries(${_ARG_NAME}
    PRIVATE
      ${_ARG_LINK_LIBRARIES}
  )
endfunction()

function(mt_setup_gtest)
  set(prefix _ARG)
  set(options
  )
  set(oneValueArgs
    GIT_TAG
  )
  set(multiValueArgs
  )
  cmake_parse_arguments(
    "${prefix}"
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
  )

  if(NOT _ARG_GIT_TAG)
    set(_ARG_GIT_TAG v1.14.0)
  endif()

  include(FetchContent)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG ${_ARG_GIT_TAG}
  )

  # Set options
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(gtest_disable_pthreads ON CACHE BOOL "" FORCE)
  set(BUILD_GMOCK ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

  FetchContent_MakeAvailable(googletest)
endfunction()

function(mt_test)
  set(prefix _ARG)
  set(options
  )
  set(oneValueArgs
    NAME
  endif()
  )
  set(multiValueArgs
    HEADERS
    SOURCES
    SOURCES_VARS
    PYMOMENTUM_SOURCES_VARS
    INCLUDE_DIRECTORIES
    LINK_LIBRARIES
    ENV
  )
  cmake_parse_arguments(
    "${prefix}"
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
  )

  foreach(var ${_ARG_SOURCES_VARS})
    mt_append_momentum_filelist("${var}" _ARG_SOURCES)
  endforeach()

  foreach(var ${_ARG_PYMOMENTUM_SOURCES_VARS})
    mt_append_pymomentum_filelist("${var}" _ARG_SOURCES)
  endforeach()

  add_executable(${_ARG_NAME}
    ${_ARG_HEADERS}
    ${_ARG_SOURCES}
  )
  add_test(NAME ${_ARG_NAME} COMMAND $<TARGET_FILE:${_ARG_NAME}>)
  target_include_directories(${_ARG_NAME}
    PRIVATE
      ${_ARG_INCLUDE_DIRECTORIES}
  )
  target_link_libraries(${_ARG_NAME}
    PRIVATE
      ${_ARG_LINK_LIBRARIES}
      GTest::gmock
      GTest::gtest
      GTest::gtest_main
  )

  set(aggregated_env "")
  foreach(env ${_ARG_ENV})
    list(APPEND aggregated_env ${env})
  endforeach()
  set_tests_properties(${_ARG_NAME} PROPERTIES ENVIRONMENT "${aggregated_env}")

endfunction()

function(mt_python_binding)
  set(prefix _ARG)
  set(options
  )
  set(oneValueArgs
    NAME
  )
  set(multiValueArgs
    HEADERS
    HEADERS_VARS
    PYMOMENTUM_HEADERS_VARS
    PYMOMENTUM_SOURCES_VARS
    SOURCES
    SOURCES_VARS
    INCLUDE_DIRECTORIES
    LINK_LIBRARIES
    COMPILE_OPTIONS
  )
  cmake_parse_arguments(
    "${prefix}"
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
  )

  foreach(var ${_ARG_HEADERS_VARS})
    mt_append_momentum_filelist("${var}" _ARG_HEADERS)
  endforeach()

  foreach(var ${_ARG_SOURCES_VARS})
    mt_append_momentum_filelist("${var}" _ARG_SOURCES)
  endforeach()

  foreach(var ${_ARG_PYMOMENTUM_HEADERS_VARS})
    mt_append_pymomentum_filelist("${var}" _ARG_HEADERS)
  endforeach()

  foreach(var ${_ARG_PYMOMENTUM_SOURCES_VARS})
    mt_append_pymomentum_filelist("${var}" _ARG_SOURCES)
  endforeach()

  pybind11_add_module(${_ARG_NAME}
    MODULE
    ${_ARG_HEADERS}
    ${_ARG_SOURCES}
  )
  target_include_directories(${_ARG_NAME}
    PRIVATE
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
      $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> # TODO: Remove once momentum/pymomentum/pymomentum is moved to momentum/pymomentum
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
      ${_ARG_INCLUDE_DIRECTORIES}
  )
  target_compile_features(${_ARG_NAME} PRIVATE cxx_std_17)
  target_link_libraries(${_ARG_NAME} PRIVATE ${_ARG_LINK_LIBRARIES})
  target_compile_options(${_ARG_NAME} PRIVATE ${_ARG_COMPILE_OPTIONS})
endfunction()
