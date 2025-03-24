# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#===============================================================================
# Find dependencies
#===============================================================================

set(ENV{NVTOOLSEXT_PATH} "$ENV{CONDA_PREFIX}/include")

find_package(ATen CONFIG REQUIRED
  HINTS
    $ENV{CONDA_PREFIX}/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages/torch/
)

find_package(Torch CONFIG REQUIRED
  HINTS
    $ENV{CONDA_PREFIX}/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages/torch/
)

find_library(torch_python
  NAMES torch_python
  HINTS
    $ENV{CONDA_PREFIX}/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages/torch/lib/
  REQUIRED
)

#===============================================================================
# Find build dependencies
#===============================================================================

# pybind11
if(MOMENTUM_USE_SYSTEM_PYBIND11)
  find_package(pybind11 CONFIG REQUIRED)
else()
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
  )
  set(PYBIND11_TEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(pybind11)
endif()

#===============================================================================
# Build PyMomentum
#===============================================================================

mt_library(
  NAME tensor_utility
  PYMOMENTUM_HEADERS_VARS tensor_utility_public_headers
  PYMOMENTUM_SOURCES_VARS tensor_utility_sources
  PUBLIC_INCLUDE_DIRECTORIES
    ${TORCH_INCLUDE_DIRS}
  PUBLIC_LINK_LIBRARIES
    Eigen3::Eigen
    ${TORCH_LIBRARIES}
  PRIVATE_LINK_LIBRARIES
    momentum
  PUBLIC_COMPILE_OPTIONS
    ${TORCH_CXX_FLAGS}
  NO_INSTALL
)

mt_library(
  NAME python_utility
  PYMOMENTUM_HEADERS_VARS python_utility_public_headers
  PYMOMENTUM_SOURCES_VARS python_utility_sources
  PUBLIC_LINK_LIBRARIES
    momentum
    nlohmann_json::nlohmann_json
    pybind11::pybind11
    Python3::Module
  NO_INSTALL
)

mt_library(
  NAME tensor_momentum
  PYMOMENTUM_HEADERS_VARS tensor_momentum_public_headers
  PYMOMENTUM_SOURCES_VARS tensor_momentum_sources
  PUBLIC_INCLUDE_DIRECTORIES
    ${ATEN_INCLUDE_DIR}
  PRIVATE_INCLUDE_DIRECTORIES
    ${TORCH_INCLUDE_DIRS}
  PUBLIC_LINK_LIBRARIES
    momentum
    diff_ik
    ${ATEN_LIBRARIES}
    pybind11::pybind11
  PRIVATE_LINK_LIBRARIES
    python_utility
    tensor_utility
    Ceres::ceres
    Dispenso::dispenso
    Eigen3::Eigen
    ${TORCH_LIBRARIES}
    ${torch_python}
  PUBLIC_COMPILE_OPTIONS
    ${TORCH_CXX_FLAGS}
  NO_INSTALL
)

mt_library(
  NAME tensor_ik
  PYMOMENTUM_HEADERS_VARS tensor_ik_public_headers
  PYMOMENTUM_SOURCES_VARS tensor_ik_sources
  PUBLIC_INCLUDE_DIRECTORIES
    ${ATEN_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
  PUBLIC_LINK_LIBRARIES
    momentum
    diff_ik
    ${ATEN_LIBRARIES}
  PRIVATE_LINK_LIBRARIES
    tensor_utility
    Dispenso::dispenso
    Eigen3::Eigen
    ${TORCH_LIBRARIES}
    ${torch_python}
  PUBLIC_COMPILE_OPTIONS
    ${TORCH_CXX_FLAGS}
  NO_INSTALL
)

mt_python_binding(
  NAME geometry
  PYMOMENTUM_HEADERS_VARS geometry_public_headers
  PYMOMENTUM_SOURCES_VARS geometry_sources
  INCLUDE_DIRECTORIES
    ${ATEN_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
  LINK_LIBRARIES
    character
    character_test_helpers
    io_fbx
    io_gltf
    io_marker
    io_openfbx
    io_shape
    io_skeleton
    io_urdf
    python_utility
    tensor_momentum
    tensor_utility
    ${ATEN_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${torch_python}
  COMPILE_OPTIONS
    ${TORCH_CXX_FLAGS}
)

mt_python_binding(
  NAME pymomentum_solver
  MODULE_NAME solver
  PYMOMENTUM_HEADERS_VARS solver_public_headers
  PYMOMENTUM_SOURCES_VARS solver_sources
  INCLUDE_DIRECTORIES
    ${ATEN_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
  LINK_LIBRARIES
    character_solver
    math
    skeleton
    python_utility
    tensor_ik
    tensor_momentum
    tensor_utility
    ${ATEN_LIBRARIES}
    ${TORCH_LIBRARIES}
    ${torch_python}
  COMPILE_OPTIONS
    ${TORCH_CXX_FLAGS}
)

mt_python_binding(
  NAME marker_tracking
  PYMOMENTUM_HEADERS_VARS marker_tracking_public_headers
  PYMOMENTUM_SOURCES_VARS marker_tracking_sources
  INCLUDE_DIRECTORIES
  LINK_LIBRARIES
    character
    marker_tracker
    math
    process_markers
  COMPILE_OPTIONS
)

mt_python_library(
  NAME quaternion
  PYMOMENTUM_SOURCES_VARS quaternion_sources
)

mt_python_library(
  NAME skel_state
  PYMOMENTUM_SOURCES_VARS skel_state_sources
)

#===============================================================================
# Tests
#===============================================================================

if(MOMENTUM_BUILD_TESTING)
  enable_testing()
  mt_setup_gtest()

  mt_test(
    NAME tensor_utility_test
    PYMOMENTUM_SOURCES_VARS tensor_utility_test_sources
    LINK_LIBRARIES tensor_utility
  )

  mt_test(
    NAME tensor_ik_test
    PYMOMENTUM_SOURCES_VARS tensor_ik_test_sources
    LINK_LIBRARIES
      character_test_helpers
      tensor_ik
      tensor_utility
  )
endif()

#===============================================================================
# Install
#===============================================================================

mt_install_pymomentum()
