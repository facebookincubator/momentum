# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.16.3)

project(momentum)

find_package(momentum CONFIG REQUIRED)
add_executable(hello_world main.cpp)
target_link_libraries(hello_world PRIVATE momentum::math)
