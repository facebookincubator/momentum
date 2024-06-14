# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

tensor_utility_public_headers = [
    "tensor_utility/autograd_utility.h",
    "tensor_utility/tensor_utility.h",
]

tensor_utility_sources = [
    "tensor_utility/tensor_utility.cpp",
]

tensor_utility_test_sources = [
    "cpp_test/tensor_utility_test.cpp",
]

python_utility_public_headers = [
    "python_utility/python_utility.h",
]

python_utility_sources = [
    "python_utility/python_utility.cpp",
]

tensor_momentum_public_headers = [
    "tensor_momentum/tensor_blend_shape.h",
    "tensor_momentum/tensor_joint_parameters_to_positions.h",
    "tensor_momentum/tensor_kd_tree.h",
    "tensor_momentum/tensor_momentum_utility.h",
    "tensor_momentum/tensor_mppca.h",
    "tensor_momentum/tensor_parameter_transform.h",
    "tensor_momentum/tensor_quaternion.h",
    "tensor_momentum/tensor_skeleton_state.h",
    "tensor_momentum/tensor_skinning.h",
    "tensor_momentum/tensor_transforms.h",
]

tensor_momentum_sources = [
    "tensor_momentum/tensor_blend_shape.cpp",
    "tensor_momentum/tensor_joint_parameters_to_positions.cpp",
    "tensor_momentum/tensor_kd_tree.cpp",
    "tensor_momentum/tensor_momentum_utility.cpp",
    "tensor_momentum/tensor_mppca.cpp",
    "tensor_momentum/tensor_parameter_transform.cpp",
    "tensor_momentum/tensor_quaternion.cpp",
    "tensor_momentum/tensor_skeleton_state.cpp",
    "tensor_momentum/tensor_skinning.cpp",
    "tensor_momentum/tensor_transforms.cpp",
]

geometry_public_headers = [
    "geometry/momentum_geometry.h",
    "geometry/momentum_io.h",
]

geometry_sources = [
    "geometry/geometry_pybind.cpp",
    "geometry/momentum_geometry.cpp",
    "geometry/momentum_io.cpp",
]

quaternion_public_headers = [
]

quaternion_sources = [
    "quaternion/quaternion_pybind.cpp",
]

skel_state_public_headers = [
]

skel_state_sources = [
    "skel_state/skel_state_pybind.cpp",
]
