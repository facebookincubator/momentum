# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import math
import pkgutil
import tempfile
import unittest

import numpy as np
import pymomentum.geometry as pym_geometry
import torch


class TestFBX(unittest.TestCase):
    def test_load_animation(self) -> None:
        """
        Loads a very simple 3-bone chain with animation from FBX and verifies that
        the animation is correct, including non-animated channels.
        """
        fbx_bytes = pkgutil.get_data(
            __package__,
            "resources/animation_test.fbx",
        )
        assert fbx_bytes is not None

        (
            character,
            animations,
            fps,
        ) = pym_geometry.Character.load_fbx_with_motion_from_bytes(fbx_bytes)
        self.assertEqual(fps, 24)
        self.assertEqual(len(animations), 1)
        joint_params = animations[0]

        joint1 = character.skeleton.joint_index("joint1")
        joint2 = character.skeleton.joint_index("joint2")
        joint3 = character.skeleton.joint_index("joint3")
        joint_params = joint_params.reshape(-1, len(character.skeleton.joint_names), 7)
        joint_params_first = joint_params[0]
        joint_params_last = joint_params[-1]

        self.assertTrue(
            np.allclose(
                joint_params_first[joint1][3:6],
                np.asarray([math.pi / 2, 0, 0], dtype=np.float32),
                atol=1e-5,
            )
        )
        self.assertTrue(
            np.allclose(
                joint_params_first[joint2][3:6],
                np.asarray([0, 0, math.pi / 2], dtype=np.float32),
                atol=1e-5,
            )
        )

        self.assertTrue(
            np.allclose(
                joint_params_last[joint1][3:6],
                np.asarray([math.pi / 2, math.pi / 2, 0], dtype=np.float32),
                atol=1e-5,
            )
        )
        self.assertTrue(
            np.allclose(
                joint_params_last[joint2][3:6],
                np.asarray([0, 0, math.pi / 2], dtype=np.float32),
                atol=1e-5,
            )
        )

        skel_state = pym_geometry.joint_parameters_to_skeleton_state(
            character, torch.from_numpy(joint_params)
        )

        skel_state_first = skel_state[0]
        skel_state_last = skel_state[-1]

        # Start and end values read out from Maya.
        # 3-joint chain basically just rotates by 90 degrees.
        self.assertTrue(
            torch.allclose(
                skel_state_first[joint1][0:3], torch.Tensor([0, 0, 4]), atol=1e-5
            )
        )
        self.assertTrue(
            torch.allclose(
                skel_state_first[joint2][0:3], torch.Tensor([4, 0, 4]), atol=1e-5
            )
        )
        self.assertTrue(
            torch.allclose(
                skel_state_first[joint3][0:3], torch.Tensor([4, 0, 8]), atol=1e-5
            )
        )

        self.assertTrue(
            torch.allclose(
                skel_state_last[joint1][0:3], torch.Tensor([0, 0, 4]), atol=1e-5
            )
        )
        self.assertTrue(
            torch.allclose(
                skel_state_last[joint2][0:3], torch.Tensor([0, 0, 0]), atol=1e-5
            )
        )
        self.assertTrue(
            torch.allclose(
                skel_state_last[joint3][0:3], torch.Tensor([4, 0, 0]), atol=1e-5
            )
        )

    def test_save_motions(self):
        character = pym_geometry.test_character()
        torch.manual_seed(0)  # ensure repeatability

        nBatch = 5
        nParams = character.parameter_transform.size
        model_params = pym_geometry.uniform_random_to_model_parameters(
            character, torch.rand(nBatch, nParams)
        ).double()
        joint_params = character.parameter_transform.apply(model_params)
        joint_params_shape_in = joint_params.shape

        def verify_fbx(file_name: str) -> None:
            # Load FBX file
            l_character, motion, fps = pym_geometry.Character.load_fbx_with_motion(
                file_name
            )
            self.assertEqual(1, len(motion))
            self.assertEqual(motion[0].shape, joint_params_shape_in)
            self.assertEqual(fps, 60)

        with tempfile.NamedTemporaryFile(suffix=".fbx") as temp_file:
            offsets = np.zeros(joint_params_shape_in[1])
            pym_geometry.Character.save_fbx(
                path=temp_file.name,
                character=character,
                motion=model_params.numpy(),
                offsets=offsets,
                fps=60,
            )
            verify_fbx(temp_file.name)

        with tempfile.NamedTemporaryFile(suffix=".fbx") as temp_file:
            pym_geometry.Character.save_fbx_with_joint_params(
                path=temp_file.name,
                character=character,
                joint_params=joint_params.numpy(),
                fps=60,
            )
            verify_fbx(temp_file.name)
