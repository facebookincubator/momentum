# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Tuple

import pymomentum.geometry as pym_geometry
import pymomentum.quaternion as pym_quaternion
import pymomentum.skel_state as pym_skel_state
import torch


def generate_skel_state_components(
    sz: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # [sz, 3
    trans: torch.Tensor = torch.normal(
        mean=0,
        std=4,
        size=(sz, 3),
        dtype=torch.float64,
        requires_grad=True,
    )

    rot: torch.Tensor = pym_quaternion.normalize(
        torch.normal(
            mean=0,
            std=4,
            size=(sz, 4),
            dtype=torch.float64,
            requires_grad=True,
        )
    )

    scale: torch.Tensor = torch.rand(
        size=(sz, 1), dtype=torch.float64, requires_grad=True
    )
    return (trans, rot, scale)


def generate_random_skel_state(sz: int) -> torch.Tensor:
    (trans, rot, scale) = generate_skel_state_components(sz)
    return torch.cat([trans, rot, scale], -1)


class TestSkelState(unittest.TestCase):
    def test_skel_state_to_transforms(self) -> None:
        character = pym_geometry.test_character()
        nBatch = 2
        modelParams = 0.2 * torch.ones(
            nBatch,
            character.parameter_transform.size,
            requires_grad=True,
            dtype=torch.float64,
        )
        joint_params = character.parameter_transform.apply(modelParams)
        skel_state = pym_geometry.joint_parameters_to_skeleton_state(
            character, joint_params
        )
        skel_state_d = pym_geometry.joint_parameters_to_skeleton_state(
            character, joint_params.to(torch.double)
        )
        inputs = [skel_state_d]
        torch.autograd.gradcheck(
            pym_skel_state.to_matrix,
            inputs,
            eps=1e-3,
            atol=1e-4,
            raise_exception=True,
        )

        self.assertTrue(torch.allclose(skel_state, skel_state_d, atol=1e-3))

    def test_multiply(self) -> None:
        nMats = 6
        s1 = generate_random_skel_state(nMats)
        s2 = generate_random_skel_state(nMats)

        s12 = pym_skel_state.multiply(s1, s2)
        m12 = pym_skel_state.to_matrix(s12)

        m1 = pym_skel_state.to_matrix(s1)
        m2 = pym_skel_state.to_matrix(s2)

        self.assertLess(
            torch.norm(torch.bmm(m1, m2) - m12),
            1e-4,
            "Multiplication is correct",
        )

    def test_inverse(self) -> None:
        nMats = 6
        s = generate_random_skel_state(nMats)
        s_inv = pym_skel_state.inverse(s)

        m = pym_skel_state.to_matrix(s)
        m_inv = pym_skel_state.to_matrix(s_inv)
        m_inv2 = torch.linalg.inv(m)

        self.assertLess(
            torch.norm(m_inv - m_inv2),
            1e-4,
            "Inverse is correct",
        )

    def test_transform_points(self) -> None:
        nMats = 6
        s = generate_random_skel_state(nMats)
        m = pym_skel_state.to_matrix(s)
        pts = torch.rand(size=(nMats, 3), dtype=torch.float64, requires_grad=False)
        hpts = torch.cat([pts, torch.ones(nMats, 1)], -1)
        transformed1 = torch.bmm(m, hpts.unsqueeze(-1))[:, 0:3].squeeze(-1)
        transformed2 = pym_skel_state.transform_points(s, pts)

        self.assertLess(
            torch.norm(transformed1 - transformed2),
            1e-4,
            "Inverse is correct",
        )

    def test_construct(self) -> None:
        nMats = 4
        (trans, rot, scale) = generate_skel_state_components(nMats)

        s1 = torch.cat([trans, rot, scale], -1)
        s2 = pym_skel_state.multiply(
            pym_skel_state.from_translation(trans),
            pym_skel_state.multiply(
                pym_skel_state.from_quaternion(rot), pym_skel_state.from_scale(scale)
            ),
        )

        self.assertLess(
            torch.norm(s1 - s2),
            1e-4,
            "constructing from components",
        )

    def test_matrix_to_skel_state(self) -> None:
        nMats = 6
        s1 = generate_random_skel_state(nMats)
        m1 = pym_skel_state.to_matrix(s1)
        s2 = pym_skel_state.from_matrix(m1)
        m2 = pym_skel_state.to_matrix(s2)
        self.assertLess(
            torch.norm(m1 - m2),
            1e-4,
            "matrix-to-skel-state conversion:L matrices should always match",
        )

        # Translation and scale should match.  Rotation might have its sign flipped:
        self.assertLess(
            torch.norm(s1[..., 0:3] - s2[..., 0:3]),
            1e-4,
            "matrix-to-skel-state conversion: translations should match",
        )
        self.assertLess(
            torch.norm(s1[..., 7] - s2[..., 7]),
            1e-4,
            "matrix-to-skel-state conversion: scales should match",
        )

    def test_skel_state_blending(self) -> None:
        t1 = torch.Tensor([1, 0, 0])
        t2 = torch.Tensor([0, 1, 0])

        s1 = pym_skel_state.from_translation(t1)
        s2 = pym_skel_state.from_translation(t2)
        s_test = pym_skel_state.from_translation(torch.Tensor([0.25, 0.75, 0]))

        weights = torch.Tensor([0.25, 0.75])
        s_blended = pym_skel_state.blend(torch.stack([s1, s2], 0), weights)

        self.assertLess(
            torch.norm(s_blended - s_test),
            1e-4,
            "matrix-to-skel-state blending",
        )

    def test_skel_state_splitting(self) -> None:
        t1 = torch.Tensor([1, 0, 0])
        skel_state1 = pym_skel_state.from_translation(t1)
        t2, r2, s2 = pym_skel_state.split(skel_state1)
        self.assertTrue(torch.allclose(t2, t2))

        r3 = torch.Tensor([0, 1, 0, 0])
        skel_state3 = pym_skel_state.from_quaternion(r3)
        t4, r4, s4 = pym_skel_state.split(skel_state3)
        self.assertTrue(torch.allclose(r3, r4))

        s5 = torch.Tensor([2])
        skel_state5 = pym_skel_state.from_scale(s5)
        t6, r6, s6 = pym_skel_state.split(skel_state5)
        self.assertTrue(torch.allclose(s5, s6))


if __name__ == "__main__":
    unittest.main()
