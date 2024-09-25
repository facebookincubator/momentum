# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest

import pymomentum.geometry as pym_geometry
import pymomentum.geometry_test_helper as pym_geometry_test_helper
import pymomentum.quaternion as quaternion
import pymomentum.skel_state as skel_state
import torch


def generateSkelStateComponents(sz):
    trans = torch.normal(
        mean=0,
        std=4,
        size=(sz, 3),
        dtype=torch.float64,
        requires_grad=True,
    )

    rot = quaternion.normalize(
        torch.normal(
            mean=0,
            std=4,
            size=(sz, 4),
            dtype=torch.float64,
            requires_grad=True,
        )
    )

    scale = torch.rand(size=(sz, 1), dtype=torch.float64, requires_grad=True)
    return (trans, rot, scale)


def generateRandomSkelState(sz):
    (trans, rot, scale) = generateSkelStateComponents(sz)
    return torch.cat([trans, rot, scale], -1)


class TestSkelState(unittest.TestCase):
    def test_skelStateToTransforms(self) -> None:
        character = pym_geometry_test_helper.test_character()
        nBatch = 2
        modelParams = 0.2 * torch.ones(
            nBatch,
            character.parameter_transform.size,
            requires_grad=True,
            dtype=torch.float64,
        )
        jointParams = character.parameter_transform.apply(modelParams)
        skelState = pym_geometry.joint_parameters_to_skeleton_state(
            character, jointParams
        )
        inputs = [skelState]
        torch.autograd.gradcheck(
            skel_state.to_matrix,
            inputs,
            eps=1e-2,
            atol=1e-3,
            raise_exception=True,
        )

    def test_multiply(self) -> None:
        nMats = 6
        s1 = generateRandomSkelState(nMats)
        s2 = generateRandomSkelState(nMats)

        s12 = skel_state.multiply(s1, s2)
        m12 = skel_state.to_matrix(s12)

        m1 = skel_state.to_matrix(s1)
        m2 = skel_state.to_matrix(s2)

        self.assertLess(
            torch.norm(torch.bmm(m1, m2) - m12),
            1e-4,
            "Multiplication is correct",
        )

    def test_inverse(self) -> None:
        nMats = 6
        s = generateRandomSkelState(nMats)
        s_inv = skel_state.inverse(s)

        m = skel_state.to_matrix(s)
        m_inv = skel_state.to_matrix(s_inv)
        m_inv2 = torch.linalg.inv(m)

        self.assertLess(
            torch.norm(m_inv - m_inv2),
            1e-4,
            "Inverse is correct",
        )

    def test_transform_points(self) -> None:
        nMats = 6
        s = generateRandomSkelState(nMats)
        m = skel_state.to_matrix(s)
        pts = torch.rand(size=(nMats, 3), dtype=torch.float64, requires_grad=False)
        hpts = torch.cat([pts, torch.ones(nMats, 1)], -1)
        transformed1 = torch.bmm(m, hpts.unsqueeze(-1))[:, 0:3].squeeze(-1)
        transformed2 = skel_state.transform_points(s, pts)

        self.assertLess(
            torch.norm(transformed1 - transformed2),
            1e-4,
            "Inverse is correct",
        )

    def test_construct(self) -> None:
        nMats = 4
        (trans, rot, scale) = generateSkelStateComponents(nMats)

        s1 = torch.cat([trans, rot, scale], -1)
        s2 = skel_state.multiply(
            skel_state.from_translation(trans),
            skel_state.multiply(
                skel_state.from_quaternion(rot), skel_state.from_scale(scale)
            ),
        )

        self.assertLess(
            torch.norm(s1 - s2),
            1e-4,
            "constructing from components",
        )

    def test_matrix_to_skel_state(self) -> None:
        nMats = 6
        s1 = generateRandomSkelState(nMats)
        m1 = skel_state.to_matrix(s1)
        s2 = skel_state.from_matrix(m1)
        m2 = skel_state.to_matrix(s2)
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

        s1 = skel_state.from_translation(t1)
        s2 = skel_state.from_translation(t2)
        s_test = skel_state.from_translation(torch.Tensor([0.25, 0.75, 0]))

        weights = torch.Tensor([0.25, 0.75])
        s_blended = skel_state.blend(torch.stack([s1, s2], 0), weights)

        self.assertLess(
            torch.norm(s_blended - s_test),
            1e-4,
            "matrix-to-skel-state blending",
        )

    def test_skel_state_splitting(self) -> None:
        t1 = torch.Tensor([1, 0, 0])
        skel_state1 = skel_state.from_translation(t1)
        t2, r2, s2 = skel_state.split(skel_state1)
        self.assertTrue(torch.allclose(t2, t2))

        r3 = torch.Tensor([0, 1, 0, 0])
        skel_state3 = skel_state.from_quaternion(r3)
        t4, r4, s4 = skel_state.split(skel_state3)
        self.assertTrue(torch.allclose(r3, r4))

        s5 = torch.Tensor([2])
        skel_state5 = skel_state.from_scale(s5)
        t6, r6, s6 = skel_state.split(skel_state5)
        self.assertTrue(torch.allclose(s5, s6))


if __name__ == "__main__":
    unittest.main()
