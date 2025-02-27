# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import math
import unittest

import numpy as np

from pymomentum.marker_tracking import (
    CalibrationConfig,
    ModelOptions,
    RefineConfig,
    TrackingConfig,
)


class TestMarkerTracker(unittest.TestCase):
    def test_configs(self) -> None:
        # test values different than default

        # CalibrationConfig
        calib_config = CalibrationConfig(
            min_vis_percent=0.1,
            loss_alpha=1.0,
            max_iter=10,
            debug=True,
            calib_frames=20,
            major_iter=2,
            global_scale_only=True,
            locators_only=True,
        )
        self.assertTrue(
            math.isclose(calib_config.min_vis_percent, 0.1, abs_tol=1e-6)
            and math.isclose(calib_config.loss_alpha, 1.0, abs_tol=1e-6)
            and calib_config.max_iter == 10
            and calib_config.debug
            and calib_config.calib_frames == 20
            and calib_config.major_iter == 2
            and calib_config.global_scale_only
            and calib_config.locators_only
        )

        # TrackingConfig
        tracking_config = TrackingConfig(
            min_vis_percent=0.2,
            loss_alpha=1.5,
            max_iter=20,
            debug=True,
            smoothing=0.1,
            collision_error_weight=0.2,
        )
        self.assertTrue(
            math.isclose(tracking_config.min_vis_percent, 0.2, abs_tol=1e-6)
            and math.isclose(tracking_config.loss_alpha, 1.5, abs_tol=1e-6)
            and tracking_config.max_iter == 20
            and tracking_config.debug
            and math.isclose(tracking_config.smoothing, 0.1, abs_tol=1e-6)
            and math.isclose(tracking_config.collision_error_weight, 0.2, abs_tol=1e-6)
        )

        # RefineConfig
        refine_config = RefineConfig(
            min_vis_percent=0.2,
            loss_alpha=1.5,
            max_iter=20,
            debug=True,
            smoothing=0.1,
            collision_error_weight=0.2,
            regularizer=0.1,
            calib_id=True,
            calib_locators=True,
        )
        self.assertTrue(
            math.isclose(refine_config.min_vis_percent, 0.2, abs_tol=1e-6)
            and math.isclose(refine_config.loss_alpha, 1.5, abs_tol=1e-6)
            and refine_config.max_iter == 20
            and refine_config.debug
            and math.isclose(refine_config.smoothing, 0.1, abs_tol=1e-6)
            and math.isclose(refine_config.collision_error_weight, 0.2, abs_tol=1e-6)
            and math.isclose(refine_config.regularizer, 0.1, abs_tol=1e-6)
            and refine_config.calib_id
            and refine_config.calib_locators
        )

        model_options = ModelOptions("blueman.glb", "blueman.model", "blueman.locators")

        self.assertTrue(
            model_options.model == "blueman.glb"
            and model_options.parameters == "blueman.model"
            and model_options.locators == "blueman.locators"
        )

    def test_refine_config(self) -> None:
        refine_config_default = RefineConfig()
        self.assertTrue(refine_config_default.smoothing_weights.size == 0)

        refine_config = RefineConfig(smoothing_weights=np.asarray([1.0, 0, 1.0]))
        self.assertEqual(refine_config.smoothing_weights.shape, (3,))
        self.assertTrue(
            np.allclose(refine_config.smoothing_weights, [1.0, 0, 1.0], atol=1e-6)
        )

    def test_tracking_config(self) -> None:
        tracking_config = TrackingConfig(smoothing_weights=np.asarray([1.0, 0, 1.0]))
        self.assertEqual(tracking_config.smoothing_weights.shape, (3,))
        self.assertTrue(
            np.allclose(tracking_config.smoothing_weights, [1.0, 0, 1.0], atol=1e-6)
        )


if __name__ == "__main__":
    unittest.main()
