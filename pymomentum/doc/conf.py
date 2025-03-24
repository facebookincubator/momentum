# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

project = "PyMomentum"

extensions = [
    "sphinx.ext.autodoc",
]

exclude_patterns = [
    ".pixi",
    "build",
]

html_theme = "sphinx_rtd_theme"
