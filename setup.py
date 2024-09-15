#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import shutil
import subprocess
import sys

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake is not available.")
        super().run()

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        if "DEBUG" in os.environ:
            cfg = "Debug" if os.environ["DEBUG"] == "1" else "Release"
        else:
            cfg = "Debug" if self.debug else "Release"

        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DBUILD_SHARED_LIBS=OFF",
            f"-DMOMENTUM_BUILD_EXAMPLES={os.environ.get(
                'MOMENTUM_BUILD_EXAMPLES', 'OFF')}",
            f"-DMOMENTUM_BUILD_PYMOMENTUM=ON",
            # TODO: OFF when pym.geometry doesn't depend on testing
            f"-DMOMENTUM_BUILD_TESTING={os.environ.get(
                'MOMENTUM_BUILD_TESTING', 'ON')}",
            f"-DMOMENTUM_ENABLE_SIMD={os.environ.get(
                'MOMENTUM_ENABLE_SIMD', 'ON')}",
            f"-DMOMENTUM_USE_SYSTEM_GOOGLETEST={os.environ.get(
                'MOMENTUM_USE_SYSTEM_GOOGLETEST', 'ON')}",
            f"-DMOMENTUM_USE_SYSTEM_PYBIND11={os.environ.get(
                'MOMENTUM_USE_SYSTEM_PYBIND11', 'ON')}",
            f"-DMOMENTUM_USE_SYSTEM_RERUN_CPP_SDK={os.environ.get(
                'MOMENTUM_USE_SYSTEM_RERUN_CPP_SDK', 'ON')}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]
        build_args = ["--target", os.path.basename(ext.name)]

        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # Default to Ninja
        if self.compiler.compiler_type == "msvc":
            cmake_args += ["-G Visual Studio 17 2022"]
        else:
            cmake_args += ["-G Ninja"]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )

        # After building, copy the shared library to pymomentum/
        target_dir = os.path.join(ROOT_DIR, "pymomentum")
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        shutil.copy(self.get_ext_fullpath(ext.name), target_dir)


def main():
    with open(os.path.join(ROOT_DIR, "README.md"), encoding="utf-8") as f:
        long_description = f.read()

    setup(
        name="pymomentum",
        # TODO: get version from a single source (e.g., version.txt)
        version="0.1.0",
        description="A library providing foundational algorithms for human kinematic motion and numerical optimization solvers to apply human motion in various applications",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/facebookincubator/momentum",
        author="Meta Reality Labs Research",
        license="MIT",
        install_requires=["numpy", "typing", "dataclasses"],  # TODO:
        python_requires=">=3.7",
        packages=find_packages(),
        zip_safe=False,
        ext_modules=[
            CMakeExtension("geometry", sourcedir=ROOT_DIR),
            CMakeExtension("quaternion", sourcedir=ROOT_DIR),
            CMakeExtension("skel_state", sourcedir=ROOT_DIR),
        ],
        cmdclass={
            "build_ext": CMakeBuild,
        },
    )


if __name__ == "__main__":
    main()
