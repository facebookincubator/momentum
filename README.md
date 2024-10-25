# Momentum

[![CI Windows][ci-windows-badge]][ci-windows]
[![CI macOS][ci-macos-badge]][ci-macos]
[![CI Ubuntu][ci-ubuntu-badge]][ci-ubuntu]
[![Publish Website][website-badge]][website]

[ci-windows-badge]: https://github.com/facebookincubator/momentum/actions/workflows/ci_windows.yml/badge.svg
[ci-windows]: https://github.com/facebookincubator/momentum/actions/workflows/ci_windows.yml
[ci-macos-badge]: https://github.com/facebookincubator/momentum/actions/workflows/ci_macos.yml/badge.svg
[ci-macos]: https://github.com/facebookincubator/momentum/actions/workflows/ci_macos.yml
[ci-ubuntu-badge]: https://github.com/facebookincubator/momentum/actions/workflows/ci_ubuntu.yml/badge.svg
[ci-ubuntu]: https://github.com/facebookincubator/momentum/actions/workflows/ci_ubuntu.yml
[website-badge]: https://github.com/facebookincubator/momentum/actions/workflows/publish_website.yml/badge.svg
[website]: https://github.com/facebookincubator/momentum/actions/workflows/publish_website.yml

Momentum provides foundational algorithms for human kinematic motion and
numerical optimization solvers to apply human motion in various applications.

<p align="center">
  <img src="momentum/website/static/img/momentum_1.png" width="30%" alt="Forward and Inverse Kinematics with Interpretable Parameterization" />
  <img src="momentum/website/static/img/momentum_3.png" width="30%" alt="RGBD Body Tracking Solver" />
  <img src="momentum/website/static/img/momentum_4.png" width="30%" alt="Monocular RGB Body Tracking Solver" />
</p>

## Getting Started

This page guides you through the process of building Momentum and running the examples.

### Installing Momentum and PyMomentum

Momentum binary builds are available for Windows, macOS, and Linux via [Pixi](https://prefix.dev/) or the Conda package manager.

#### Pixi

```
# Momentum (C++)
pixi add momentum-cpp

# PyMomentum (Python)
pixi add pymomentum

# Both
pixi add momentum
```

#### Conda

```
conda install -c conda-forge momentum-cpp
conda install -c conda-forge pymomentum # Linux only
conda install -c conda-forge momentum
```

#### Micromamba

```
micromamba install -c conda-forge momentum-cpp
micromamba install -c conda-forge pymomentum # Linux only
micromamba install -c conda-forge momentum
```

### Building Momentum from Source

#### Prerequisite

Complete the following steps only once:

1. Install Pixi by following the instructions on https://prefix.dev/

1. Clone the repository and navigate to the root directory:

   ```
   git clone https://github.com/facebookincubator/momentum
   cd momentum
   ```

   Ensure that all subsequent commands are executed in the project's root directory unless specified otherwise.

#### Build and Test

- Build the project with the following command (note that the first run may take a few minutes as it installs all dependencies):

  ```
  pixi run build
  ```

- Run the tests with:

  ```
  pixi run test
  ```

To view all available command lines, run `pixi task list`.

#### Hello World Example

To run the `hello_world` example:

```
pixi run hello_world
```

Alternatively, you can directly run the executable:

```
# Linux and macOS
./build/hello_world

# Windows
./build/Release/hello_world.exe
```

#### Running Other Examples

To run other examples:

```
pixi run glb_viewer --help
```

For more examples, please refer to the [Examples](https://facebookincubator.github.io/momentum/docs/examples/viewers) page.

#### Clean Up

If you need to start over for any reason:

```
pixi run clean
```

Momentum uses the `build/` directory for CMake builds, and `.pixi/` for the Pixi virtual environment. You can clean up everything by either manually removing these directories or by running the command above.

#### FBX support (Windows only)

To load and save Autodesk's FBX file format, you need to install the FBX SDK 2019.2 from Autodesk's [website](https://aps.autodesk.com/developer/overview/fbx-sdk) or [this direct link](https://www.autodesk.com/content/dam/autodesk/www/adn/fbx/20192/fbx20192_fbxsdk_vs2017_win.exe) first. After installing the SDK, you can build with `MOMENTUM_BUILD_IO_FBX=ON`:

```
# Powershell
$env:MOMENTUM_BUILD_IO_FBX = "ON"; pixi run <target>

# cmd
set MOMENTUM_BUILD_IO_FBX=ON && pixi run <target>
```

For example, file conversion can be run as follows:

```
# Powershell
$env:MOMENTUM_BUILD_IO_FBX = "ON"; pixi run convert_model -d <input.glb> -o <out.fbx>

# cmd
set MOMENTUM_BUILD_IO_FBX=ON && pixi run convert_model -d <input.glb> -o <out.fbx>
```

## 📖 Documentation

The full documentation for Momentum can be found on our [website](https://facebookincubator.github.io/momentum) and the C++ API documentation is available [here](https://facebookincubator.github.io/momentum/doxygen/index.html).

## Contributing

Check our [contributing guide](CONTRIBUTING.md) to learn about how to contribute
to the project.

## License

Momentum is licensed under the MIT License. A copy of the license
[can be found here.](LICENSE)
