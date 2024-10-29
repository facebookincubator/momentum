---
sidebar_position: 1
---

# Getting Started

This page guides you through the process of building Momentum and running the examples.

## Installing Momentum and PyMomentum

Momentum binary builds are available for Windows, macOS, and Linux via [Pixi](https://prefix.dev/) or the Conda package manager.

For Windows, please install [Visual Studio 2022](https://visualstudio.microsoft.com/vs/) or greater.

### Pixi

```
# Momentum (C++)
pixi add momentum-cpp

# PyMomentum (Python)
pixi add pymomentum

# Both
pixi add momentum
```

### Conda

```
conda install -c conda-forge momentum-cpp
conda install -c conda-forge pymomentum # Windows is not supported yet
conda install -c conda-forge momentum
```

## Building Momentum from Source

### Prerequisite

Complete the following steps only once:

1. Install Pixi by following the instructions on https://prefix.dev/

1. Clone the repository and navigate to the root directory:

   ```
   git clone https://github.com/facebookincubator/momentum
   cd momentum
   ```

   Ensure that all subsequent commands are executed in the project's root directory unless specified otherwise.

### Build and Test

- Build the project with the following command (note that the first run may take a few minutes as it installs all dependencies):

  ```
  pixi run build
  ```

- Run the tests with:

  ```
  pixi run test
  ```

To view all available command lines, run `pixi task list`.

### Hello World Example

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

### Running Other Examples

To run other examples:

```
pixi run glb_viewer --help
```

For more examples, please refer to the [Examples](https://facebookincubator.github.io/momentum/docs/examples/viewers) page.

### Clean Up

If you need to start over for any reason:

```
pixi run clean
```

Momentum uses the `build/` directory for CMake builds, and `.pixi/` for the Pixi virtual environment. You can clean up everything by either manually removing these directories or by running the command above.

### FBX support (Windows only)

Momentum uses OpenFBX to load Autodesk's FBX file format, which is built by default. To save files in FBX format, you need to install the FBX SDK 2020.3. You can download it from Autodesk's [website](https://aps.autodesk.com/developer/overview/fbx-sdk) or use [this direct link](https://damassets.autodesk.net/content/dam/autodesk/www/files/fbx202037_fbxsdk_vs2019_win.exe). After installing the SDK, build Momentum from source with `MOMENTUM_BUILD_IO_FBX=ON` option as:

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
