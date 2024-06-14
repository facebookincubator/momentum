# Momentum

Momentum provides foundational algorithms for human kinematic motion and
numerical optimization solvers to apply human motion in various applications.

<p align="center">
  <img src="momentum/website/static/img/momentum_1.png" width="30%" alt="Forward and Inverse Kinematics with Interpretable Parameterization" />
  <img src="momentum/website/static/img/momentum_3.png" width="30%" alt="RGBD Body Tracking Solver" />
  <img src="momentum/website/static/img/momentum_4.png" width="30%" alt="Monocular RGB Body Tracking Solver" />
</p>

## Getting Started

### Supported Environments

* **OS**: Windows, Linux, macOS

### Prerequisite

Complete the following steps only once:

- Install Pixi by following the instructions on https://prefix.dev/

### Install Momentum

1. Clone the repository and navigate to the root directory:

  ```
  clone https://github.com/facebookincubator/momentum
  cd momentum
  ```

2. Build and install Momentum in a virtual environment using Pixi:

  ```
  pixi run cmake --build build --target install --parallel
  ```

### Creating Your First Application

#### Writing Source Code

Now, let's define a simple Momentum application. Create a new file named `main.cpp` with the following content:

```cpp
#include <momentum/math/mesh.h>

using namespace momentum;

int main() {
  auto mesh = Mesh();
  mesh.updateNormals();
  return EXIT_SUCCESS;
}
```

#### Writing CMake Script

Create a `CMakeLists.txt` file in the same directory as `main.cpp`.

To add momentum to your CMake project, first find the momentum package using the
`find_package` function and then add the appropriate `momentum::<target>` as a
dependency to your library or executable. For example, if you want to use the
character functionality from momentum, you would add `momentum::character` as a
dependency:

```cmake
cmake_minimum_required(VERSION 3.16.3)

project(momentum)

find_package(momentum CONFIG REQUIRED)
add_executable(hello_world main.cpp)
target_link_libraries(hello_world PRIVATE momentum::math)
```

Refer to the example project located at `momentum/examples/hello_world/` for the complete source code.

#### Building using CMake

To configure the application, run:

```
# Linux and macOS
pixi run cmake -S momentum/examples/hello_world -B momentum/examples/hello_world/build -DCMAKE_BUILD_TYPE=Release

# Windows
pixi run cmake -S momentum/examples/hello_world -B momentum/examples/hello_world/build
```

To build the application, run:

```
# Linux and macOS
pixi run cmake --build momentum/examples/hello_world/build

# Windows
pixi run cmake --build momentum/examples/hello_world/build --config Release
```

#### Run the Application

Execute the application with:

```
./momentum/examples/hello_world/build/hello_world
```

## 📖 Documentation

The full documentation for Momentum can be found on our [website][docs].

[docs]: https://facebookincubator.github.io/momentum

## Contributing

Check our [contributing guide](CONTRIBUTING.md) to learn about how to contribute
to the project.

## License

Momentum is licensed under the MIT License. A copy of the license
[can be found here.](LICENSE)
