---
sidebar_position: 1
---

# Viewers

Momentum supports various file formats for storing characters and motions. In this section, we demonstrate how to visualize these file formats using Momentum powered by [Rerun](https://rerun.io/).

For simplicity, we have built individual viewers for each file format. Each viewer example may require different arguments. Please use the `--help` option to see the correct usage of each example.

## GLB Viewer

To run the GLB viewer, use the following command:

```
pixi run glb_viewer --input <my_file.glb>
```

![glb_viewer](/img/glb_viewer.png)

* [Source Code](https://github.com/facebookresearch/momentum/tree/main/momentum/examples/glb_viewer)

## FBX Viewer

To run the FBX viewer, use the following command:

```
pixi run fbx_viewer --input <my_file.fbx>
```

* [Source Code](https://github.com/facebookresearch/momentum/tree/main/momentum/examples/fbx_viewer)

## C3D Viewer

To run the C3D viewer, use the following command:

```
pixi run c3d_viewer --input <my_file.c3d>
```

* [Source Code](https://github.com/facebookresearch/momentum/tree/main/momentum/examples/c3d_viewer)

## URDF Viewer

To run the URDF viewer, use the following command:

```
pixi run urdf_viewer --input <my_file.urdf>
```

* [Source Code](https://github.com/facebookresearch/momentum/tree/main/momentum/examples/urdf_viewer)

For example, you can download an Atlas robot from this [link](https://github.com/Daniella1/urdf_files_dataset/blob/main/urdf_files/matlab/Atlas/urdf/atlas.urdf), which may look like:

![urdf_viewer](/img/urdf_viewer.png)
