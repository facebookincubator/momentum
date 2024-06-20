/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/marker.h>
#include <momentum/character/skeleton.h>
#include <momentum/common/filesystem.h>
#include <momentum/io/gltf/gltf_file_format.hpp>
#include <momentum/io/gltf/gltf_io.h>
#include <momentum/math/types.h>

#include <fx/gltf.h>

namespace fx {
namespace gltf {
struct Document;
} // namespace gltf
} // namespace fx

namespace momentum {

using MotionParameters = std::tuple<std::vector<std::string>, MatrixXf>;
using IdentityParameters = std::tuple<std::vector<std::string>, VectorXf>;

// Helper class to build a glb scene. It supports adding multiple characters
// and motions for each character.
// By default, momentum extension is added to the glb nodes. This is required if you
// want to correctly load the exported character back.
class GltfBuilder final {
 public:
  GltfBuilder();

  ~GltfBuilder();

  GltfBuilder(GltfBuilder const&) = delete;

  void operator=(GltfBuilder const&) = delete;

  // Specify how marker mesh is represented in the glb file
  enum class MarkerMesh : uint8_t { None, UnitCube };

  // Add a character to the scene. Each character will have a root node with the character's
  // name as the parent of the skeleton root and the character mesh.
  // positionOffset and rotationOffset can be provided as an initial offset to the character.
  void addCharacter(
      const Character& character,
      const Vector3f& positionOffset = Vector3f::Zero(),
      const Quaternionf& rotationOffset = Quaternionf::Identity(),
      bool addExtensions = true,
      bool addCollisions = true,
      bool addLocators = true,
      bool addMesh = true);

  void setFps(float fps);

  // Add a motion to the provided character. If addCharacter is not called before adding
  // the motion, the character will be automatically added with the default settings.
  void addMotion(
      const Character& character,
      float fps = 120.0f,
      const MotionParameters& motion = {},
      const IdentityParameters& offsets = {},
      bool addExtensions = true,
      const std::string& customName = "default");

  // Add a skeleton states to the provided character. If addCharacter is not called before adding
  // the skeleton states, the character will be automatically added with the default settings.
  void addSkeletonStates(
      const Character& character,
      float fps,
      gsl::span<const SkeletonState> skeletonStates,
      const std::string& customName = "default");

  /// Add marker data to the file
  ///
  /// @param[in] fps The frame rate of the motion capture data in frames per second.
  /// @param[in] markerSequence A 2D vector specifying the Marker data (name/position/occlusion) for
  /// all markers across all captured frames. Size: [numFrames][numMarkers]
  /// @param[in] markerMesh Optional parameter specifying the MarkerMesh type (default is
  /// MarkerMesh::None).
  /// @param[in] animName Optional parameter specifying the animation name (default is "default").
  void addMarkerSequence(
      float fps,
      gsl::span<const std::vector<momentum::Marker>> markerSequence,
      MarkerMesh markerMesh = MarkerMesh::None,
      const std::string& animName = "default");

  // Save the file with the provided filename. If the fileFormat is 'GltfFileFormat::EXTENSION',
  // will deduct the file format by filename.
  // When embedResources is true, it will set all the existing buffers to embed the data.
  void save(
      const filesystem::path& filename,
      GltfFileFormat fileFormat = GltfFileFormat::EXTENSION,
      bool embedResources = false);

  static void save(
      fx::gltf::Document& document,
      const filesystem::path& filename,
      GltfFileFormat fileFormat = GltfFileFormat::EXTENSION,
      bool embedResources = false);

  // Set all existing buffers to embed resources.
  void forceEmbedResources();

  static void forceEmbedResources(fx::gltf::Document& document);

  // Allow copy the document, but do not allow modify the file
  // outside of the builder to keep metadata consistent
  const fx::gltf::Document& getDocument();

  size_t getNumCharacters();

  size_t getCharacterRootIndex(const std::string& name);

  size_t getNumJoints(const std::string& name);

  size_t getNumMotions();

  float getFps();

  std::vector<size_t> getCharacterMotions(const std::string& characterName);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace momentum
