/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/geometry/momentum_geometry.h"
#include "pymomentum/geometry/momentum_io.h"
#include "pymomentum/python_utility/python_utility.h"
#include "pymomentum/tensor_momentum/tensor_blend_shape.h"
#include "pymomentum/tensor_momentum/tensor_joint_parameters_to_positions.h"
#include "pymomentum/tensor_momentum/tensor_kd_tree.h"
#include "pymomentum/tensor_momentum/tensor_mppca.h"
#include "pymomentum/tensor_momentum/tensor_parameter_transform.h"
#include "pymomentum/tensor_momentum/tensor_skeleton_state.h"
#include "pymomentum/tensor_momentum/tensor_skinning.h"

#include <momentum/character/blend_shape.h>
#include <momentum/character/character.h>
#include <momentum/character/character_utility.h>
#include <momentum/character/fwd.h>
#include <momentum/character/inverse_parameter_transform.h>
#include <momentum/character/joint.h>
#include <momentum/character/locator.h>
#include <momentum/character/skeleton.h>
#include <momentum/character/skeleton_state.h>
#include <momentum/character/skin_weights.h>
#include <momentum/io/shape/blend_shape_io.h>
#include <momentum/math/mesh.h>
#include <momentum/math/mppca.h>
#include <momentum/test/character/character_helpers.h>

#include <fmt/format.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/python.h>
#include <Eigen/Core>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace py = pybind11;
namespace mm = momentum;

using namespace pymomentum;

PYBIND11_MODULE(geometry, m) {
  // TODO more explanation
  m.doc() = "Geometry and forward kinematics for momentum models.  ";
  m.attr("__name__") = "pymomentum.geometry";

  pybind11::module_::import("torch"); // @dep=//caffe2:torch

  m.attr("PARAMETERS_PER_JOINT") = mm::kParametersPerJoint;

  // We need to forward-declare classes so that if we refer to them they get
  // typed correctly; otherwise we end up with "momentum::Locator" in the
  // docstrings/type descriptors.
  auto characterClass = py::class_<mm::Character>(
      m,
      "Character",
      "A complete momentum character including its skeleton and mesh.");
  auto parameterTransformClass =
      py::class_<mm::ParameterTransform>(m, "ParameterTransform");
  auto inverseParameterTransformClass =
      py::class_<mm::InverseParameterTransform>(m, "InverseParameterTransform");
  auto meshClass = py::class_<mm::Mesh>(m, "Mesh");
  auto jointClass = py::class_<mm::Joint>(m, "Joint");
  auto skeletonClass = py::class_<mm::Skeleton>(m, "Skeleton");
  auto skinWeightsClass = py::class_<mm::SkinWeights>(m, "SkinWeights");
  auto locatorClass = py::class_<mm::Locator>(m, "Locator");
  auto blendShapeClass =
      py::class_<mm::BlendShape, std::shared_ptr<mm::BlendShape>>(
          m, "BlendShape");
  auto capsuleClass = py::class_<mm::TaperedCapsule>(m, "TaperedCapsule");
  auto markerClass = py::class_<mm::Marker>(m, "Marker");
  auto markerSequenceClass =
      py::class_<mm::MarkerSequence>(m, "MarkerSequence");

  // =====================================================
  // momentum::Character
  // - name
  // - skeleton
  // - parameter_transform
  // - locators
  // - mesh
  // - skin_weights
  // - blend_shape
  // - collision_geometry
  // - model_parameter_limits
  // - joint_parameter_limits
  // - [constructor](name, skeleton, parameter_transform, locators)
  // - with_mesh_and_skin_weights(mesh, skin_weights)
  // - with_blend_shape(blend_shape, n_shapes)
  //
  // [memeber methods]
  // - pose_mesh(jointParams)
  // - skin_points(skel_state, rest_vertices)
  // - scaled(scale)
  // - transformed(xform)
  // - rebind_skin()
  // - find_locators(names)
  // - apply_model_param_limits(model_params)
  // - simplify(enabled_parameters)
  // - load_locators(filename)
  // - load_locators_from_bytes(locator_bytes)
  // - load_model_definition(filename)
  // - load_model_definition_from_bytes(model_bytes)
  //
  // [static methods for io]
  // - from_gltf_bytes(gltf_btyes)
  // - to_gltf(character, fps, motion, offsets)
  // - load_fbx(fbxFilename, modelFilename, locatorsFilename)
  // - load_fbx_from_bytes(fbx_bytes, permissive)
  // - load_fbx_with_motion(fbxFilename, permissive)
  // - load_fbx_with_motion_from_bytes(fbx_bytes, permissive)
  // - load_gltf(path)
  // - load_gltf_with_motion(gltfFilename)
  // - save_gltf(path, character, fps, motion, offsets, markers)
  // - save_gltf_from_skel_states(path, character, fps, skel_states,
  // joint_params, markers)
  // - save_fbx(path, character, fps, motion, offsets)
  // =====================================================
  characterClass
      .def(
          py::init([](const std::string& name,
                      const mm::Skeleton& skeleton,
                      const mm::ParameterTransform& parameterTransform,
                      const mm::LocatorList& locators = mm::LocatorList()) {
            auto character = mm::Character(skeleton, parameterTransform);
            character.name = name;
            character.locators = locators;
            return character;
          }),
          py::arg("name"),
          py::arg("skeleton"),
          py::arg("parameter_transform"),
          py::kw_only(),
          py::arg("locators") = mm::LocatorList())
      .def(
          "with_mesh_and_skin_weights",
          [](const mm::Character& character,
             const mm::Mesh* mesh,
             const mm::SkinWeights* skinWeights) {
            mm::Character characterWithMesh = character;
            characterWithMesh.mesh = std::make_unique<mm::Mesh>(*mesh);
            const auto numMeshVertices = mesh->vertices.size();
            if (skinWeights && mesh &&
                numMeshVertices != skinWeights->index.rows() &&
                numMeshVertices != skinWeights->weight.rows()) {
              throw std::runtime_error(fmt::format(
                  "The number of mesh vertices and skin weight index/weight matrix rows should be the same {} vs {} vs {}",
                  numMeshVertices,
                  skinWeights->index.rows(),
                  skinWeights->weight.rows()));
            }
            characterWithMesh.skinWeights =
                std::make_unique<mm::SkinWeights>(*skinWeights);
            return characterWithMesh;
          },
          "Adds mesh and skin weight to the character and return a new character instance",
          py::arg("mesh"),
          py::arg("skin_weights"))
      .def_readonly("name", &mm::Character::name, "The character's name.")
      .def_readonly(
          "skeleton", &mm::Character::skeleton, "The character's skeleton.")
      .def_readonly(
          "parameter_transform",
          &mm::Character::parameterTransform,
          "Maps the reduced k-dimensional modelParameters that are used in the IK solve "
          "to the full 7*n-dimensional parameters used in the skeleton.")
      .def_readonly(
          "locators", &mm::Character::locators, "List of locators on the mesh.")
      .def_property_readonly(
          "mesh",
          [](const mm::Character& c) -> std::unique_ptr<mm::Mesh> {
            return (c.mesh) ? std::make_unique<mm::Mesh>(*c.mesh)
                            : mm::Mesh_u();
          },
          ":return: The character's mesh, or None if not present.")
      .def_property_readonly(
          "has_mesh",
          [](const mm::Character& c) -> bool {
            return static_cast<bool>(c.mesh) &&
                static_cast<bool>(c.skinWeights);
          })
      .def_property_readonly(
          "skin_weights",
          [](const mm::Character& c) -> std::unique_ptr<mm::SkinWeights> {
            return (c.skinWeights)
                ? std::make_unique<mm::SkinWeights>(*c.skinWeights)
                : mm::SkinWeights_u();
          },
          "The character's skinning weights.")
      .def_property_readonly(
          "blend_shape",
          [](const mm::Character& c)
              -> std::optional<std::shared_ptr<const mm::BlendShape>> {
            if (c.blendShape) {
              return c.blendShape;
            } else {
              return {};
            }
          },
          ":return: The character's :class:`BlendShape` basis, if present, or None.")
      .def_property_readonly(
          "collision_geometry",
          [](const mm::Character& c) -> mm::CollisionGeometry {
            if (c.collision) {
              return *c.collision;
            } else {
              return {};
            }
          },
          ":return: A list of :class:`TaperedCapsule` representing the character's collision geometry.")
      .def(
          "with_blend_shape",
          [](const mm::Character& c,
             std::shared_ptr<mm::BlendShape> blendShape,
             int nShapes) {
            return c.withBlendShape(
                blendShape, nShapes < 0 ? INT_MAX : nShapes);
          },
          R"(Returns a character that uses the parameter transform to control the passed-in blend shape basis.
It can be used to solve for shapes and pose simultaneously.

:param blend_shape: Blend shape basis.
:param n_shapes: Max blend shapes to retain.  Pass -1 to keep all of them (but warning: the default allgender basis is quite large with hundreds of shapes).
)",
          py::arg("blend_shape"),
          py::arg("n_shapes") = -1)
      .def(
          "pose_mesh",
          &pymomentum::getPosedMesh,
          R"(Poses the mesh

:param joint_params: The (7*nJoints) joint parameters for the given pose.
:return: A :class:`Mesh` object with the given pose.)",
          py::arg("joint_params"))
      .def(
          "skin_points",
          &pymomentum::skinPoints,
          R"(Skins the points using the character's linear blend skinning.

:param character: A :class:`Character` with both a rest mesh and skinning weights.
:param skel_state: A torch.Tensor containing either a [nBatch x nJoints x 8] skeleton state or a [nBatch x nJoints x 4 x 4] transforms.
:param rest_vertices: An optional torch.Tensor containing the rest points; if not passed, the ones stored inside the character are used.
:return: The vertex positions in worldspace.
          )",
          py::arg("skel_state"),
          py::arg("rest_vertices") = std::optional<at::Tensor>{})
      .def(
          "scaled",
          &momentum::scaleCharacter,
          R"(Scale the character (mesh and skeleton) by the desired amount.

Note that this primarily be used when transforming the character into different units; if you
simply want to apply an identity-specific scale to the character, you should use the
'scale_global' parameter in the :class:`ParameterTransform`.

:return: a new :class:`Character` that has been scaled.
:param character: The character to be scaled.
:param scale: The scale to apply.)",
          py::arg("scale"))
      .def(
          "transformed",
          [](const momentum::Character& character,
             const Eigen::Matrix4f& xform) {
            return momentum::transformCharacter(
                character, Eigen::Affine3f(xform));
          },
          R"(Transform the character (mesh and skeleton) by the desired transformation matrix.

Note that this is primarily intended for transforming between different spaces (e.g. x-up vs y-up).
If you want to translate/rotate/scale a character, you should preferentially use the model parameters to do so.

:return: a new :class:`Character` that has been transformed.
:param character: The character to be transformed.
:param xform: The transform to apply.)",
          py::arg("xform"))
      .def(
          "rebind_skin",
          [](const momentum::Character& character) {
            momentum::Character result(character);
            result.initInverseBindPose();
            return result;
          },
          "Rebind the character's inverse bind pose from the resting skeleton pose.")
      .def(
          "find_locators",
          &getLocators,
          R"(Return the parents/offsets of the passed-in locators.

:param names: The names of the locators or joints.
:return: a pair [parents, offsets] of numpy arrays.)",
          py::arg("names"))
      .def(
          "apply_model_param_limits",
          &applyModelParameterLimits,
          R"(Clamp model parameters by parameter limits stored in Character.

Note the function is differentiable.

:param model_params: the (can be batched) body model parameters.
:return: clampled model parameters. Same tensor shape as the input.)",
          py::arg("model_params"))
      .def_property_readonly(
          "model_parameter_limits",
          &modelParameterLimits,
          R"(A tuple (min, max) where each is an nParameter-length ndarray containing the upper or lower limits for the model parameters.  Note that not all parameters will have limits; for those parameters (such as global translation) without limits, (-FLT_MAX, FLT_MAX) is returned.)")
      .def_property_readonly(
          "joint_parameter_limits",
          &jointParameterLimits,
          R"(A tuple (min, max) where each is an (nJoints x 7)-length ndarray containing the upper or lower limits for the joint parameters.

Note that not all parameters will have limits; for those parameters (such as global translation) without limits, (-FLT_MAX, FLT_MAX) is returned.

Note: In practice, most limits are enforced on the model parameters, but momentum's joint limit functionality permits applying limits to joint parameters also as a conveninence.  )")
      .def_static(
          "from_gltf_bytes",
          &loadGLTFCharacterFromBytes,
          py::call_guard<py::gil_scoped_release>(),
          R"(Load a character from a gltf byte array.

:parameter gltf_bytes: A :class:`bytes` containing the GLTF JSON/messagepack data.
:return: a valid Character.
      )",
          py::arg("gltf_bytes"))
      // toGLTF(character, fps, motion)
      .def_static(
          "to_gltf",
          &toGLTF,
          py::call_guard<py::gil_scoped_release>(),
          R"(Serialize a character as a GLTF using dictionary form.

:parameter character: A valid character.
:parameter fps: Frames per second for describing the motion.
:parameter motion: tuple of vector of parameter names and a P X T matrix. P is number of parameters, T is number of frames.
:parameter offsets: tuple of vector of joint names and a Vector of size J * 7 (Parameters per joint). Eg. for 3 joints, you would have 21 params.
:return: a GLTF representation of Character with motion
      )",
          py::arg("character"))
      // loadFBXCharacterFromFile(fbxFilename, modelFilename, locatorsFilename)
      .def_static(
          "load_fbx",
          &loadFBXCharacterFromFile,
          py::call_guard<py::gil_scoped_release>(),
          R"(Load a character from an FBX file.  Optionally pass in a separate model definition and locators file.

:parameter fbx_filename: .fbx file that contains the skeleton and skinned mesh; e.g. blue_man_s0.fbx.
:parameter model_filename: Configuration file that defines the parameter mappings and joint limits; e.g. character.cfg.
:parameter locators_filename: File containing the locators, e.g. character.locators.
:return: A valid Character.)",
          py::arg("fbx_filename"),
          py::arg("model_filename") = std::optional<std::string>{},
          py::arg("locators_filename") = std::optional<std::string>{},
          py::arg("permissive") = false)
      // loadFBXCharacterFromFileWithMotion(fbxFilename, modelFilename,
      // locatorsFilename)
      .def_static(
          "load_fbx_with_motion",
          &loadFBXCharacterWithMotionFromFile,
          py::call_guard<py::gil_scoped_release>(),
          R"(Load a character and animation curves from an FBX file.

:parameter fbx_filename: .fbx file that contains the skeleton and skinned mesh; e.g. blue_man_s0.fbx.
:return: A valid Character, a vector of motions in the format of nFrames X nNumJointParameters, and fps. The caller needs to decide how to handle the joint parameters.)",
          py::arg("fbx_filename"),
          py::arg("permissive") = false)

      .def_static(
          "load_fbx_with_motion_from_bytes",
          &loadFBXCharacterWithMotionFromBytes,
          py::call_guard<py::gil_scoped_release>(),
          R"(Load a character and animation curves from an FBX file.

:parameter fbx_bytes: A Python bytes that is an .fbx file containing the skeleton and skinned mesh.
:return: A valid Character, a vector of motions in the format of nFrames X nNumJointParameters, and fps. The caller needs to decide how to handle the joint parameters.)",
          py::arg("fbx_bytes"),
          py::arg("permissive") = false)

      // loadFBXCharacterFromBytes(fbxBytes)
      .def_static(
          "load_fbx_from_bytes",
          &pymomentum::loadFBXCharacterFromBytes,
          py::call_guard<py::gil_scoped_release>(),
          R"(Load a character from byte array for an FBX file.

:parameter fbx_bytes: An array of bytes in FBX format.
:return: A valid Character.)",
          py::arg("fbx_bytes"),
          py::arg("permissive") = false)
      // loadLocatorsFromFile(character, locatorFile)
      .def(
          "load_locators",
          &pymomentum::loadLocatorsFromFile,
          py::call_guard<py::gil_scoped_release>(),
          R"(Load locators from a .locators file.

:parameter character: The character to map the locators onto.
:parameter filename: Filename for the locators.
:return: A valid Character.)",
          py::arg("filename"))
      // loadLocatorsFromBytes(character, locatorBytes)
      .def(
          "load_locators_from_bytes",
          &pymomentum::loadLocatorsFromBytes,
          py::call_guard<py::gil_scoped_release>(),
          R"(Load locators from a .locators file.

:parameter character: The character to map the locators onto.
:parameter locator_bytes: A byte array containing the locators.
:return: A valid Character.)",
          py::arg("locator_bytes"))
      // localModelDefinitionFromFile(character, modelFile)
      .def(
          "load_model_definition",
          &pymomentum::loadConfigFromFile,
          py::call_guard<py::gil_scoped_release>(),
          R"(Load a model definition from a .model file.  This defines the parameter transform, model parameters, and joint limits.

:parameter character: The character containing a valid skeleton.
:parameter filename: Filename for the model definition.
:return: A valid Character.)",
          py::arg("filename"))
      // localModelDefinitionFromBytes(character, modelBytes)
      .def(
          "load_model_definition_from_bytes",
          &pymomentum::loadConfigFromBytes,
          py::call_guard<py::gil_scoped_release>(),
          R"(Load a model definition from a .model file.  This defines the parameter transform, model parameters, and joint limits.

:parameter character: The character containing a valid skeleton.
:parameter model_bytes: Bytes array containing the model definition.
:return: A valid Character.)",
          py::arg("model_bytes"))
      // loadCharacterWithMotion(gltfFilename)
      .def_static(
          "load_gltf_with_motion",
          &loadCharacterWithMotion,
          py::call_guard<py::gil_scoped_release>(),
          R"(Load a character and a motion sequence from a gltf file.

:parameter gltfFilename: A .gltf file; e.g. character_s0.glb.
:return: a tuple [Character, motion, identity, fps], where motion is the motion matrix [nFrames x nParams] and identity is a JointParameter at rest pose.
      )",
          py::arg("gltf_filename"))
      // loadGLTFCharacterFromFile(filename)
      .def_static(
          "load_gltf",
          &loadGLTFCharacterFromFile,
          py::call_guard<py::gil_scoped_release>(),
          R"(Load a character from a gltf file.

:parameter path: A .gltf file; e.g. character_s0.glb.
      )",
          py::arg("path"))
      // saveGLTFCharacterToFile(filename, character)
      .def_static(
          "save_gltf",
          &saveGLTFCharacterToFile,
          py::call_guard<py::gil_scoped_release>(),
          R"(Save a character to a gltf file.

:parameter path: A .gltf export filename.
:parameter character: A Character to be saved to the output file.
:parameter fps: Frequency in frames per second
:parameter motion: Pose array in [n_frames x n_parameters]
:parameter offsets: Offset array in [n_joints x n_parameters_per_joint]
:parameter markers: Additional marker (3d positions) data in [n_frames][n_markers]
      )",
          py::arg("path"),
          py::arg("character"),
          py::arg("fps") = 120.f,
          py::arg("motion") = std::optional<momentum::MotionParameters>{},
          py::arg("offsets") =
              std::optional<const momentum::IdentityParameters>{},
          py::arg("markers") =
              std::optional<const std::vector<std::vector<momentum::Marker>>>{})
      .def_static(
          "save_gltf_from_skel_states",
          &saveGLTFCharacterToFileFromSkelStates,
          py::call_guard<py::gil_scoped_release>(),
          R"(Save a character to a gltf file.

:parameter path: A .gltf export filename.
:parameter character: A Character to be saved to the output file.
:parameter fps: Frequency in frames per second
:parameter skel_states: Skeleton states [n_frames x n_joints x n_parameters_per_joint]
:parameter joint_params: Joint parameters [n_joints x n_parameters_per_joint]
:parameter markers: Additional marker (3d positions) data in [n_frames][n_markers]
      )",
          py::arg("path"),
          py::arg("character"),
          py::arg("fps"),
          py::arg("skel_states"),
          py::arg("joint_params"),
          py::arg("markers") =
              std::optional<const std::vector<std::vector<momentum::Marker>>>{})
      .def_static(
          "save_fbx",
          &saveFBXCharacterToFile,
          py::call_guard<py::gil_scoped_release>(),
          R"(Save a character to an fbx file.

:parameter path: An .fbx export filename.
:parameter character: A Character to be saved to the output file.
:parameter fps: Frequency in frames per second
:parameter motion: [Optional] 2D pose matrix in [n_frames x n_parameters]
:parameter offsets: [Optional] Offset array in [(n_joints x n_parameters_per_joint)]
      )",
          py::arg("path"),
          py::arg("character"),
          py::arg("fps") = 120.f,
          py::arg("motion") = std::optional<const Eigen::MatrixXf>{},
          py::arg("offsets") = std::optional<const Eigen::VectorXf>{})
      .def_static(
          "save_fbx_with_joint_params",
          &saveFBXCharacterToFileWithJointParams,
          py::call_guard<py::gil_scoped_release>(),
          R"(Save a character to an fbx file with joint params.

:parameter path: An .fbx export filename.
:parameter character: A Character to be saved to the output file.
:parameter fps: Frequency in frames per second
:parameter joint_params: [Optional] 2D pose matrix in [n_frames x n_parameters]
      )",
          py::arg("path"),
          py::arg("character"),
          py::arg("fps") = 120.f,
          py::arg("joint_params") = std::optional<const Eigen::MatrixXf>{})
      .def(
          "simplify",
          [](const momentum::Character& character,
             std::optional<at::Tensor> enabledParamsTensor)
              -> momentum::Character {
            momentum::ParameterSet enabledParams;
            if (enabledParamsTensor) {
              enabledParams = tensorToParameterSet(
                  character.parameterTransform, *enabledParamsTensor);
            } else {
              enabledParams.set();
            }
            return character.simplify(enabledParams);
          },
          R"(Simplifies the character by removing extra joints; this can help to speed up IK.

:parameter enabled_parameters: Model parameters to be kept in the simplified model.  Defaults to including all parameters.
:return: a new :class:`Character` with extraneous joints removed.)",
          py::arg("enabled_parameters") = std::optional<at::Tensor>{});

  // =====================================================
  // momentum::Joint
  // - name
  // - parent
  // - preRotation ((x, y, z), w)
  // - translationOffset
  // =====================================================

  jointClass
      .def(
          py::init([](const std::string& name,
                      const int parent,
                      const Eigen::Vector4f& preRotation,
                      const Eigen::Vector3f& translationOffset) {
            return momentum::Joint{
                name,
                parent == -1 ? mm::kInvalidIndex : parent,
                {preRotation[3],
                 preRotation[0],
                 preRotation[1],
                 preRotation[2]},
                translationOffset};
          }),
          py::arg("name"),
          py::arg("parent"),
          py::arg("pre_rotation"),
          py::arg("translation_offset"))
      .def_property_readonly(
          "name",
          [](const mm::Joint& joint) { return joint.name; },
          "Returns the name of the joint.")
      .def_property_readonly(
          "parent",
          [](const mm::Joint& joint) {
            return joint.parent == mm::kInvalidIndex ? -1 : joint.parent;
          },
          "Returns the index of the parent joint (-1 for the parent)")
      .def_property_readonly(
          "pre_rotation",
          [](const mm::Joint& joint) {
            return Eigen::Vector4f(
                joint.preRotation.x(),
                joint.preRotation.y(),
                joint.preRotation.z(),
                joint.preRotation.w());
          },
          "Returns the pre-rotation for this joint in default pose of the character. Quaternion format: (x, y, z, w)")
      .def_property_readonly(
          "translation_offset",
          [](const mm::Joint& joint) { return joint.name; },
          "Returns the translation offset for this joint in default pose of the character.");

  // =====================================================
  // momentum::Skeleton
  // - size
  // - joint_names
  // - joint_parents
  // - get_parent(joint_index)
  // - get_child_joints(rootJointIndex, recursive)
  // - upper_body_joints
  // =====================================================
  skeletonClass
      .def(
          py::init([](const std::vector<mm::Joint>& jointList) {
            return mm::Skeleton(jointList);
          }),
          py::arg("joint_list"))
      .def_property_readonly(
          "size",
          [](const mm::Skeleton& skel) { return skel.joints.size(); },
          "Returns the number of joints in the skeleton.")
      .def(
          "__len__",
          [](const mm::Skeleton& skel) { return skel.joints.size(); },
          "Returns the number of joints in the skeleton.")
      .def_property_readonly(
          "joint_names",
          [](const mm::Skeleton& skel) { return skel.getJointNames(); },
          "Returns a list of joint names in the skeleton.")
      .def_property_readonly(
          "joint_parents",
          [](const mm::Skeleton& skel) -> std::vector<int64_t> {
            // For the root joint, we'll use -1 as the reported parent; this
            // just makes a lot more sense in a Python context where it would be
            // hard to compare against SIZE_MAX (and you're relying on the
            // typesystem to keep it as a uint64_t instead of an int64_t which
            // seems unreliable).
            std::vector<int64_t> result(skel.joints.size(), -1);
            for (size_t i = 0; i < skel.joints.size(); ++i) {
              const auto parent = skel.joints[i].parent;
              if (parent != momentum::kInvalidIndex) {
                result[i] = parent;
              }
            }
            return result;
          },
          ":return: the parent of each joint in the skeleton.  The root joint has parent -1.")
      .def(
          "joint_index",
          [](const mm::Skeleton& skel,
             const std::string& name,
             bool allow_missing = false) -> int {
            auto result = skel.getJointIdByName(name);
            if (result == momentum::kInvalidIndex) {
              if (allow_missing) {
                return -1;
              } else {
                throw std::runtime_error(
                    "Joint '" + name + "' not found in skeleton.");
              }
            } else {
              return result;
            }
          },
          "Get the joint index for a given joint name.  Returns -1 if joint is not found and allow_missing is True.",
          py::arg("name"),
          py::arg("allow_missing") = false)
      .def(
          "get_parent",
          [](const mm::Skeleton& skel, int jointIndex) -> int64_t {
            if (jointIndex < 0 || jointIndex >= skel.joints.size()) {
              std::ostringstream oss;
              oss << "get_parent() called with invalid joint index "
                  << jointIndex;
              throw std::runtime_error(oss.str());
            }
            const auto parent = skel.joints[jointIndex].parent;
            if (parent == momentum::kInvalidIndex) {
              return -1;
            } else {
              return static_cast<int64_t>(parent);
            }
          },
          R"(Get the parent joint index of the given joint. Return -1 for root.

:param joint_index: the index of a skeleton joint.
:return: The index of the parent joint, or -1 if it is the root of the skeleton. )",
          py::arg("joint_index"))
      .def(
          "get_child_joints",
          &mm::Skeleton::getChildrenJoints,
          R"(Find all joints parented under the given joint.

:return: A list of integers, one per joint. )",
          py::arg("root_joint_index"),
          py::arg("recursive"))
      .def(
          "is_ancestor",
          &mm::Skeleton::isAncestor,
          R"(Checks if one joint is an ancestor of another, inclusive.

:param joint_index: The index of a skeleton joint.
:param ancestor_joint_index: The index of a possible ancestor joint.

:return: true if ancestorJointId is an ancestor of jointId; that is,
if jointId is in the tree rooted at ancestorJointId.
Note that a joint is considered to be its own ancestor; that is,
isAncestor(id, id) returns true. )",
          py::arg("joint_index"),
          py::arg("ancestor_joint_index"))
      .def_property_readonly(
          "upper_body_joints",
          &getUpperBodyJoints,
          R"(Convenience function to get all upper-body joints (defined as those parented under 'b_spine0').

:return: A list of integers, one per joint.)")
      .def_property_readonly(
          "offsets",
          [](const mm::Skeleton& skeleton) {
            std::vector<Eigen::Vector3f> translationOffsets;
            std::transform(
                skeleton.joints.cbegin(),
                skeleton.joints.cend(),
                std::back_inserter(translationOffsets),
                [](const mm::Joint& joint) { return joint.translationOffset; });
            return pymomentum::asMatrix(translationOffsets);
          },
          "Returns skeleton joint offsets tensor for all joints (num_joints, 3")
      .def_property_readonly(
          "pre_rotations",
          [](const mm::Skeleton& skeleton) {
            std::vector<Eigen::Vector4f> preRotations;
            std::transform(
                skeleton.joints.cbegin(),
                skeleton.joints.cend(),
                std::back_inserter(preRotations),
                [](const mm::Joint& joint) {
                  return joint.preRotation.coeffs();
                });
            return pymomentum::asMatrix(preRotations);
          },
          "Returns skeleton joint offsets tensor for all joints shape: (num_joints, 4)");

  // =====================================================
  // momentum::SkinWeights
  // - weight
  // - index
  // =====================================================
  skinWeightsClass
      .def(
          py::init(
              [](const Eigen::MatrixXi& index, const Eigen::MatrixXf& weights) {
                return mm::SkinWeights{index.cast<uint32_t>(), weights};
              }),
          py::arg("index"),
          py::arg("weights"))
      .def_property_readonly(
          "weight",
          [](const mm::SkinWeights& skinning) { return skinning.weight; },
          "Returns the skinning weights.")
      .def_property_readonly(
          "index",
          [](const mm::SkinWeights& skinning) { return skinning.index; },
          "Returns the skinning indices.");

  // =====================================================
  // momentum::Mesh
  // - vertices
  // - normals
  // - faces
  // - colors
  // =====================================================
  meshClass
      .def(
          py::init([](const RowMatrixf& vertices,
                      const RowMatrixf& normals,
                      const RowMatrixi& faces,
                      const std::vector<std::vector<int32_t>>& lines,
                      const RowMatrixb& colors,
                      const std::vector<float>& confidence,
                      const RowMatrixf& texcoords,
                      const RowMatrixi& texcoord_faces,
                      const std::vector<std::vector<int32_t>>& texcoord_lines) {
            mm::Mesh mesh;
            const auto nVerts = vertices.rows();
            if (normals.rows() != nVerts) {
              throw std::runtime_error(
                  "vertices and normals must have the same number of rows");
            }

            if (vertices.cols() != 3) {
              throw std::runtime_error("vertices must have size n x 3");
            }

            if (normals.cols() != 3) {
              throw std::runtime_error("normals must have size n x 3");
            }

            if (faces.cols() != 3) {
              throw std::runtime_error("faces must have size n x 3");
            }

            if (faces.size() > 0 && faces.maxCoeff() >= nVerts) {
              throw std::runtime_error("face index exceeded vertex count");
            }

            mesh.vertices = asVectorList<float, 3>(vertices);
            mesh.normals = asVectorList<float, 3>(normals);
            mesh.faces = asVectorList<int, 3>(faces);

            for (const auto& l : lines) {
              if (!l.empty() &&
                  *std::max_element(l.begin(), l.end()) >= nVerts) {
                throw std::runtime_error("line index exceeded vertex count");
              }
            }
            mesh.lines = lines;

            if (colors.rows() != 0 && colors.cols() != nVerts) {
              throw std::runtime_error(
                  "colors should be empty or equal to the number of vertices");
            }
            mesh.colors = asVectorList<uint8_t, 3>(colors);

            if (confidence.size() != 0 && confidence.size() != nVerts) {
              throw std::runtime_error(
                  "confidence should be empty or equal to the number of vertices");
            }
            mesh.confidence = confidence;

            if (texcoords.size() != 0 && texcoords.cols() != 2) {
              throw std::runtime_error(
                  "texcoords should be empty or must have size n x 2");
            }

            const auto nTextureCoords = texcoords.rows();
            if (texcoord_faces.size() != 0 &&
                texcoord_faces.rows() != faces.rows()) {
              throw std::runtime_error(
                  "texcoords_faces should be empty or equal to the size of faces");
            }

            if (texcoord_faces.size() != 0 && texcoord_faces.cols() != 3) {
              throw std::runtime_error(
                  "texcoord_faces should be empty or must have size n x 3");
            }

            if (texcoord_faces.size() > 0 &&
                texcoord_faces.maxCoeff() >= nTextureCoords) {
              throw std::runtime_error(
                  "texcoord_face index exceeded texcoord count");
            }
            mesh.texcoords = asVectorList<float, 2>(texcoords);
            mesh.texcoord_faces = asVectorList<int32_t, 3>(texcoord_faces);
            mesh.texcoord_lines = texcoord_lines;

            return mesh;
          }),
          R"(
:parameter vertices: n x 3 array of vertex locations.
:parameter normals: n x 3 array of vertex normals.
:parameter faces: n x 3 array of triangles.
:parameter lines: list of lines, where each line is a list of vertex indices.
:parameter colors: n x 3 array of vertex colors.
:parameter confidence: n x 1 array of vertex confidence values.
:parameter texcoords: n x 2 array of texture coordinates.
:parameter texcoord_faces: n x 3 array of triangles in the texture map.  Each triangle corresponds to a triangle on the mesh, but indices should refer to the texcoord array.
:parameter texcoord_lines: list of lines, where each line is a list of texture coordinate indices.
          )",
          py::arg("vertices"),
          py::arg("normals"),
          py::arg("faces"),
          py::kw_only(),
          py::arg("lines") = std::vector<std::vector<int>>{},
          py::arg("colors") = RowMatrixb{},
          py::arg("confidence") = std::vector<float>{},
          py::arg("texcoords") = RowMatrixf{},
          py::arg("texcoord_faces") = RowMatrixi{},
          py::arg("texcoord_lines") = RowMatrixf{})
      .def_property_readonly(
          "n_vertices",
          [](const mm::Mesh& mesh) { return mesh.vertices.size(); },
          ":return: The number of vertices in the mesh.")
      .def_property_readonly(
          "n_faces",
          [](const mm::Mesh& mesh) { return mesh.faces.size(); },
          ":return: The number of faces in the mesh.")
      .def_property_readonly(
          "vertices",
          [](const mm::Mesh& mesh) {
            return pymomentum::asMatrix(mesh.vertices);
          },
          ":return: The vertices of the mesh in a [n x 3] numpy array.")
      .def_property_readonly(
          "normals",
          [](const mm::Mesh& mesh) {
            return pymomentum::asMatrix(mesh.normals);
          },
          ":return: The per-vertex normals of the mesh in a [n x 3] numpy array.")
      .def_property_readonly(
          "faces",
          [](const mm::Mesh& mesh) { return pymomentum::asMatrix(mesh.faces); },
          ":return: The triangles of the mesh in an [n x 3] numpy array.")
      .def_readonly(
          "lines", &mm::Mesh::lines, "list of list of vertex indices per line")
      .def_property_readonly(
          "colors",
          [](const mm::Mesh& mesh) {
            return pymomentum::asMatrix(mesh.colors);
          },
          ":return: Per-vertex colors if available; returned as a (possibly empty) [n x 3] numpy array.")
      .def_readonly(
          "confidence", &mm::Mesh::confidence, "list of per-vertex confidences")
      .def_property_readonly(
          "texcoords",
          [](const mm::Mesh& mesh) {
            return pymomentum::asMatrix(mesh.texcoords);
          },
          "texture coordinates as m x 3 array.  Note that the number of texture coordinates may "
          "be different from the number of vertices as there can be cuts in the texture map.  "
          "Use texcoord_faces to index the texture coordinates.")
      .def_property_readonly(
          "texcoord_faces",
          [](const mm::Mesh& mesh) {
            return pymomentum::asMatrix(mesh.texcoord_faces);
          },
          "n x 3 faces in the texture map.  Each face maps 1-to-1 to a face in the original "
          "mesh but indexes into the texcoords array.")
      .def_readonly(
          "texcoord_lines",
          &mm::Mesh::texcoord_lines,
          "Texture coordinate indices for each line.  ");

  blendShapeClass
      .def_property_readonly(
          "base_shape",
          [](const mm::BlendShape& blendShape) {
            return pymomentum::asMatrix(blendShape.getBaseShape());
          },
          ":return: The base shape of the blend shape solver.")
      .def_property_readonly(
          "shape_vectors",
          [](const mm::BlendShape& blendShape) -> py::array_t<float> {
            const Eigen::MatrixXf& shapeVectors = blendShape.getShapeVectors();
            const Eigen::Index nVerts = shapeVectors.rows() / 3;
            if (shapeVectors.rows() % 3 != 0) {
              throw std::runtime_error("Invalid blend shape basis.");
            }
            py::array_t<float> result(
                std::vector<ptrdiff_t>{shapeVectors.cols(), nVerts, 3});
            py::buffer_info buf = result.request();
            memcpy(buf.ptr, shapeVectors.data(), result.nbytes());
            return result;
          },
          ":return: The base shape of the blend shape solver.")
      .def_static(
          "load",
          &loadBlendShapeFromFile,
          R"(Load a blend shape basis from a file.

:parameter blend_shape_path: The path to a blend shape file.
:parameter num_expected_shapes: Trim the shape basis if it contains more shapes than this.  Pass -1 (the default) to leave the shapes untouched.
:parameter num_expected_vertices: Trim the shape basis if it contains more vertices than this.  Pass -1 (the default) to leave the shapes untouched.
:return: A :class:`BlendShape`.)",
          py::arg("path"),
          py::arg("num_expected_shapes") = -1,
          py::arg("num_expected_vertices") = -1)
      .def_static(
          "from_bytes",
          &loadBlendShapeFromBytes,
          R"(Load a blend shape basis from bytes in memory.

:parameter blend_shape_bytes: A chunk of bytes containing the blend shape basis.
:parameter num_expected_shapes: Trim the shape basis if it contains more shapes than this.  Pass -1 (the default) to leave the shapes untouched.
:parameter num_expected_vertices: Trim the shape basis if it contains more vertices than this.  Pass -1 (the default) to leave the shapes untouched.
:return: a :class:`BlendShape`.)",
          py::arg("blend_shape_bytes"),
          py::arg("num_expected_shapes") = -1,
          py::arg("num_expected_vertices") = -1)
      .def_static(
          "from_tensors",
          &loadBlendShapeFromTensors,
          R"(Create a blend shape basis from numpy.ndarrays.

:parameter base_shape: A [nPts x 3] ndarray containing the base shape.
:parameter shape_vectors: A [nShapes x nPts x 3] ndarray containing the blend shape basis.
:return: a :class:`BlendShape`.)",
          py::arg("base_shape"),
          py::arg("shape_vectors"))
      .def_property_readonly(
          "n_shapes",
          [](const mm::BlendShape& blendShape) {
            return blendShape.shapeSize();
          },
          "Number of shapes in the blend shape basis.")
      .def_property_readonly(
          "n_vertices",
          [](const mm::BlendShape& blendShape) {
            return blendShape.modelSize();
          },
          "Number of vertices in the mesh.")
      .def(
          "compute_shape",
          [](py::object blendShape, at::Tensor coeffs) {
            return applyBlendShapeCoefficients(blendShape, coeffs);
          },
          R"(Apply the blend shape coefficients to compute the rest shape.

The resulting shape is equal to the base shape plus a linear combination of the shape vectors.

:parameter coeffs: A torch.Tensor of size [n_batch x n_shapes] containing blend shape coefficients.
:result: A [n_batch x n_vertices x 3] tensor containing the vertex positions.)",
          py::arg("coeffs"));

  // =====================================================
  // momentum::Locator
  // - name
  // - parent
  // - offset
  // =====================================================
  locatorClass
      .def(
          py::init<
              const std::string&,
              const size_t,
              const Eigen::Vector3f&,
              const Eigen::Vector3i&,
              float,
              const Eigen::Vector3f&,
              const Eigen::Vector3f&>(),
          py::arg("name") = "uninitialized",
          py::arg("parent") = mm::kInvalidIndex,
          py::arg("offset") = Eigen::Vector3f::Zero(),
          py::arg("locked") = Eigen::Vector3i::Zero(),
          py::arg("weight") = 1.0f,
          py::arg("limit_origin") = Eigen::Vector3f::Zero(),
          py::arg("limit_weight") = Eigen::Vector3f::Zero())
      .def_readonly("name", &mm::Locator::name, "The locator's name.")
      .def_readonly(
          "parent", &mm::Locator::parent, "The locator's parent joint index.")
      .def_readonly(
          "offset",
          &mm::Locator::offset,
          "The locator's offset to parent joint location.")
      .def_readonly(
          "locked",
          &mm::Locator::locked,
          "Flag per axes to indicate whether that axis can be moved during optimization or not.")
      .def_readonly(
          "weight",
          &mm::Locator::weight,
          "Weight for this locator during IK optimization.")
      .def_readonly(
          "limit_origin",
          &mm::Locator::limitOrigin,
          "Defines the limit reference position. equal to offset on loading.")
      .def_readonly(
          "limit_weight",
          &mm::Locator::limitOrigin,
          "Defines how close an unlocked locator should stay to it's original position")
      .def("__repr__", [](const mm::Locator& l) {
        std::ostringstream oss;
        oss << "[" << l.name << "; parent: " << l.parent
            << "; offset: " << l.offset.transpose() << "]";
        return oss.str();
      });

  // =====================================================
  // momentum::ParameterTransform
  // - names
  // - size()
  // - apply(modelParameters)
  // - getScalingParameters()
  // - getRigidParameters()
  // - getParametersForJoints(jointIndices)
  // - createInverseParameterTransform()
  // =====================================================
  parameterTransformClass
      .def(
          py::init([](const std::vector<std::string>& names,
                      const mm::Skeleton& skeleton,
                      const Eigen::SparseMatrix<float, Eigen::RowMajor>&
                          transform) {
            mm::ParameterTransform parameterTransform;
            parameterTransform.name = names;
            parameterTransform.transform.resize(
                static_cast<int>(skeleton.joints.size()) *
                    mm::kParametersPerJoint,
                static_cast<int>(names.size()));

            for (int i = 0; i < transform.outerSize(); ++i) {
              for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator
                       it(transform, i);
                   it;
                   ++it) {
                parameterTransform.transform.coeffRef(
                    static_cast<long>(it.row()), static_cast<long>(it.col())) =
                    it.value();
              }
            }

            parameterTransform.offsets.setZero(
                skeleton.joints.size() * mm::kParametersPerJoint);
            return parameterTransform;
          }),
          py::arg("names"),
          py::arg("skeleton"),
          py::arg("transform"))
      .def_readonly(
          "names",
          &mm::ParameterTransform::name,
          "List of model parameter names")
      .def_property_readonly(
          "size",
          &mm::ParameterTransform::numAllModelParameters,
          "Size of the model parameter vector.")
      .def(
          "apply",
          [](const mm::ParameterTransform* paramTransform,
             torch::Tensor modelParams) -> torch::Tensor {
            return applyParamTransform(paramTransform, modelParams);
          },
          R"(Apply the parameter transform to a k-dimensional model parameter vector (returns the 7*nJoints joint parameter vector).

The modelParameters store the reduced set of parameters (typically around 50) that are actually
optimized in the IK step.

The jointParameters are stored (tx, ty, tz; rx, ry, rz; s) and each represents the transform relative to the parent joint.
Rotations are in Euler angles.)",
          py::arg("model_parameters"))
      .def_property_readonly(
          "scaling_parameters",
          &getScalingParameters,
          "Boolean torch.Tensor indicating which parameters are used to control the character's scale.")
      .def_property_readonly(
          "rigid_parameters",
          &getRigidParameters,
          "Boolean torch.Tensor indicating which parameters are used to control the character's rigid transform (translation and rotation).")
      .def_property_readonly(
          "all_parameters",
          &getAllParameters,
          "Boolean torch.Tensor with all parameters enabled.")
      .def_property_readonly(
          "blend_shape_parameters",
          &getBlendShapeParameters,
          "Boolean torch.Tensor with just the blend shape parameters enabled.")
      .def_property_readonly(
          "pose_parameters",
          &getPoseParameters,
          "Boolean torch.Tensor with all the parameters used to pose the body, excluding and scaling, blend shape, or physics parameters.")
      .def_property_readonly(
          "no_parameters",
          [](const momentum::ParameterTransform& parameterTransform) {
            return parameterSetToTensor(
                parameterTransform, momentum::ParameterSet());
          },
          "Boolean torch.Tensor with no parameters enabled.")
      .def_property_readonly(
          "parameter_sets",
          &getParameterSets,
          R"(A dictionary mapping names to sets of parameters (as a boolean torch.Tensor) that are defined in the .model file.
This is convenient for turning off certain body features; for example the 'fingers' parameters
can be used to enable/disable finger motion in the character model.  )")
      .def(
          "parameters_for_joints",
          &getParametersForJoints,
          R"(Gets a boolean torch.Tensor indicating which parameters affect the passed-in joints.

:param jointIndices: List of integers of skeleton joints.)",
          py::arg("joint_indices"))
      .def(
          "find_parameters",
          &findParameters,
          R"(Return a boolean tensor with the named parameters set to true.

:param parameter_names: Names of the parameters to find.
:param allow_missng: If false, missing parameters will throw an exception.
        )",
          py::arg("names"),
          py::arg("allow_missing") = false)
      .def(
          "inverse",
          &createInverseParameterTransform,
          R"(Compute the inverse of the parameter transform (a mapping from joint parameters to model parameters).

:return: The inverse parameter transform.)")
      .def_property_readonly(
          "transform",
          &getParameterTransformTensor,
          "Returns the parameter transform matrix which when applied maps model parameters to joint parameters.");
  ;

  // =====================================================
  // momentum::InverseParameterTransform
  // - apply()
  // =====================================================
  inverseParameterTransformClass.def(
      "apply",
      &applyInverseParamTransform,
      R"(Apply the inverse parameter transform to a 7*nJoints-dimensional joint parameter vector (returns the k-dimensional model parameter vector).

Because the number of joint parameters is much larger than the number of model parameters, this will in general have a non-zero residual.

:param joint_parameters: Joint parameter tensor with dimensions (nBatch x 7*nJoints).
:return: A torch.Tensor containing the (nBatch x nModelParameters) model parameters.)",
      py::arg("joint_parameters"));

  // =====================================================
  // momentum::Mppca
  // - Mppca()
  // - Mppca(pi, mu, W, sigma2, names)
  // - numModels
  // - dimension
  // - names
  // - getModel(iModel)
  // =====================================================
  py::class_<mm::Mppca, std::shared_ptr<mm::Mppca>>(
      m,
      "Mppca",
      R"(Probability distribution over poses, used by the PosePriorErrorFunction.
Currently contains a mixture of probabilistic PCA models.

Each PPCA model is a Gaussian with mean mu and covariance (sigma^2*I + W*W^T).
)")
      .def(py::init())
      .def(
          py::init(&createMppcaModel),
          R"(Construct an Mppca model from numpy arrays.
:param pi: The (nModels) mixture weights
:param mu: The (nModels x dimension) mean vectors.
:param W: The (nModels x dimension x nParams) weight matrices.
:param sigma2: The (nModels) squared sigma uniform variance parameters.
:param names: The (nDimension) names of the affected parameters.
)",
          py::arg("pi"),
          py::arg("mu"),
          py::arg("W"),
          py::arg("sigma"),
          py::arg("names"))
      .def_readonly(
          "n_mixtures",
          &mm::Mppca::p,
          R"(The number of individual Gaussians in the mixture model.)")
      .def_readonly(
          "n_dimension",
          &mm::Mppca::d,
          R"(The dimension of the parameter space.)")
      .def_readonly(
          "names", &mm::Mppca::names, R"(The names of the parameters.)")
      .def(
          "to_tensors",
          &mppcaToTensors,
          R"(Return the parameters defining the mixture of probabilistic PCA models.

Each PPCA model a Gaussian N(mu, cov) where the covariance matrix is
(sigma*sigma*I + W * W^T).  pi is the mixture weight for this particular Gaussian.

Note that mu is a vector of length :meth:`dimension` and W is a matrix of dimension :meth:`dimension` x q
where q is the dimensionality of the PCA subspace.

The resulting tensors are as follows:

* pi: a [n]-dimensional tensor containing the mixture weights.  It sums to 1.
* mu: a [n x d]-dimensional tensor containing the mean pose for each mixture.
* weights: a [n x d x q]-dimensional tensor containing the q vectors spanning the PCA space.
* sigma: a [n]-dimensional tensor containing the uniform part of the covariance matrix.
* param_idx: a [d]-dimensional tensor containing the indices of the parameters.

:param parameter_transform: An optional parameter transform used to map the parameters; if not present, then the param_idx tensor will be empty.
:return: an tuple (pi, mean, weights, sigma, param_idx) for the Probabilistic PCA model.)",
          py::arg("parameter_transform") =
              std::optional<const mm::ParameterTransform*>())
      .def("get_mixture", &getMppcaModel, py::arg("i_model"))
      .def_static(
          "load",
          &loadPosePriorFromFile,
          "Load a mixture PCA model (e.g. poseprior.mppca).",
          py::arg("mppca_filename"))
      .def_static(
          "from_bytes",
          &loadPosePriorFromBytes,
          "Load a mixture PCA model (e.g. poseprior.mppca).",
          py::arg("mppca_bytes"));

  // Class TaperedCapsule, defining the properties:
  //    transformation
  //    radius
  //    parent
  //    length
  capsuleClass
      .def_property_readonly(
          "transformation",
          [](const mm::TaperedCapsule& capsule) -> Eigen::Matrix4f {
            return capsule.transformation.matrix();
          })
      .def_property_readonly(
          "radius",
          [](const mm::TaperedCapsule& capsule) -> Eigen::Vector2f {
            return capsule.radius;
          })
      .def_property_readonly(
          "parent",
          [](const mm::TaperedCapsule& capsule) -> int {
            return capsule.parent;
          })
      .def_property_readonly(
          "length", [](const mm::TaperedCapsule& capsule) -> float {
            return capsule.length;
          });

  // Class Marker, defining the properties:
  //    name
  //    pos
  //    occluded
  markerClass.def(py::init())
      .def(py::init<const std::string&, const Eigen::Vector3d&, const bool>())
      .def_readwrite("name", &mm::Marker::name, "Name of the marker")
      .def_readwrite("pos", &mm::Marker::pos, "Marker 3d position")
      .def_readwrite(
          "occluded",
          &mm::Marker::occluded,
          "True if the marker is occluded with no position info");

  // Class MarkerSequence, defining the properties:
  //    name
  //    frames
  //    fps
  markerSequenceClass.def(py::init())
      .def_readwrite("name", &mm::MarkerSequence::name, "Name of the subject")
      .def_readwrite(
          "frames",
          &mm::MarkerSequence::frames,
          "Marker data in [nframes][nMarkers]")
      .def_readwrite("fps", &mm::MarkerSequence::fps, "Frame rate");

  // loadMotion(gltfFilename)
  m.def(
      "load_motion",
      &loadMotion,
      R"(Load a motion sequence from a gltf file.

Unless you can guarantee that the parameters in the motion files match your existing character,
you will likely want to retarget the parameters using the :meth:`mapParameters` function.

:parameter gltfFilename: A .gltf file; e.g. character_s0.glb.
:return: a tuple [motionData, motionParameterNames, identityData, identityParameterNames].
      )",
      py::arg("gltf_filename"));

  // loadMarkersFromFile(path, mainSubjectOnly)
  // TODO(T138941756): Expose the loadMarker and loadMarkersForMainSubject
  // APIs separately from markerIO.h loadMarkersFromFile(path,
  // mainSubjectOnly)
  m.def(
      "load_markers",
      &loadMarkersFromFile,
      R"(Load 3d mocap marker data from file.

:param path: A marker data file: .c3d, .gltf, or .trc.
:param mainSubjectOnly: True to load only one subject's data.
:return: an array of MarkerSequence, one per subject in the file.
      )",
      py::arg("path"),
      py::arg("main_subject_only") = true);

  // mapModelParameters_names(motionData, sourceParameterNames,
  // targetCharacter)
  m.def(
      "map_model_parameters",
      &mapModelParameters_names,
      R"(Remap model parameters from one character to another.

:param motionData: The source motion data as a nFrames x nParams torch.Tensor.
:param sourceParameterNames: The source parameter names as a list of strings (e.g. c.parameterTransform.name).
:param targetCharacter: The target character to remap onto.

:return: The motion with the parameters remapped to match the passed-in Character. The fields with no match are filled with zero.
      )",
      py::arg("motion_data"),
      py::arg("source_parameter_names"),
      py::arg("target_character"));

  // mapModelParameters(motionData, sourceCharacter, targetCharacter)
  m.def(
      "map_model_parameters",
      &mapModelParameters,
      R"(Remap model parameters from one character to another.

:param motionData: The source motion data as a nFrames x nParams torch.Tensor.
:param sourceCharacter: The source character.
:param targetCharacter: The target character to remap onto.

:return: The motion with the parameters remapped to match the passed-in Character. The fields with no match are filled with zero.
      )",
      py::arg("motion_data"),
      py::arg("source_character"),
      py::arg("target_character"));

  // mapJointParameters(motionData, sourceCharacter, targetCharacter)
  m.def(
      "map_joint_parameters",
      &mapJointParameters,
      R"(Remap joint parameters from one character to another.

:param motionData: The source motion data as a [nFrames x (nBones * 7)] torch.Tensor.
:param sourceCharacter: The source character.
:param targetCharacter: The target character to remap onto.

:return: The motion with the parameters remapped to match the passed-in Character. The fields with no match are filled with zero.
      )",
      py::arg("motion_data"),
      py::arg("source_character"),
      py::arg("target_character"));

  // createTestCharacter()
  m.def(
      "test_character",
      &momentum::createTestCharacter<float>,
      R"(Create a simple 3-joint test character.  This is useful for writing confidence tests that
execute quickly and don't rely on outside files.

The mesh is made by a few vertices on the line segment from (1,0,0) to (1,1,0) and a few dummy
faces. The skeleton has three joints: root at (0,0,0), joint1 parented by root, at world-space
(0,1,0), and joint2 parented by joint1, at world-space (0,2,0).
The character has only one parameter limit: min-max type [-0.1, 0.1] for root.

:parameter numJoints: The number of joints in the resulting character.
:return: A simple character with 3 joints and 10 model parameters.
      )",
      py::arg("num_joints") = 3);

  // createTestPosePrior()
  m.def(
      "create_test_mppca",
      &momentum::createDefaultPosePrior<float>,
      R"(Create a pose prior that acts on the simple 3-joint test character.

:return: A simple pose prior.)");

  // uniformRandomToModelParameters(character, unifNoise)
  m.def(
      "uniform_random_to_model_parameters",
      &uniformRandomToModelParameters,
      R"(Convert a uniform noise vector into a valid body pose.

:parameter character: The character to use.
:parameter unifNoise: A uniform noise tensor, with dimensions (nBatch x nModelParams).
:return: A torch.Tensor with dimensions (nBatch x nModelParams).)",
      py::arg("character"),
      py::arg("unif_noise"));

  m.def(
      "apply_parameter_transform",
      [](py::object character, at::Tensor modelParameters) {
        return applyParamTransform(character, modelParameters);
      },
      R"(Apply the parameter transform to a [nBatch x nParams] tensor of model parameters.
This is functionally identical to :meth:`ParameterTransform.apply` except that it allows
batching on the character.

:parameter character: A character or list of characters.
:type character: Union[Character, List[Character]]
:parameter modelParameters: A [nBatch x nParams] tensor of model parameters.

:return: a tensor of joint parameters.)",
      py::arg("character"),
      py::arg("model_parameters"));

  m.def(
      "model_parameters_to_blend_shape_coefficients",
      &modelParametersToBlendShapeCoefficients,
      R"(Extract the model parameters that correspond to the blend shape coefficients, in the order
required to call `meth:BlendShape.compute_shape`.

:param character: A character.
:parameter model_parameters: A [nBatch x nParams] tensor of model parameters.

:return: A [nBatch x nBlendShape] torch.Tensor of blend shape coefficients.)",
      py::arg("character"),
      py::arg("model_parameters"));

  // modelParametersToPositions(character, modelParameters, parents, offsets)
  m.def(
      "model_parameters_to_positions",
      &modelParametersToPositions,
      R"(Convert model parameters to 3D positions relative to skeleton joints using forward kinematics.  You can use this (for example) to
supervise a model to produce the correct 3D ground truth.

Working directly from modelParameters is preferable to mapping to jointParameters first because it does a better job exploiting the
sparsity in the model and therefore can be made somewhat faster.

:param character: Character to use.
:type character: Union[Character, List[Character]]
:param model_parameters: Model parameter tensor, with dimension (nBatch x nModelParams).
:param parents: Joint parents, on for each target position.
:param offsets: 3-d offset in each joint's local space.
:return: Tensor of size (nBatch x nParents x 3), representing the world-space position of each point.
)",
      py::arg("character"),
      py::arg("model_parameters"),
      py::arg("parents"),
      py::arg("offsets"));

  // jointParametersToPositions(character, jointParameters, parents, offsets)
  m.def(
      "joint_parameters_to_positions",
      &jointParametersToPositions,
      R"(Convert joint parameters to 3D positions relative to skeleton joints using forward kinematics.  You can use this (for example) to
supervise a model to produce the correct 3D ground truth.

You should prefer :meth:`model_parameters_to_positions` when working from modelParameters because it is better able to exploit sparsity; this
function is provided as a convenience because motion read from external files generally uses jointParameters.

:param character: Character to use.
:type character: Union[Character, List[Character]]
:param joint_parameters: Joint parameter tensor, with dimension (nBatch x (7*nJoints)).
:param parents: Joint parents, on for each target position.
:param offsets: 3-d offset in each joint's local space.
:return: Tensor of size (nBatch x nParents x 3), representing the world-space position of each point.
)",
      py::arg("character"),
      py::arg("joint_parameters"),
      py::arg("parents"),
      py::arg("offsets"));

  // modelParametersToSkeletonState(characters, modelParameters)
  m.def(
      "model_parameters_to_skeleton_state",
      &modelParametersToSkeletonState,
      R"(Map from the k modelParameters to the 8*nJoints global skeleton state.

The skeletonState is stored (tx, ty, tz; rx, ry, rz, rw; s) and each maps the transform from the joint's local space to worldspace.
Rotations are Quaternions in the ((x, y, z), w) format.  This is deliberately identical to the representation used in mopy.)

:param character: Character to use.
:type character: Union[Character, List[Character]]
:param model_parameters: torch.Tensor containing the (nBatch x nModelParameters) model parameters.

:return: torch.Tensor of size (nBatch x nJoints x 8) containing the skeleton state; should be also compatible with mopy's skeleton state representation.)",
      py::arg("character"),
      py::arg("model_parameters"));

  // modelParametersToLocalSkeletonState(characters, modelParameters)
  m.def(
      "model_parameters_to_local_skeleton_state",
      &modelParametersToLocalSkeletonState,
      R"(Map from the k modelParameters to the 8*nJoints local skeleton state.

The skeletonState is stored (tx, ty, tz; rx, ry, rz, rw; s) and each maps the transform from the joint's local space to its parent joint space.
Rotations are Quaternions in the ((x, y, z), w) format.  This is deliberately identical to the representation used in mopy.)

:param character: Character to use.
:type character: Union[Character, List[Character]]
:param model_parameters: torch.Tensor containing the (nBatch x nModelParameters) model parameters.

:return: torch.Tensor of size (nBatch x nJoints x 8) containing the skeleton state; should be also compatible with mopy's skeleton state representation.)",
      py::arg("character"),
      py::arg("model_parameters"));

  // jointParametersToSkeletonState(character, jointParameters)
  m.def(
      "joint_parameters_to_skeleton_state",
      &jointParametersToSkeletonState,
      R"(Map from the 7*nJoints jointParameters to the 8*nJoints global skeleton state.

The skeletonState is stored (tx, ty, tz; rx, ry, rz, rw; s) and each maps the transform from the joint's local space to worldspace.
Rotations are Quaternions in the ((x, y, z), w) format.  This is deliberately identical to the representation used in mopy.)

:param character: Character to use.
:type character: Union[Character, List[Character]]
:param joint_parameters: torch.Tensor containing the (nBatch x nJointParameters) joint parameters.

:return: torch.Tensor of size (nBatch x nJoints x 8) containing the skeleton state; should be also compatible with mopy's skeleton state representation.)",
      py::arg("character"),
      py::arg("joint_parameters"));

  // jointParametersToLocalSkeletonState(character, jointParameters)
  m.def(
      "joint_parameters_to_local_skeleton_state",
      &jointParametersToLocalSkeletonState,
      R"(Map from the 7*nJoints jointParameters (representing transforms to the parent joint) to the 8*nJoints local skeleton state.

The skeletonState is stored (tx, ty, tz; rx, ry, rz, rw; s) and each maps the transform from the joint's local space to its parent joint space.
Rotations are Quaternions in the ((x, y, z), w) format.  This is deliberately identical to the representation used in mopy.)

:param character: Character to use.
:type character: Union[Character, List[Character]]
:param joint_parameters: torch.Tensor containing the (nBatch x nJointParameters) joint parameters.

:return: torch.Tensor of size (nBatch x nJoints x 8) containing the skeleton state; should be also compatible with mopy's skeleton state representation.)",
      py::arg("character"),
      py::arg("joint_parameters"));

  // localSkeletonStateToJointParameters(character, skelState)
  m.def(
      "skeleton_state_to_joint_parameters",
      &skeletonStateToJointParameters,
      R"(Map from the 8*nJoints skeleton state (representing transforms to world-space) to the 7*nJoints jointParameters.  This performs the following operations:

* Removing the translation offset.
* Inverting out the pre-rotation.
* Converting to Euler angles.

The skeleton state is stored (tx, ty, tz; rx, ry, rz, rw; s) and transforms from the joint's local space to world-space.
The joint parameters are stored (tx, ty, tz; ry, rz, ry; s) where rotations are in Euler angles and are relative to the parent joint.

:param character: Character to use.
:param skel_state: torch.Tensor containing the ([nBatch] x nJoints x 8) skeleton state.

:return: torch.Tensor of size ([nBatch] x nJoints x 7) containing the joint parameters.)",
      py::arg("character"),
      py::arg("skel_state"));

  // localSkeletonStateToJointParameters(character, skelState)
  m.def(
      "local_skeleton_state_to_joint_parameters",
      &localSkeletonStateToJointParameters,
      R"(Map from the 8*nJoints local skeleton state to the 7*nJoints jointParameters.  This performs the following operations:

* Removing the translation offset.
* Inverting out the pre-rotation.
* Converting to Euler angles.

The local skeleton state is stored (tx, ty, tz; rx, ry, rz, rw; s) and each maps the transform from the joint's local space to its parent joint space.
The joint parameters are stored (tx, ty, tz; ry, rz, ry; s) where rotations are in Euler angles and are relative to the parent joint.

:param character: Character to use.
:param local_skel_state: torch.Tensor containing the ([nBatch] x nJoints x 8) skeleton state.

:return: torch.Tensor of size ([nBatch] x nJoints x 7) containing the joint parameters.)",
      py::arg("character"),
      py::arg("local_skel_state"));

  // stripLowerBodyVertices(character)
  m.def(
      "strip_lower_body_vertices",
      &stripLowerBodyVertices,
      R"(Returns a character where all vertices below the waist have been stripped out (without modifying the skeleton).
This can be useful for visualization if you don't want the legs to distract.

:param character: Full-body character.
:return: A new character with only the upper body visible.)",
      py::arg("character"));

  m.def(
      "strip_joints",
      [](const momentum::Character& c,
         const std::vector<std::string>& joints_in) {
        std::vector<size_t> joints;
        for (const auto& j : joints_in) {
          const auto idx = c.skeleton.getJointIdByName(j);
          if (idx == momentum::kInvalidIndex) {
            throw std::runtime_error(
                "Trying to remove nonexistent joint '" + j +
                "' from skeleton.");
          }
          joints.push_back(idx);
        }

        return momentum::removeJoints(c, joints);
      },
      R"(Returns a character where the passed-in joints and all joints parented underneath them have been removed.

:param character: Full-body character.
:param joint_names: Names of the joints to remove.
:return: A new character with only the upper body visible.)",
      py::arg("character"),
      py::arg("joint_names"));

  // replace_skeleton_recursive(character, activeParameters)
  m.def(
      "replace_skeleton_hierarchy",
      momentum::replaceSkeletonHierarchy,
      R"(Replaces the part of target_character's skeleton rooted at target_root with the part of
source_character's skeleton rooted at source_root.
This is used e.g. to swap one character's hand skeleton with another.

:param source_character: Source character.
:param target_character: Target character.
:param source_root: Root of the source skeleton hierarchy to be copied.
:param target_root: Root of the target skeleton hierarchy to be replaced.
:return: A new skeleton that is identical to tgt_skeleton except that everything under target_root
   has been replaced by the part of source_character rooted at source_root.
    )",
      py::arg("source_character"),
      py::arg("target_character"),
      py::arg("source_root"),
      py::arg("target_root"));

  // reduceToSelectedModelParameters(character, activeParameters)
  m.def(
      "reduce_to_selected_model_parameters",
      &reduceToSelectedModelParameters,
      R"(Strips out unused parameters from the parameter transform.

:param character: Full-body character.
:param activeParameters: A boolean tensor marking which parameters should be retained.
:return: A new character whose parameter transform only includes the marked parameters.)",
      py::arg("character"),
      py::arg("active_parameters"));

  m.def(
      "find_closest_points",
      &findClosestPoints,
      R"(For each point in the points_source tensor, find the closest point in the points_target tensor.  This version of find_closest points supports both 2- and 3-dimensional point sets.

:param points_source: [nBatch x nPoints x dim] tensor of source points (dim must be 2 or 3).
:param points_target: [nBatch x nPoints x dim] tensor of target points (dim must be 2 or 3).
:param max_dist: Maximum distance to search, can be used to speed up the method by allowing the search to return early.  Defaults to FLT_MAX.
:return: A tuple of three tensors.  The first is [nBatch x nPoints x dim] and contains the closest point for each point in the target set.
         The second is [nBatch x nPoints] and contains the index of each closest point in the target set (or -1 if none).
         The third is [nBatch x nPoints] and is a boolean tensor indicating whether a valid closest point was found for each source point.
      )",
      py::arg("points_source"),
      py::arg("points_target"),
      py::arg("max_dist") = std::numeric_limits<float>::max());

  m.def(
      "find_closest_points",
      &findClosestPointsWithNormals,
      R"(For each point in the points_source tensor, find the closest point in the points_target tensor whose normal is compatible (n_source . n_target > max_normal_dot).
Using the normal is a good way to avoid certain kinds of bad matches, such as matching the front of the body against depth values from the back of the body.

:param points_source: [nBatch x nPoints x 3] tensor of source points.
:param normals_source: [nBatch x nPoints x 3] tensor of source normals (must be normalized).
:param points_target: [nBatch x nPoints x 3] tensor of target points.
:param normals_target: [nBatch x nPoints x 3] tensor of target normals (must be normalized).
:param max_dist: Maximum distance to search, can be used to speed up the method by allowing the search to return early.  Defaults to FLT_MAX.
:param max_normal_dot: Maximum dot product allowed between the source and target normal.  Defaults to 0.
:return: A tuple of three tensors.  The first is [nBatch x nPoints x dim] and contains the closest point for each point in the target set.
         The second is [nBatch x nPoints] and contains the index of each closest point in the target set (or -1 if none).
         The third is [nBatch x nPoints] and is a boolean tensor indicating whether a valid closest point was found for each source point.
      )",
      py::arg("points_source"),
      py::arg("normals_source"),
      py::arg("points_target"),
      py::arg("normals_target"),
      py::arg("max_dist") = std::numeric_limits<float>::max(),
      py::arg("max_normal_dot") = 0.0f);

  m.def(
      "replace_rest_mesh",
      &replaceRestMesh,
      R"(Return a new :class:`Character` with the rest mesh positions replaced by the passed-in positions.
        Can be used to e.g. bake the blend shapes into the character's mesh.  Does not allow changing the topology.

:param rest_vertex_positions: nVert x 3 numpy array of vertex positions.
        )",
      py::arg("character"),
      py::arg("rest_vertex_positions"));

  m.def(
      "compute_vertex_normals",
      &computeVertexNormals,
      R"(
Computes vertex normals for a triangle mesh given its positions.

:param vertex_positions: [nBatch] x nVert x 3 Tensor of vertex positions.
:param triangles: nTriangles x 3 Tensor of triangle indices.
:return: Smooth per-vertex normals.
    )",
      py::arg("vertex_positions"),
      py::arg("triangles"));
}
