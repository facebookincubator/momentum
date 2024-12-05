// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <marker_tracking/analysis_utils/marker_error_utils.hpp> // @manual=fbsource//arvr/projects/kuiper_belt/projects/marker_tracking:analysis_utils
#include <marker_tracking/metrics/self_collision_detector.h> // @manual=fbsource//arvr/projects/kuiper_belt/projects/marker_tracking:self_collision_detector
#include <marker_tracking/validate_c3d.h> // @manual=fbsource//arvr/projects/kuiper_belt/projects/marker_tracking:validate_c3d

#include <momentum/character/marker.h>
#include <momentum/marker_tracking/marker_tracker.h>
#include <momentum/marker_tracking/process_markers.h>
#include <momentum/marker_tracking/tracker_utils.h>
#include <momentum/math/mesh.h>

#include <pybind11/eigen.h> // @manual=fbsource//third-party/pybind11/fbcode_py_versioned:pybind11
#include <pybind11/pybind11.h> // @manual=fbsource//third-party/pybind11/fbcode_py_versioned:pybind11
#include <pybind11/stl.h> // @manual=fbsource//third-party/pybind11/fbcode_py_versioned:pybind11
#include <string>

namespace py = pybind11;

// Python bindings for marker tracking APIs defined under:
// //arvr/projects/kuiper_belt/projects/marker_tracking

// @dep=fbsource//arvr/libraries/dispenso:dispenso

PYBIND11_MODULE(marker_tracking, m) {
  m.doc() = "Module for exposing the C++ APIs of the marker tracking pipeline ";
  m.attr("__name__") = "pymomentum.marker_tracking";

  pybind11::module_::import(
      "pymomentum.geometry"); // @dep=fbcode//pymomentum:geometry

  // Bindings for APIs and types defined in
  // marker_tracking/analysis_utils/marker_error_utils.h
  // (https://pybind11.readthedocs.io/en/latest/classes.html#creating-bindings-for-a-custom-type)
  auto markerErrorReport = py::class_<marker_tracking::MarkerErrorReport>(
      m,
      "MarkerErrorReport",
      "Represents a report generated from computed marker errors");

  py::class_<marker_tracking::MarkerErrorReport::LocatorEntry>(
      markerErrorReport,
      "LocatorEntry",
      "a struct within MarkerErrorReport, called LocatorEntry")
      .def(py::init<>())
      .def_readwrite(
          "locator_name",
          &marker_tracking::MarkerErrorReport::LocatorEntry::locatorName)
      .def_readwrite(
          "frames_error",
          &marker_tracking::MarkerErrorReport::LocatorEntry::framesError)
      .def_readwrite(
          "has_data",
          &marker_tracking::MarkerErrorReport::LocatorEntry::hasData)
      .def_readwrite(
          "average_error",
          &marker_tracking::MarkerErrorReport::LocatorEntry::averageError);

  py::class_<marker_tracking::MarkerErrorReport::OcclusionInfo>(
      markerErrorReport,
      "OcclusionInfo",
      "a struct within MarkerErrorReport, called OcclusionInfo")
      .def(py::init<>())
      .def_readwrite(
          "frame_id",
          &marker_tracking::MarkerErrorReport::OcclusionInfo::frameId)
      .def_readwrite(
          "locator_names",
          &marker_tracking::MarkerErrorReport::OcclusionInfo::locatorNames);

  markerErrorReport.def(py::init<>())
      .def_readwrite(
          "unused_locators",
          &marker_tracking::MarkerErrorReport::unusedLocators,
          "Locators that are not used")
      .def_readwrite(
          "unused_markers",
          &marker_tracking::MarkerErrorReport::unusedMarkers,
          "Markers that are not used")
      .def_readwrite(
          "entries",
          &marker_tracking::MarkerErrorReport::entries,
          "vector<struct called LocatorEntry>")
      .def_readwrite(
          "occluded",
          &marker_tracking::MarkerErrorReport::occluded,
          "vector<struct called OcclusionInfo>");

  // (see find_segment_collisions)
  m.def(
      "generate_marker_error_report_on_files",
      [](const std::string& markerFile, const std::string& motionGlbFile)
          -> marker_tracking::MarkerErrorReport {
        return marker_tracking::generateMarkerErrorReportOnFiles(
            markerFile, motionGlbFile);
      },
      py::arg("marker_file"),
      py::arg("motion_glb_file"));

  // Bindings for APIs and types defined in
  // marker_tracking/metrics/self_collision_detector.h
  auto bodySegmentCollisionResult = py::class_<
      marker_tracking::BodySegmentCollisionResult>(
      m,
      "BodySegmentCollisionResult",
      "Represents a body part segment collision result e.g. left hand collides with right hand.");

  bodySegmentCollisionResult.def(py::init<>())
      .def_readonly(
          "first_segment",
          &marker_tracking::BodySegmentCollisionResult::firstSegment,
          "Name of the first body segment")
      .def_readonly(
          "second_segment",
          &marker_tracking::BodySegmentCollisionResult::secondSegment,
          "Name of the second body segment")
      .def_readonly(
          "num_contacts",
          &marker_tracking::BodySegmentCollisionResult::numContacts,
          "Number of contacts between two body segments")
      .def_readonly(
          "max_penetration_depth",
          &marker_tracking::BodySegmentCollisionResult::maxPenetrationDepth,
          "Max penetration depth from detected contacts");
  ;

  auto frameCollisionResult = py::class_<marker_tracking::FrameCollisionResult>(
      m,
      "FrameCollisionResult",
      "Represents the body part segment collision pairs in one frame");

  frameCollisionResult.def(py::init<>())
      .def_readonly(
          "segment_collision_results",
          &marker_tracking::FrameCollisionResult::segmentCollisionResults,
          "List of all body part segment collisions at the current frame");

  auto motionSequenceCollisionResult =
      py::class_<marker_tracking::MotionSequenceCollisionResult>(
          m,
          "MotionSequenceCollisionResult",
          "Represents collision results for a motion sequence.");

  motionSequenceCollisionResult.def(py::init<>())
      .def_readonly(
          "sequence_name",
          &marker_tracking::MotionSequenceCollisionResult::sequenceName,
          "Name of the sequence file")
      .def_readonly(
          "body_segment_root_names",
          &marker_tracking::MotionSequenceCollisionResult::bodySegmentRootNames,
          "List of body part segment names used for collision detection")
      .def_readonly(
          "frame_collision_results",
          &marker_tracking::MotionSequenceCollisionResult::
              frameCollisionResults,
          "Collision results for all frames in the mocap sequence");

  py::list bodySegmentRootNameList;
  for (const std::string& s : marker_tracking::kBodySegmentRootNames) {
    bodySegmentRootNameList.append(s);
  }
  m.attr("body_segment_roots_default") = bodySegmentRootNameList;
  m.def(
      "create_body_segments_to_vertices_mapping",
      &marker_tracking::createBodySegmentsToVerticesMapping);
  m.def(
      "find_segment_collisions",
      [](const momentum::Character& character,
         Eigen::Ref<const Eigen::VectorXf> jointParams,
         const std::unordered_map<std::string, std::vector<size_t>>&
             bodySegmentToVertices,
         const size_t maxContacts) {
        return marker_tracking::findSegmentCollisions(
            character, jointParams, bodySegmentToVertices, maxContacts);
      },
      py::arg("character"),
      py::arg("joint_params"),
      py::arg("body_segment_to_vertices"),
      py::arg("max_contacts"));

  m.def(
      "find_collision",
      [](const momentum::Mesh& mesh1,
         const momentum::Mesh& mesh2,
         const size_t maxContacts) {
        return marker_tracking::findCollision(mesh1, mesh2, maxContacts);
      },
      py::arg("mesh_1"),
      py::arg("mesh_2"),
      py::arg("max_contacts"));

  m.def(
      "create_body_segment_mesh",
      [](const momentum::Mesh& originalMesh,
         const std::vector<size_t>& vertexIds,
         const std::vector<Eigen::Vector3f>& newVertices) {
        return marker_tracking::createBodySegmentMesh(
            originalMesh, vertexIds, newVertices);
      },
      py::arg("original_mesh"),
      py::arg("vertices_ids"),
      py::arg("pose_vertices"));

  m.def("detect_self_collisions", &marker_tracking::detectSelfCollisions);

  // Bindings for types defined in marker_tracking/marker_tracker.h
  auto baseConfig = py::class_<marker_tracking::BaseConfig>(
      m, "BaseConfig", "Represents base config class");

  baseConfig.def(py::init<>())
      .def(
          py::init<float, float, size_t, bool>(),
          py::arg("min_vis_percent") = 0.0,
          py::arg("loss_alpha") = 2.0,
          py::arg("max_iter") = 30,
          py::arg("debug") = false)
      .def_readwrite(
          "min_vis_percent",
          &marker_tracking::BaseConfig::minVisPercent,
          "Minimum percetange of visible markers to be used")
      .def_readwrite(
          "loss_alpha",
          &marker_tracking::BaseConfig::lossAlpha,
          "Parameter to control the loss function")
      .def_readwrite(
          "max_iter", &marker_tracking::BaseConfig::maxIter, "Max iterations")
      .def_readwrite(
          "debug",
          &marker_tracking::BaseConfig::debug,
          "Whether to output debugging info");

  auto calibrationConfig = py::
      class_<marker_tracking::CalibrationConfig, marker_tracking::BaseConfig>(
          m, "CalibrationConfig", "Config for the body scale calibration step");

  // Default values are set from the configured values in marker_tracker.h
  calibrationConfig.def(py::init<>())
      .def(
          py::init<float, float, size_t, bool, size_t, size_t, bool, bool>(),
          py::arg("min_vis_percent") = 0.0,
          py::arg("loss_alpha") = 2.0,
          py::arg("max_iter") = 30,
          py::arg("debug") = false,
          py::arg("calib_frames") = 100,
          py::arg("major_iter") = 3,
          py::arg("global_scale_only") = false,
          py::arg("locators_only") = false)
      .def_readwrite(
          "calib_frames",
          &marker_tracking::CalibrationConfig::calibFrames,
          "Number of frames used for model calibration")
      .def_readwrite(
          "major_iter",
          &marker_tracking::CalibrationConfig::majorIter,
          "Number of calibration loops to run")
      .def_readwrite(
          "global_scale_only",
          &marker_tracking::CalibrationConfig::globalScaleOnly,
          "Calibrate only the global scale and not all proportions")
      .def_readwrite(
          "locators_only",
          &marker_tracking::CalibrationConfig::locatorsOnly,
          "Calibrate only the locator offsets");

  auto trackingConfig =
      py::class_<marker_tracking::TrackingConfig, marker_tracking::BaseConfig>(
          m, "TrackingConfig", "Config for the tracking optimization step");

  trackingConfig.def(py::init<>())
      .def(
          py::init<float, float, size_t, bool, float, float>(),
          py::arg("min_vis_percent") = 0.0,
          py::arg("loss_alpha") = 2.0,
          py::arg("max_iter") = 30,
          py::arg("debug") = false,
          py::arg("smoothing") = 0.0,
          py::arg("collision_error_weight") = 0.0)
      .def_readwrite(
          "smoothing",
          &marker_tracking::TrackingConfig::smoothing,
          "Smoothing weight; 0 to disable")
      .def_readwrite(
          "collision_error_weight",
          &marker_tracking::TrackingConfig::collisionErrorWeight,
          "Collision error weight; 0 to disable");

  auto refineConfig = py::
      class_<marker_tracking::RefineConfig, marker_tracking::TrackingConfig>(
          m, "RefineConfig", "Config for refining a tracked motion.");

  refineConfig.def(py::init<>())
      .def(
          py::init<
              float,
              float,
              size_t,
              bool,
              float,
              float,
              float,
              bool,
              bool>(),
          py::arg("min_vis_percent") = 0.0,
          py::arg("loss_alpha") = 2.0,
          py::arg("max_iter") = 30,
          py::arg("debug") = false,
          py::arg("smoothing") = 0.0,
          py::arg("collision_error_weight") = 0.0,
          py::arg("regularizer") = 0.0,
          py::arg("calib_id") = false,
          py::arg("calib_locators") = false)
      .def_readwrite(
          "regularizer",
          &marker_tracking::RefineConfig::regularizer,
          "Regularize the time-invariant parameters to prevent large changes.")
      .def_readwrite(
          "calib_id",
          &marker_tracking::RefineConfig::calibId,
          "Calibrate identity parameters; default to False.")
      .def_readwrite(
          "calib_locators",
          &marker_tracking::RefineConfig::calibLocators,
          "Calibrate locator offsets; default to False.");

  auto modelOptions = py::class_<marker_tracking::ModelOptions>(
      m,
      "ModelOptions",
      "Model options to specify the template model, parameter transform and locator mappings");

  modelOptions.def(py::init<>())
      .def(py::init<
           const std::string&,
           const std::string&,
           const std::string&>())
      .def_readwrite(
          "model",
          &marker_tracking::ModelOptions::model,
          "Path to template model file with locators e.g. blueman.glb")
      .def_readwrite(
          "parameters",
          &marker_tracking::ModelOptions::parameters,
          "Path of parameter trasform model file e.g. blueman.model")
      .def_readwrite(
          "locators",
          &marker_tracking::ModelOptions::locators,
          "Path to locator mapping file e.g. blueman.locators");
  m.def(
      "process_marker_file",
      &marker_tracking::processMarkerFile,
      py::arg("input_marker_file"),
      py::arg("output_file"),
      py::arg("tracking_config"),
      py::arg("calibration_config"),
      py::arg("model_options"),
      py::arg("calibrate"),
      py::arg("first_frame") = 0,
      py::arg("max_frames") = 0);

  m.def(
      "process_markers",
      [](momentum::Character& character,
         const Eigen::VectorXf& identity,
         const std::vector<std::vector<momentum::Marker>>& markerData,
         const marker_tracking::TrackingConfig& trackingConfig,
         const marker_tracking::CalibrationConfig& calibrationConfig,
         bool calibrate = true,
         size_t firstFrame = 0,
         size_t maxFrames = 0) {
        momentum::ModelParameters params(identity);

        if (params.size() == 0) { // If no identity is passed in, use default
          params = momentum::ModelParameters::Zero(
              character.parameterTransform.name.size());
        }

        return marker_tracking::processMarkers(
            character,
            params,
            markerData,
            trackingConfig,
            calibrationConfig,
            calibrate,
            firstFrame,
            maxFrames);
      },
      R"(process markers given character and identity.

:parameter character: Character to be used for tracking
:parameter identity: Identity parameters, pass in empty array for default identity
:parameter marker_data: A list of marker data for each frame
:parameter tracking_config: Tracking config to be used for tracking
:parameter calibration_config: Calibration config to be used for calibration
:parameter calibrate: Whether to calibrate the model
:parameter first_frame: First frame to be processed
:parameter max_frames: Max number of frames to be processed
:return: Transform parameters for each frame)",
      py::arg("character"),
      py::arg("identity"),
      py::arg("marker_data"),
      py::arg("tracking_config"),
      py::arg("calibration_config"),
      py::arg("calibrate") = true,
      py::arg("first_frame") = 0,
      py::arg("max_frames") = 0);

  m.def(
      "save_motion",
      [](const std::string& outFile,
         const momentum::Character& character,
         const Eigen::VectorXf& identity,
         Eigen::MatrixXf& motion,
         const std::vector<std::vector<momentum::Marker>>& markerData,
         const float fps,
         const bool saveMarkerMesh = true) {
        momentum::ModelParameters params(identity);

        if (params.size() == 0) { // If no identity is passed in, use default
          params = momentum::ModelParameters::Zero(
              character.parameterTransform.name.size());
        }
        return marker_tracking::saveMotion(
            outFile,
            character,
            params,
            motion,
            markerData,
            fps,
            saveMarkerMesh);
      },
      py::arg("out_file"),
      py::arg("character"),
      py::arg("identity"),
      py::arg("motion"),
      py::arg("marker_data"),
      py::arg("fps"),
      py::arg("save_marker_mesh") = true);

  m.def(
      "refine_motion",
      [](momentum::Character& character,
         const Eigen::VectorXf& identity,
         const Eigen::MatrixXf& motion,
         const std::vector<std::vector<momentum::Marker>>& markerData,
         const marker_tracking::RefineConfig& refineConfig) {
        // python and cpp have the motion matrix transposed from each other.
        // Let's do that on the way in and out here so it's consistent for both
        // lanangues.
        Eigen::MatrixXf inputMotion(motion.transpose());

        // If input identity is not empty, it means the motion is stripped of
        // identity field (eg. read from a glb file), so we need to fill it in.
        // If the input identity is empty, we assume the identity fields already
        // exist in the motion matrix.
        if (identity.size() > 0) {
          marker_tracking::ParameterSet idParamSet =
              character.parameterTransform.getScalingParameters();
          marker_tracking::fillIdentity(idParamSet, identity, inputMotion);
        }
        Eigen::MatrixXf finalMotion = marker_tracking::refineMotion(
            markerData, inputMotion, refineConfig, character);
        auto finalMotionTransposed = Eigen::MatrixXf(finalMotion.transpose());
        return finalMotionTransposed;
      },
      py::arg("character"),
      py::arg("identity"),
      py::arg("motion"),
      py::arg("marker_data"),
      py::arg("refine_config"));

  auto markerActorStats = py::class_<marker_tracking::MarkerActorStats>(
      m,
      "markerActorStats",
      "Class to collect stats from a single actor of a marker file");

  markerActorStats.def(py::init<>())
      .def_readonly(
          "actor_name",
          &marker_tracking::MarkerActorStats::actorName,
          "Name of a given actor in the marker file")
      .def_readonly(
          "num_frames",
          &marker_tracking::MarkerActorStats::numFrames,
          "Number of frames characterizing the marker sequence of the given actor")
      .def_readonly(
          "fps",
          &marker_tracking::MarkerActorStats::fps,
          "FPS characterizing the marker sequence of the given actor")
      .def_readonly(
          "marker_names",
          &marker_tracking::MarkerActorStats::markerNames,
          "List of all the marker names")
      .def_readonly(
          "min_count_markers",
          &marker_tracking::MarkerActorStats::minCountMarkers,
          "Minimum amount of non-occluded markers across the motion sequence")
      .def_readonly(
          "max_count_markers",
          &marker_tracking::MarkerActorStats::maxCountMarkers,
          "Maximum amount of non-occluded markers across the motion sequence")
      .def_readonly(
          "mean_count_markers",
          &marker_tracking::MarkerActorStats::meanCountMarkers,
          "Average amount of non-occluded markers across the motion sequence");

  auto markerFileStats = py::class_<marker_tracking::MarkerFileStats>(
      m, "markerFileStats", "Class to extract stats from an input marker file");

  markerFileStats.def(py::init<>())
      .def_readonly(
          "num_actors",
          &marker_tracking::MarkerFileStats::numActors,
          "Name of actors in the input marker file")
      .def_readonly(
          "marker_actor_stats",
          &marker_tracking::MarkerFileStats::markerActorStats,
          "List of the stats related to all actors in the marker file");

  m.def("stats_from_marker_file", &marker_tracking::statsFromMarkerFile);
}
