/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/common/filesystem.h"
#include "momentum/io/marker/marker_io.h"
#include "momentum/test/io/io_helpers.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace momentum;

namespace {

std::string getMarkerFile() {
  auto envVar = GetEnvVar("TEST_RESOURCES_PATH");
  const auto markerFilePath = filesystem::path(envVar.value()) / "markers.c3d";
  return markerFilePath.string();
}

TEST(MarkerIOTest, testLoadMarkers) {
  const std::string markerFile = getMarkerFile();
  const std::vector<MarkerSequence> actorSequences = loadMarkers(markerFile);
  // the file has one actor sequence with total 89 frames and 36 markers per frame
  EXPECT_EQ(actorSequences.size(), 1);
  EXPECT_EQ(actorSequences[0].frames.size(), 89);
  EXPECT_EQ(actorSequences[0].frames[0].size(), 36);
}

TEST(MarkerIOTest, testFindMainSubject) {
  // 4 visible markers
  std::vector<Marker> markersFrame0 = {
      {"RFT1", {0.0, 0.0, 0.0}, false},
      {"RFT2", {1.0, 0.0, 0.0}, false},
      {"RFT3", {1.0, 1.0, 0.0}, false},
      {"RFT4", {1.0, 1.0, 0.0}, false}};

  // 3 visible markers
  std::vector<Marker> markersFrame1 = {
      {"RFT1", {0.0, 0.0, 0.5}, false},
      {"RFT2", {1.0, 0.0, 0.5}, true},
      {"RFT3", {1.0, 1.0, 0.5}, false},
      {"RFT4", {1.0, 1.0, 0.5}, false}};

  // 1 visible markers
  std::vector<Marker> markersFrame2 = {
      {"RFT1", {0.0, 0.0, 0.5}, false},
      {"RFT2", {1.0, 0.0, 0.5}, true},
      {"RFT3", {1.0, 1.0, 0.5}, true},
      {"RFT4", {1.0, 1.0, 0.5}, true}};

  // actor-0 has more visible markers on the first frame, but actor-1 has more visible markers on
  // the second frame so it's the main subject.
  std::vector<MarkerSequence> actorSequences = {
      {
          "actor-0",
          {markersFrame1},
      },
      {
          "actor-1",
          {markersFrame2, markersFrame0},
      }};

  const int mainSubjectID = findMainSubjectIndex(actorSequences);
  EXPECT_EQ(mainSubjectID, 1);
}

TEST(MarkerIOTest, testLoadMarkersForMainSubject) {
  const std::string markerFile = getMarkerFile();
  std::optional<MarkerSequence> mainSubjectSequence = loadMarkersForMainSubject(markerFile);
  // the file has one actor sequence with total 89 frames and 36 markers per frame
  EXPECT_TRUE(mainSubjectSequence.has_value());
  EXPECT_EQ(mainSubjectSequence->frames.size(), 89);
  EXPECT_EQ(mainSubjectSequence->frames[0].size(), 36);
}

TEST(MarkerIOTest, testLoadMarkersEmpty) {
  const auto actorSequences = loadMarkers("");
  EXPECT_EQ(actorSequences.size(), 0);
}

} // namespace
