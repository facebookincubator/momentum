/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "axel/TriBvhEmbree.h"

#include "axel/Checks.h"
#include "axel/math/PointTriangleProjection.h"

namespace axel {

namespace {

template <typename T>
Eigen::Vector3<T> interpolatePosition(
    const Eigen::Vector3i& triangleIndices,
    const Eigen::MatrixX3<T>& positions,
    const Eigen::Vector3<T>& bary) {
  return positions.row(triangleIndices[0]) * bary[0] + positions.row(triangleIndices[1]) * bary[1] +
      positions.row(triangleIndices[2]) * bary[2];
}

template <typename S>
struct ClosestPointQueryUserData {
  const Eigen::MatrixX3<S>& positions;
  const Eigen::MatrixX3i& triangles;
  ClosestSurfacePointResult<S> result;
};

template <typename S>
bool pointQueryFunc(RTCPointQueryFunctionArguments* args) {
  XR_DEV_CHECK_NOTNULL(args->userPtr);
  auto* userData = static_cast<ClosestPointQueryUserData<S>*>(args->userPtr);
  XR_DEV_CHECK_NOTNULL(userData);

  const uint32_t primIdx = args->primID;

  const Eigen::Vector3<S> q(args->query->x, args->query->y, args->query->z);

  const Eigen::Vector3i tri = userData->triangles.row(primIdx);
  const Eigen::Vector3<S> v0 = userData->positions.row(tri[0]);
  const Eigen::Vector3<S> v1 = userData->positions.row(tri[1]);
  const Eigen::Vector3<S> v2 = userData->positions.row(tri[2]);

  Eigen::Vector3<S> p;
  projectOnTriangle(q, v0, v1, v2, p);
  const float d = (q - p).norm();

  if (d < args->query->radius) {
    args->query->radius = d;
    userData->result.point = p; // NOLINT
    userData->result.triangleIdx = primIdx;
    return true; // Return true to indicate that the query radius changed.
  }

  return false;
}
} // namespace

template <typename S>
TriBvhEmbree<S>::TriBvhEmbree(
    const Eigen::MatrixX3<S>& positions,
    const Eigen::MatrixX3i& triangles)
    : device_(rtcNewDevice(nullptr)),
      scene_(rtcNewScene(device_)),
      positions_(positions),
      triangles_(triangles) {
  const RTCGeometry geometry = rtcNewGeometry(device_, RTC_GEOMETRY_TYPE_TRIANGLE);
  rtcSetGeometryPointQueryFunction(geometry, pointQueryFunc<S>);

  auto* vb = static_cast<float*>(rtcSetNewGeometryBuffer(
      geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 3 * sizeof(float), positions.rows()));
  for (int i = 0; i < positions.rows(); ++i) {
    vb[i * 3 + 0] = positions(i, 0);
    vb[i * 3 + 1] = positions(i, 1);
    vb[i * 3 + 2] = positions(i, 2);
  }

  auto* ib = static_cast<uint32_t*>(rtcSetNewGeometryBuffer(
      geometry,
      RTC_BUFFER_TYPE_INDEX,
      0,
      RTC_FORMAT_UINT3,
      3 * sizeof(uint32_t),
      triangles.rows()));
  for (int i = 0; i < triangles.rows(); ++i) {
    ib[i * 3 + 0] = triangles(i, 0);
    ib[i * 3 + 1] = triangles(i, 1);
    ib[i * 3 + 2] = triangles(i, 2);
  }

  rtcCommitGeometry(geometry);
  rtcAttachGeometry(scene_, geometry);
  rtcReleaseGeometry(geometry);

  rtcCommitScene(scene_);
}

template <typename S>
TriBvhEmbree<S>::~TriBvhEmbree() {
  if (scene_) {
    rtcReleaseScene(scene_);
  }
  if (device_) {
    rtcReleaseDevice(device_);
  }
}

template <typename S>
TriBvhEmbree<S>::TriBvhEmbree(TriBvhEmbree<S>&& other) noexcept
    : device_{std::exchange(other.device_, nullptr)},
      scene_{std::exchange(other.scene_, nullptr)},
      positions_{std::move(other.positions_)},
      triangles_{std::move(other.triangles_)} {}

template <typename S>
TriBvhEmbree<S>& TriBvhEmbree<S>::operator=(TriBvhEmbree&& other) noexcept {
  device_ = std::exchange(other.device_, nullptr);
  scene_ = std::exchange(other.scene_, nullptr);
  positions_ = std::move(other.positions_);
  triangles_ = std::move(other.triangles_);
  return *this;
}

template <typename S>
std::optional<IntersectionResult<S>> TriBvhEmbree<S>::closestHit(const Ray3<S>& ray) const {
  RTCRayHit rayHit;
  rayHit.ray.org_x = ray.origin.x();
  rayHit.ray.org_y = ray.origin.y();
  rayHit.ray.org_z = ray.origin.z();
  rayHit.ray.dir_x = ray.direction.x();
  rayHit.ray.dir_y = ray.direction.y();
  rayHit.ray.dir_z = ray.direction.z();
  rayHit.ray.tnear = ray.minT;
  rayHit.ray.tfar = ray.maxT;
  rayHit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

  rtcIntersect1(scene_, &rayHit);

  if (rayHit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
    return std::nullopt;
  }

  const Eigen::Vector3<S> bary(
      static_cast<S>(1.0) - rayHit.hit.u - rayHit.hit.v, rayHit.hit.u, rayHit.hit.v);
  return IntersectionResult<S>{
      static_cast<int32_t>(rayHit.hit.primID),
      rayHit.ray.tfar,
      interpolatePosition<S>(triangles_.row(rayHit.hit.primID), positions_, bary),
      bary};
}

template <typename S>
bool TriBvhEmbree<S>::anyHit(const Ray3<S>& ray) const {
  RTCRay rayData;
  rayData.org_x = ray.origin.x();
  rayData.org_y = ray.origin.y();
  rayData.org_z = ray.origin.z();
  rayData.dir_x = ray.direction.x();
  rayData.dir_y = ray.direction.y();
  rayData.dir_z = ray.direction.z();
  rayData.tnear = ray.minT;
  rayData.tfar = ray.maxT;
  rtcOccluded1(scene_, &rayData);

  // According to Embree docs, a hit will set .tfar to -inf.
  return rayData.tfar < 0.0f;
}

template <typename S>
ClosestSurfacePointResult<S> TriBvhEmbree<S>::closestSurfacePoint(
    const Eigen::Vector3<S>& query) const {
  RTCPointQuery queryData{};
  queryData.x = query.x();
  queryData.y = query.y();
  queryData.z = query.z();
  queryData.time = 0.0;
  queryData.radius = std::numeric_limits<float>::max();

  ClosestPointQueryUserData<S> userData{positions_, triangles_, {}};

  RTCPointQueryContext context{};
  rtcInitPointQueryContext(&context);
  rtcPointQuery(scene_, &queryData, &context, nullptr, &userData);

  return userData.result;
}

template <typename S>
size_t TriBvhEmbree<S>::getPrimitiveCount() const {
  return static_cast<size_t>(triangles_.rows());
}

template class TriBvhEmbree<float>;
template class TriBvhEmbree<double>;

} // namespace axel
