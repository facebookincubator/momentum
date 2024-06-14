/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/math/constants.h"
#include "momentum/math/utility.h"

#include <gtest/gtest.h>

using namespace momentum;

using Types = testing::Types<float, double>;

template <typename T>
struct UtilityTest : testing::Test {
  using Type = T;
};

TYPED_TEST_SUITE(UtilityTest, Types);

TYPED_TEST(UtilityTest, IsNanNoOpt) {
  using T = typename TestFixture::Type;

  const T nanValue = std::numeric_limits<T>::quiet_NaN();
  const T infValue = std::numeric_limits<T>::infinity();
  const T normalValue = static_cast<T>(42.0);

#ifndef MOMENTUM_TEST_FAST_MATH
  EXPECT_TRUE(std::isnan(nanValue));
#endif
  EXPECT_FALSE(std::isnan(infValue));
  EXPECT_FALSE(std::isnan(normalValue));

  EXPECT_TRUE(IsNanNoOpt(nanValue));
  EXPECT_FALSE(IsNanNoOpt(infValue));
  EXPECT_FALSE(IsNanNoOpt(normalValue));
}

TYPED_TEST(UtilityTest, EulerAngles) {
  using T = typename TestFixture::Type;

  std::vector<Vector3<T>> angles_set = {
      Vector3<T>(0.0, 0.0, 0.0),
      Vector3<T>(0.0, 0.0, pi<T>() / 2.0),
      Vector3<T>(0.0, 0.0, -pi<T>() / 2.0),
      Vector3<T>(0.0, pi<T>() / 2.0, 0.0),
      Vector3<T>(0.0, -pi<T>() / 2.0, 0.0),
      Vector3<T>(pi<T>() / 2.0, 0.0, 0.0),
      Vector3<T>(-pi<T>() / 2.0, 0.0, 0.0),
      Vector3<T>(pi<T>() / 2.0, pi<T>() / 2.0, 0.0),
      Vector3<T>(pi<T>() / 2.0, -pi<T>() / 2.0, 0.0),
  };

  const auto numTests = 10;
  for (auto i = 0u; i < numTests; ++i) {
    angles_set.push_back(Vector3<T>::Random());
  }

  // Intrinsic XYZ
  for (const auto& angles : angles_set) {
    const auto convention = EulerConvention::INTRINSIC;
    const Matrix3<T> mat1 = Quaternion<T>(
                                AngleAxis<T>(angles(0), Vector3<T>::UnitX()) *
                                AngleAxis<T>(angles(1), Vector3<T>::UnitY()) *
                                AngleAxis<T>(angles(2), Vector3<T>::UnitZ()))
                                .toRotationMatrix();
    const Matrix3<T> mat2 = eulerToRotationMatrix(angles, 0, 1, 2, convention);
    const Matrix3<T> mat3 = eulerToRotationMatrix(
        rotationMatrixToEuler(mat2, 0, 1, 2, convention), 0, 1, 2, convention);
    const Matrix3<T> mat4 = eulerXYZToRotationMatrix(angles, convention);
    const Matrix3<T> mat5 =
        eulerXYZToRotationMatrix(rotationMatrixToEulerXYZ(mat4, convention), convention);
    EXPECT_TRUE(mat1.isApprox(mat2));
    EXPECT_TRUE(mat2.isApprox(mat3));
    EXPECT_TRUE(mat3.isApprox(mat4));
    EXPECT_TRUE(mat4.isApprox(mat5));
  }

  // Intrinsic ZYX
  for (const auto& angles : angles_set) {
    const auto convention = EulerConvention::INTRINSIC;
    const Matrix3<T> mat1 = Quaternion<T>(
                                AngleAxis<T>(angles(0), Vector3<T>::UnitZ()) *
                                AngleAxis<T>(angles(1), Vector3<T>::UnitY()) *
                                AngleAxis<T>(angles(2), Vector3<T>::UnitX()))
                                .toRotationMatrix();
    const Matrix3<T> mat2 = eulerToRotationMatrix(angles, 2, 1, 0, convention);
    const Matrix3<T> mat3 = eulerToRotationMatrix(
        rotationMatrixToEuler(mat2, 2, 1, 0, convention), 2, 1, 0, convention);
    const Matrix3<T> mat4 = eulerZYXToRotationMatrix(angles, convention);
    const Matrix3<T> mat5 =
        eulerZYXToRotationMatrix(rotationMatrixToEulerZYX(mat4, convention), convention);
    EXPECT_TRUE(mat1.isApprox(mat2));
    EXPECT_TRUE(mat2.isApprox(mat3));
    EXPECT_TRUE(mat3.isApprox(mat4));
    EXPECT_TRUE(mat4.isApprox(mat5));
  }

  // Extrinsic XYZ
  for (const auto& angles : angles_set) {
    const auto convention = EulerConvention::EXTRINSIC;
    const Matrix3<T> mat1 = Quaternion<T>(
                                AngleAxis<T>(angles(2), Vector3<T>::UnitZ()) *
                                AngleAxis<T>(angles(1), Vector3<T>::UnitY()) *
                                AngleAxis<T>(angles(0), Vector3<T>::UnitX()))
                                .toRotationMatrix();
    const Matrix3<T> mat2 = eulerToRotationMatrix(angles, 0, 1, 2, convention);
    const Matrix3<T> mat3 = eulerToRotationMatrix(
        rotationMatrixToEuler(mat2, 0, 1, 2, convention), 0, 1, 2, convention);
    const Matrix3<T> mat4 = eulerXYZToRotationMatrix(angles, convention);
    const Matrix3<T> mat5 =
        eulerXYZToRotationMatrix(rotationMatrixToEulerXYZ(mat4, convention), convention);
    EXPECT_TRUE(mat1.isApprox(mat2));
    EXPECT_TRUE(mat2.isApprox(mat3));
    EXPECT_TRUE(mat3.isApprox(mat4));
    EXPECT_TRUE(mat4.isApprox(mat5));
  }

  // Extrinsic ZYX
  for (const auto& angles : angles_set) {
    const auto convention = EulerConvention::EXTRINSIC;
    const Matrix3<T> mat1 = Quaternion<T>(
                                AngleAxis<T>(angles(2), Vector3<T>::UnitX()) *
                                AngleAxis<T>(angles(1), Vector3<T>::UnitY()) *
                                AngleAxis<T>(angles(0), Vector3<T>::UnitZ()))
                                .toRotationMatrix();
    const Matrix3<T> mat2 = eulerToRotationMatrix(angles, 2, 1, 0, convention);
    const Matrix3<T> mat3 = eulerToRotationMatrix(
        rotationMatrixToEuler(mat2, 2, 1, 0, convention), 2, 1, 0, convention);
    const Matrix3<T> mat4 = eulerZYXToRotationMatrix(angles, convention);
    const Matrix3<T> mat5 =
        eulerZYXToRotationMatrix(rotationMatrixToEulerZYX(mat4, convention), convention);
    EXPECT_TRUE(mat1.isApprox(mat2));
    EXPECT_TRUE(mat2.isApprox(mat3));
    EXPECT_TRUE(mat3.isApprox(mat4));
    EXPECT_TRUE(mat4.isApprox(mat5));
  }
}

// Test function to check the conversion between intrinsic and extrinsic Euler angles
template <typename T>
void testIntrinsicExtrinsicConversion(int axis0, int axis1, int axis2) {
  for (auto i = 0u; i < 10; ++i) {
    const Vector3<T> euler_angles = Vector3<T>::Random();
    Matrix3<T> m;
    m = AngleAxis<T>(euler_angles[0], Vector3<T>::Unit(axis0)) *
        AngleAxis<T>(euler_angles[1], Vector3<T>::Unit(axis1)) *
        AngleAxis<T>(euler_angles[2], Vector3<T>::Unit(axis2));

    // Compute the intrinsic and extrinsic Euler angles
    const Vector3<T> euler_intrinsic =
        rotationMatrixToEuler(m, axis0, axis1, axis2, EulerConvention::INTRINSIC);
    const Vector3<T> euler_extrinsic =
        rotationMatrixToEuler(m, axis2, axis1, axis0, EulerConvention::EXTRINSIC);

    // Check if the extrinsic Euler angles are the reverse of the intrinsic Euler angles
    EXPECT_TRUE(euler_intrinsic.reverse().isApprox(euler_extrinsic))
        << "euler_intrinsic: " << euler_intrinsic.transpose() << "\n"
        << "euler_extrinsic: " << euler_extrinsic.transpose() << "\n";
  }
}

TYPED_TEST(UtilityTest, TestRoundTripIntrinsicExtrinsic) {
  using T = typename TestFixture::Type;

  // Test all possible angle orders (assuming the axes are distinct)
  testIntrinsicExtrinsicConversion<T>(0, 1, 2);
  testIntrinsicExtrinsicConversion<T>(0, 2, 1);
  testIntrinsicExtrinsicConversion<T>(1, 0, 2);
  testIntrinsicExtrinsicConversion<T>(1, 2, 0);
  testIntrinsicExtrinsicConversion<T>(2, 0, 1);
  testIntrinsicExtrinsicConversion<T>(2, 1, 0);
}

// Test function to check the conversion between intrinsic and extrinsic Euler angles
// and their corresponding quaternions
template <typename T>
void testEulerToQuaternionConversion(int axis0, int axis1, int axis2) {
  for (auto i = 0u; i < 10; ++i) {
    const Vector3<T> euler_angles = Vector3<T>::Random();

    const Quaternion<T> quaternion_intrinsic =
        eulerToQuaternion(euler_angles, axis0, axis1, axis2, EulerConvention::INTRINSIC);

    const Quaternion<T> quaternion_extrinsic = eulerToQuaternion(
        euler_angles.reverse().eval(), axis2, axis1, axis0, EulerConvention::EXTRINSIC);

    // Compute the rotation matrices from the quaternions
    const Matrix3<T> rotation_matrix_intrinsic = quaternion_intrinsic.toRotationMatrix();
    const Matrix3<T> rotation_matrix_extrinsic = quaternion_extrinsic.toRotationMatrix();

    // Check if the rotation matrices are approximately equal
    EXPECT_TRUE(rotation_matrix_intrinsic.isApprox(rotation_matrix_extrinsic))
        << "rotation_matrix_intrinsic:\n"
        << rotation_matrix_intrinsic << "\n"
        << "rotation_matrix_extrinsic:\n"
        << rotation_matrix_extrinsic << "\n";
  }
}

TYPED_TEST(UtilityTest, TestEulerToQuaternionIntrinsicExtrinsic) {
  using T = typename TestFixture::Type;

  // Test all possible angle orders (assuming the axes are distinct)
  testEulerToQuaternionConversion<T>(0, 1, 2);
  testEulerToQuaternionConversion<T>(0, 2, 1);
  testEulerToQuaternionConversion<T>(1, 0, 2);
  testEulerToQuaternionConversion<T>(1, 2, 0);
  testEulerToQuaternionConversion<T>(2, 0, 1);
  testEulerToQuaternionConversion<T>(2, 1, 0);
}

TYPED_TEST(UtilityTest, TestQuaternionToRotVec) {
  using T = typename TestFixture::Type;

  const auto numTests = 10;
  for (auto i = 0u; i < numTests; ++i) {
    const Vector3<T> rand_angles = Vector3<T>::Random();
    const Quaternion<T> quaternion = AngleAxis<T>(rand_angles(0), Vector3<T>::UnitX()) *
        AngleAxis<T>(rand_angles(1), Vector3<T>::UnitY()) *
        AngleAxis<T>(rand_angles(2), Vector3<T>::UnitZ());

    const Vector3<T> rot_vec = quaternionToRotVec<T>(quaternion);
    const Quaternion<T> quaternion_from_rot_vec = rotVecToQuaternion<T>(rot_vec);

    EXPECT_TRUE(quaternion.isApprox(quaternion_from_rot_vec))
        << "quaternion: " << quaternion.coeffs().transpose() << "\n"
        << "quaternion from rotation vector: " << quaternion_from_rot_vec.coeffs().transpose()
        << "\n";
  }
}

TYPED_TEST(UtilityTest, TestRotVecToQuaternion) {
  using T = typename TestFixture::Type;

  const auto numTests = 10;
  for (auto i = 0u; i < numTests; ++i) {
    const Vector3<T> rand_rot_vec = Vector3<T>::Random();

    const Quaternion<T> quaternion = rotVecToQuaternion<T>(rand_rot_vec);
    const Vector3<T> rot_vec_from_quaternion = quaternionToRotVec<T>(quaternion);

    EXPECT_TRUE(rand_rot_vec.isApprox(rot_vec_from_quaternion))
        << "rotation vector: " << rand_rot_vec.transpose() << "\n"
        << "rotation vector from quaternion: " << rot_vec_from_quaternion.transpose() << "\n";
  }
}
