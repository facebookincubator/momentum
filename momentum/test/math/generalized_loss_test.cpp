/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include "momentum/math/constants.h"
#include "momentum/math/fwd.h"
#include "momentum/math/generalized_loss.h"
#include "momentum/math/random.h"

using namespace momentum;

namespace {

template <typename T>
[[nodiscard]] bool AreAlmostEqualRelative(T a, T b, T relTol) {
  return std::abs(a - b) <= relTol * std::max(std::abs(a), std::abs(b));
}

} // namespace

using Types = testing::Types<float, double>;

template <typename T>
struct GeneralizedLossTest : testing::Test {
  using Type = T;
};

TYPED_TEST_SUITE(GeneralizedLossTest, Types);

template <typename T>
void testGeneralizedLoss(T alpha, T c, T absTol, T relTol) {
  Random rand;
  const GeneralizedLossT<T> loss(alpha, c);

  const T stepSize = Eps<T>(5e-3f, 5e-5);
  // make sure the value is greater than stepSize for finite difference test
  const T sqrError = rand.uniform<T>(stepSize, 1e3);
  const T refDeriv = loss.deriv(sqrError);

  // finite difference test
  const T val1 = loss.value(sqrError - stepSize);
  const T val2 = loss.value(sqrError + stepSize);
  T testDeriv = (val2 - val1) / (T(2) * stepSize);

  // use an esp as input because larger alpha needs larger threshold
  bool result = (std::abs(testDeriv - refDeriv) <= absTol);
  if (!result) {
    result = AreAlmostEqualRelative(testDeriv, refDeriv, relTol);
  }
  EXPECT_TRUE(result)
      // clang-format off
      << "Failure in testGeneralizedLoss. Local variables are:"
      << "\n - alpha    : " << alpha
      << "\n - c        : " << c
      << "\n - absTol   : " << absTol
      << "\n - relTol   : " << relTol
      << "\n - stepSize : " << stepSize
      << "\n - sqrError : " << sqrError
      << "\n - val1     : " << val1
      << "\n - val2     : " << val2
      << "\n - refDeriv : " << refDeriv
      << "\n - testDeriv: " << testDeriv << std::endl;
      // clang-format off
}

TYPED_TEST(GeneralizedLossTest, SpecialCaseTest) {
  using T = typename TestFixture::Type;

  Random rand;
  const size_t nTrials = 20;
  const T absTol = Eps<T>(5e-1f, 5e-5);
  const T relTol = 0.01;  // 1% relative tolerance
  for (size_t i = 0; i < nTrials; ++i) {
    testGeneralizedLoss<T>(GeneralizedLossd::kL2, rand.uniform<T>(1, 10), absTol, relTol);
    testGeneralizedLoss<T>(GeneralizedLossd::kL1, rand.uniform<T>(1, 10), absTol, relTol);
    testGeneralizedLoss<T>(GeneralizedLossd::kCauchy, rand.uniform<T>(1, 10), absTol, relTol);
    testGeneralizedLoss<T>(GeneralizedLossd::kWelsch, rand.uniform<T>(1, 10), absTol, relTol);
  }
}

TYPED_TEST(GeneralizedLossTest, GeneralCaseTest) {
  using T = typename TestFixture::Type;

  Random rand;
  const size_t nTrials = 100;
  const T absTol = Eps<T>(1e-3f, 2e-6);
  const T relTol = 0.02;  // 2% relative tolerance
  testGeneralizedLoss<T>(10, 10, absTol, relTol);  // most extreme case
  for (size_t i = 0; i < nTrials; ++i) {
    testGeneralizedLoss<T>(rand.uniform<T>(-1e6, 10), rand.uniform<T>(0, 10), absTol, relTol);
  }
}
