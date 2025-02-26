// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <pymomentum/tensor_ik/tensor_error_function.h>

#include <momentum/math/fwd.h>

namespace pymomentum {

template <typename T>
std::unique_ptr<TensorErrorFunction<T>> createPosePriorErrorFunction(
    size_t batchSize,
    size_t nFrames,
    const momentum::Mppca* posePrior_model);

}
