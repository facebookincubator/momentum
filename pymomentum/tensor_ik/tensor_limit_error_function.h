// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <pymomentum/tensor_ik/tensor_error_function.h>

namespace pymomentum {

template <typename T>
std::unique_ptr<TensorErrorFunction<T>> createLimitErrorFunction(
    size_t batchSize,
    size_t nFrames);

}
