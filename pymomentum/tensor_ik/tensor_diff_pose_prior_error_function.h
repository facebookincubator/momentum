/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <pymomentum/tensor_ik/tensor_error_function.h>

namespace pymomentum {

template <typename T>
std::unique_ptr<TensorErrorFunction<T>> createDiffPosePriorErrorFunction(
    size_t batchSize,
    size_t nFrames,
    at::Tensor pi,
    at::Tensor mu,
    at::Tensor W,
    at::Tensor sigma,
    at::Tensor parameterIndices);

}
