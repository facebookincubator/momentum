/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <fmt/ostream.h>
#include <Eigen/Core>

#if FMT_VERSION > 100000

// https://stackoverflow.com/a/73755864
#ifndef MOMENTUM_FMT_EIGEN_SHARED_PTR_FORMATTER
#define MOMENTUM_FMT_EIGEN_SHARED_PTR_FORMATTER
template <typename T>
struct fmt::formatter<T, std::enable_if_t<std::is_base_of_v<Eigen::DenseBase<T>, T>, char>>
    : ostream_formatter {};
#endif

#endif
