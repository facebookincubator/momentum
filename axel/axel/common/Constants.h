/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <type_traits>

namespace axel::detail {

template <typename T>
[[nodiscard]] constexpr T eps(T floatEps = 1e-6, T doubleEps = 1e-15) {
  if constexpr (std::is_same_v<T, float>) {
    return floatEps;
  } else if constexpr (std::is_same_v<T, double>) {
    return doubleEps;
  } else {
    // Using !sizeof(T) makes the static_assert condition type-dependent, ensuring
    // it's evaluated only when the template is instantiated with a specific type.
    static_assert(!sizeof(T), "Unsupported type");
  }
}

} // namespace axel::detail
