/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#if defined(MOMENTUM_WITH_PORTABILITY)

#include <portability/Filesystem.h>

#else

#include <filesystem>

namespace filesystem = std::filesystem;

#endif
