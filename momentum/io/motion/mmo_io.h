/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/fwd.h>
#include <momentum/math/types.h>

namespace momentum {

void saveMmo(
    const std::string& filename,
    gsl::span<const VectorXf> poses,
    const VectorXf& scale,
    const Character& character);

void saveMmo(
    const std::string& filename,
    const MatrixXf& poses,
    const VectorXf& scale,
    const Character& character,
    const MatrixXf& additionalParameters = MatrixXf(),
    gsl::span<const std::string> additionalParameterNames = std::vector<std::string>());

void saveMmo(
    const std::string& filename,
    const MatrixXf& poses,
    const VectorXf& scale,
    gsl::span<const std::string> parameterNames,
    gsl::span<const std::string> jointNames);

std::tuple<MatrixXf, VectorXf, std::vector<std::string>, std::vector<std::string>> loadMmo(
    const std::string& filename);

std::tuple<MatrixXf, VectorXf> loadMmo(const std::string& filename, const Character& character);

std::tuple<MatrixXf, std::vector<std::string>> getAuxilaryDataFromMotion(
    const MatrixXf& poses,
    gsl::span<const std::string> parameterNames);

// map motion to character
std::tuple<MatrixXf, VectorXf> mapMotionToCharacter(
    const MatrixXf& poses,
    const VectorXf& offsets,
    gsl::span<const std::string> parameterNames,
    gsl::span<const std::string> jointNames,
    const Character& character);

} // namespace momentum
