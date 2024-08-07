/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "pymomentum/python_utility/python_utility.h"

#include <momentum/math/mesh.h>

#include <pybind11/pybind11.h> // @manual

#include <cstdint>
#include <vector>

namespace pymomentum {

namespace py = pybind11;

std::vector<const momentum::Character*> toCharacterList(
    PyObject* obj,
    int64_t nBatch,
    const char* context,
    bool forceBatchSize) {
  assert(obj != nullptr);
  if (PyList_Check(obj)) {
    std::vector<const momentum::Character*> result;
    for (Py_ssize_t i = 0; i < PyList_Size(obj); ++i) {
      result.push_back(
          py::cast<const momentum::Character*>(PyList_GetItem(obj, i)));
    }

    if (result.empty()) {
      throw std::runtime_error(
          "Expected a valid pymomentum.Character of list of Characters in " +
          std::string(context));
    }

    if (result.size() != nBatch) {
      throw std::runtime_error(
          "Expected a valid pymomentum.Character of list of Characters in " +
          std::string(context));
    }

    for (const auto& c : result) {
      if (c->parameterTransform.name !=
          result.front()->parameterTransform.name) {
        throw std::runtime_error(
            std::string("Mismatch in parameter transforms in ") + context +
            "; when processing a batch of characters, all characters must have the same parameters in the same order.");
      }

      if (c->parameterTransform.numJointParameters() !=
          result.front()->parameterTransform.numJointParameters()) {
        throw std::runtime_error(
            std::string("Mismatch in parameter transforms in ") + context +
            "; when processing a batch of characters, all characters must have the same number of skeleton bones.");
      }

      if (result.front()->mesh) {
        if (!c->mesh ||
            c->mesh->vertices.size() != result.front()->mesh->vertices.size()) {
          throw std::runtime_error(
              std::string("Mismatch in meshes in ") + context +
              "; when processing a batch of characters, all characters must have the same number of vertices.");
        }
      }

      if (result.front()->skinWeights) {
        if (!c->skinWeights) {
          throw std::runtime_error(
              std::string("Mismatch in meshes in ") + context +
              "; when processing a batch of characters, all characters must have the same number of vertices.");
        }
      }
    }

    return result;
  } else {
    return std::vector<const momentum::Character*>(
        forceBatchSize ? nBatch : 1, py::cast<const momentum::Character*>(obj));
  }
}

// Extracts any single character that can be used to check the parameter counts,
// etc.
const momentum::Character& anyCharacter(PyObject* obj, const char* context) {
  assert(obj != nullptr);
  if (PyList_Check(obj)) {
    for (Py_ssize_t i = 0; i < PyList_Size(obj); ++i) {
      try {
        return *py::cast<const momentum::Character*>(PyList_GetItem(obj, i));
      } catch (std::exception&) {
        throw std::runtime_error(
            "Expected a valid pymomentum.Character of list of Characters in " +
            std::string(context));
      }
    }

    throw std::runtime_error(
        "Expected a valid pymomentum.Character of list of Characters in " +
        std::string(context) + "; got an empty list.");
  }

  return *py::cast<const momentum::Character*>(obj);
}

nlohmann::json from_msgpack(const pybind11::bytes& bytes) {
  py::buffer_info info(py::buffer(bytes).request());
  const uint8_t* data = reinterpret_cast<const uint8_t*>(info.ptr);
  const size_t length = static_cast<size_t>(info.size);

  if (data == nullptr) {
    throw std::runtime_error("Unable to extract contents from bytes.");
  }

  return nlohmann::json::from_msgpack(data, data + length);
}

pybind11::bytes to_msgpack(const nlohmann::json& j) {
  // Use of Vector of char because pybind11::bytes would like to receive
  // const char* for initialization of bytes.
  std::string buffer;
  nlohmann::json::to_msgpack(j, buffer);
  pybind11::bytes pybytes(buffer);
  return pybytes;
}

PyBytesStreamBuffer::PyBytesStreamBuffer(const pybind11::bytes& bytes) {
  py::buffer_info info(py::buffer(bytes).request());

  // C++ does not distinguish between const and non-const streambufs, but we
  // promise to only use this with std::istreams.
  char* data = const_cast<char*>(reinterpret_cast<const char*>(info.ptr));
  const size_t length = static_cast<size_t>(info.size);

  if (data == nullptr) {
    throw std::runtime_error("Unable to extract contents from bytes.");
  }

  this->setg(data, data, data + length);
}

} // namespace pymomentum
