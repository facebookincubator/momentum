/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <fbxsdk.h>
#include <fbxsdk/core/fbxstream.h>
#include <gsl/span>

#include <string_view>

namespace momentum {

// Simplest FbxStream to read file from a string_view memory buffer
class FbxMemoryStream : public FbxStream {
 public:
  FbxMemoryStream(gsl::span<const std::byte> buffer, int pReaderId);
  ~FbxMemoryStream() override;
  EState GetState() override;
  bool Open(void* pStreamData) override;
  bool Close() override;
  bool Flush() override;
  int Write(const void* buffer, int count) override;
  int Read(void* buffer, int count) const override;
  int GetReaderID() const override;
  int GetWriterID() const override;
  void Seek(const FbxInt64& pOffset, const FbxFile::ESeekPos& pSeekPos) override;
  long GetPosition() const override;
  void SetPosition(long pPosition) override;
  int GetError() const override;
  void ClearError() override;

 private:
  gsl::span<const std::byte> buffer_;
  long length_;
  mutable long position_;
  EState state_;
  int readerId_;
  mutable int errorCode_;
};

} // namespace momentum
