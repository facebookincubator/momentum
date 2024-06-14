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
  virtual EState GetState() override;
  virtual bool Open(void* pStreamData) override;
  virtual bool Close() override;
  virtual bool Flush() override;
  virtual int Write(const void*, int) override;
  virtual int Read(void*, int) const override;
  virtual int GetReaderID() const override;
  virtual int GetWriterID() const override;
  virtual void Seek(const FbxInt64& pOffset, const FbxFile::ESeekPos& pSeekPos) override;
  virtual long GetPosition() const override;
  virtual void SetPosition(long pPosition) override;
  virtual int GetError() const override;
  virtual void ClearError() override;

 private:
  gsl::span<const std::byte> buffer_;
  long length_;
  mutable long position_;
  EState state_;
  int readerId_;
  mutable int errorCode_;
};

} // namespace momentum
