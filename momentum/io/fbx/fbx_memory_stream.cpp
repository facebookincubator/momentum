/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/io/fbx/fbx_memory_stream.h"

namespace momentum {

FbxMemoryStream::FbxMemoryStream(gsl::span<const std::byte> buffer, int pReaderId)
    : buffer_(buffer),
      length_(buffer.size()),
      position_(0),
      state_(EState::eClosed),
      readerId_(pReaderId) {}

FbxMemoryStream::~FbxMemoryStream() {
  Close();
}

FbxStream::EState FbxMemoryStream::GetState() {
  return state_;
}

bool FbxMemoryStream::Open(void* /* pStreamData */) {
  position_ = 0;
  state_ = EState::eOpen;
  return true;
}

bool FbxMemoryStream::Close() {
  position_ = 0;
  state_ = EState::eClosed;
  return true;
}

bool FbxMemoryStream::Flush() {
  return true;
}

int FbxMemoryStream::Write(const void*, int) {
  errorCode_ = 1;
  return 0;
}

int FbxMemoryStream::Read(void* buffer, int count) const {
  long remaining = length_ - position_;
  if (count > remaining) {
    errorCode_ = 1;
    return 0;
  }

  memcpy(buffer, buffer_.data() + position_, count);
  position_ += count;
  return count;
}

int FbxMemoryStream::GetReaderID() const {
  return readerId_;
}

int FbxMemoryStream::GetWriterID() const {
  return -1;
}

void FbxMemoryStream::Seek(const FbxInt64& pOffset, const FbxFile::ESeekPos& pSeekPos) {
  long offset = (long)pOffset;
  switch (pSeekPos) {
    case FbxFile::ESeekPos::eCurrent:
      position_ += offset;
      break;
    case FbxFile::ESeekPos::eEnd:
      position_ = length_ - offset;
      break;
    default:
      position_ = offset;
      break;
  }
}

long FbxMemoryStream::GetPosition() const {
  return position_;
}

void FbxMemoryStream::SetPosition(long pos) {
  position_ = pos;
}

int FbxMemoryStream::GetError() const {
  return errorCode_;
}

void FbxMemoryStream::ClearError() {
  errorCode_ = 0;
}

} // namespace momentum
