#pragma once
#include <cstddef>
#include <cstdint>
extern "C" {
struct Dataset;
struct Output;
struct Sample { uint64_t index, started, received, finished; };
Dataset* px_load(const char*,const char*,const char*,size_t);
void px_free(Dataset*);
uint64_t px_now();
Output* px_fuse(const Dataset*,size_t);
int px_validate_free(const Dataset*,size_t,Output*);
int px_native(const Dataset*,size_t,Sample*,size_t);
int px_write(const Dataset*,const Sample*,size_t,const char*);
}

static_assert(sizeof(Sample)==32, "Rust/C++ sample ABI mismatch");
