#include "core/allocator.h"
#include <utility>

namespace infini {
Allocator::Allocator(Runtime runtime) : runtime(runtime) {
  used = 0;
  peak = 0;
  ptr = nullptr;

  // 'alignment' defaults to sizeof(uint64_t), because it is the length of
  // the longest data type currently supported by the DataType field of
  // the tensor
  alignment = sizeof(uint64_t);
}

Allocator::~Allocator() {
  if (this->ptr != nullptr) {
    runtime->dealloc(this->ptr);
  }
}

size_t Allocator::alloc(size_t size) {
  size = this->getAlignedSize(size);

  // 初始化
  if (ptr == nullptr) {
    ptr = runtime->alloc(size);
    no_used[0] = size;
    peak = size;
  }

  // 相邻碎片合并
  for (auto it = no_used.begin(); no_used.size() && next(it) != no_used.end();
       it++) {
    size_t Offset = it->first;
    size_t &no_used_size = it->second;
    if (Offset + no_used_size - 1 == next(it)->first) {
      no_used_size += next(it)->second;
      no_used.erase(next(it));
    }
  }

  // 分配内存
  for (auto [Offset, no_used_size] : no_used) {
    if (size <= no_used_size) {
      used += size;
      no_used.erase(Offset);
      no_used[Offset + size] = no_used_size - size;
      return Offset;
    } else if (size == no_used_size) {
      no_used.erase(Offset);
      used += size;
      return Offset;
    }
  }

  // 扩展内存
  size_t result = no_used.size() ? no_used.end()->first : peak;

  do {
    peak *= 2;
  } while (peak < size);

  void *new_ptr = runtime->alloc(peak);
  memcpy(new_ptr, ptr, peak);
  runtime->dealloc(ptr);
  ptr = new_ptr;
  used += size;

  return result;
}

void Allocator::free(size_t addr, size_t size) {
  // IT_ASSERT(this->ptr == nullptr);
  size = getAlignedSize(size);
  used -= size;
  no_used[addr] = size;
}

void *Allocator::getPtr() {
  if (this->ptr == nullptr) {
    this->ptr = runtime->alloc(this->peak);
    printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
  }
  return this->ptr;
}

size_t Allocator::getAlignedSize(size_t size) {
  return ((size - 1) / this->alignment + 1) * this->alignment;
}

void Allocator::info() {
  std::cout << "Used memory: " << this->used << ", peak memory: " << this->peak
            << std::endl;
}
} // namespace infini
