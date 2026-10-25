#include "strata/core/vmm.hpp"

#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_GPU_METAL)
#include <cuda.h>
#include <cuda_runtime.h>

#include <mutex>
#include <map>

namespace strata::core {
namespace {

// The entry points are the same for every device and are resolved once; granularity and VMM support are properties
// of an allocation location, so they are queried once per device (a second prefill stage runs on another device).
struct Api {
    bool ok = false;
    decltype(&cuDeviceGetAttribute) attr = nullptr;
    decltype(&cuMemGetAllocationGranularity) granularity = nullptr;
    decltype(&cuMemAddressReserve) reserve = nullptr;
    decltype(&cuMemAddressFree) free_va = nullptr;
    decltype(&cuMemCreate) create = nullptr;
    decltype(&cuMemRelease) release = nullptr;
    decltype(&cuMemMap) map = nullptr;
    decltype(&cuMemUnmap) unmap = nullptr;
    decltype(&cuMemSetAccess) access = nullptr;
};

template <class F> bool resolve(const char* name, F& f) {
    cudaDriverEntryPointQueryResult q{};
    void* p = nullptr;
#if CUDART_VERSION >= 12050
    const cudaError_t e = cudaGetDriverEntryPointByVersion(name, &p, 12000, cudaEnableDefault, &q);
#else   // the versioned query arrived in CUDA 12.5
    const cudaError_t e = cudaGetDriverEntryPoint(name, &p, cudaEnableDefault, &q);
#endif
    if (e != cudaSuccess || q != cudaDriverEntryPointSuccess || p == nullptr) {
        (void) cudaGetLastError();
        return false;
    }
    f = (F) p;
    return true;
}

// never destroyed: the K/V pools and the cache release their chunks from static destructors at exit
const Api& api() {
    static Api& a = *new Api;
    static std::once_flag once;
    std::call_once(once, [] {
        a.ok = resolve("cuDeviceGetAttribute", a.attr) &&
               resolve("cuMemGetAllocationGranularity", a.granularity) &&
               resolve("cuMemAddressReserve", a.reserve) && resolve("cuMemAddressFree", a.free_va) &&
               resolve("cuMemCreate", a.create) && resolve("cuMemRelease", a.release) &&
               resolve("cuMemMap", a.map) && resolve("cuMemUnmap", a.unmap) &&
               resolve("cuMemSetAccess", a.access);
    });
    return a;
}

struct DeviceCap {
    bool ok = false;
    uint64_t gran = 0;
};

// The current device's VMM support and granularity.  Cached per device: on a multi-GPU host the second stage's
// device is a different one, with its own granularity, and capturing either once (from GPU 0) would be wrong.
DeviceCap device_cap(int dev) {
    static std::mutex mu;
    static std::map<int, DeviceCap> cache;
    std::lock_guard<std::mutex> lk(mu);
    const auto it = cache.find(dev);
    if (it != cache.end()) return it->second;
    DeviceCap c;
    const Api& a = api();
    if (a.ok && dev >= 0) {
        int vmm = 0;
        if (a.attr(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, (CUdevice) dev) == CUDA_SUCCESS && vmm) {
            CUmemAllocationProp prop{};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = dev;
            size_t g = 0;
            if (a.granularity(&g, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) == CUDA_SUCCESS && g != 0) {
                c.gran = g;
                c.ok = true;
            }
        }
    }
    cache.emplace(dev, c);
    return c;
}

int current_device() {
    int d = -1;
    return cudaGetDevice(&d) == cudaSuccess ? d : -1;
}

}  // namespace

bool vmm_available() { return device_cap(current_device()).ok; }
uint64_t vmm_granularity() { return device_cap(current_device()).gran; }

VmmChunk vmm_chunk_new(int device) {
    const Api& a = api();
    const DeviceCap c = device_cap(device);
    if (!a.ok || !c.ok) return 0;
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    CUmemGenericAllocationHandle h = 0;
    if (a.create(&h, (size_t) c.gran, &prop, 0) != CUDA_SUCCESS) return 0;
    return (VmmChunk) h;
}

VmmChunk vmm_chunk_new() { return vmm_chunk_new(current_device()); }

void vmm_chunk_free(VmmChunk h) {
    if (h != 0 && api().ok) api().release((CUmemGenericAllocationHandle) h);
}

bool VmmRange::reserve(uint64_t bytes) {
    release();
    const Api& a = api();
    const int dev = current_device();
    const DeviceCap c = device_cap(dev);
    if (!a.ok || !c.ok || bytes == 0) return false;
    const uint64_t n = (bytes + c.gran - 1) / c.gran;
    CUdeviceptr p = 0;
    if (a.reserve(&p, (size_t) (n * c.gran), 0, 0, 0) != CUDA_SUCCESS) return false;
    base_ = (unsigned long long) p;
    dev_ = dev;
    gran_ = c.gran;
    h_.assign((size_t) n, 0);
    return true;
}

void VmmRange::release() {
    if (base_ == 0) return;
    const Api& a = api();
    for (int64_t i = 0; i < chunks(); ++i) vmm_chunk_free(unmap(i));
    a.free_va((CUdeviceptr) base_, (size_t) ((uint64_t) h_.size() * gran_));
    base_ = 0;
    dev_ = -1;
    gran_ = 0;
    h_.clear();
}

int64_t VmmRange::mapped_count() const {
    int64_t n = 0;
    for (const VmmChunk h : h_) n += h != 0;
    return n;
}

bool VmmRange::map_one(int64_t i, VmmChunk h) {
    const Api& a = api();
    if (a.map((CUdeviceptr) (base_ + (uint64_t) i * gran_), (size_t) gran_, 0, (CUmemGenericAllocationHandle) h, 0) !=
        CUDA_SUCCESS)
        return false;
    h_[(size_t) i] = h;
    return true;
}

bool VmmRange::set_access(int64_t lo, int64_t hi) {
    if (hi <= lo) return true;
    const Api& a = api();
    CUmemAccessDesc d{};
    d.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    d.location.id = dev_;
    d.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    return a.access((CUdeviceptr) (base_ + (uint64_t) lo * gran_), (size_t) ((uint64_t) (hi - lo) * gran_), &d, 1) ==
           CUDA_SUCCESS;
}

bool VmmRange::commit_run(int64_t lo, int64_t hi) {
    if (set_access(lo, hi)) return true;
    for (int64_t c = lo; c < hi; ++c) vmm_chunk_free(unmap(c));
    return false;
}

VmmChunk VmmRange::unmap(int64_t i) {
    if (!mapped(i)) return 0;
    const Api& a = api();
    const VmmChunk h = h_[(size_t) i];
    if (a.unmap((CUdeviceptr) (base_ + (uint64_t) i * gran_), (size_t) gran_) != CUDA_SUCCESS) return 0;
    h_[(size_t) i] = 0;
    return h;
}

}  // namespace strata::core

#else  // not CUDA
// HIP (ROCm 6+): the same elastic range through hipMemAddressReserve / hipMemCreate / hipMemMap / hipMemSetAccess.  Available
// only where the device reports virtual memory management support (STRATA_HIP_VMM=0 turns it off): --kv-grow stays off
// otherwise, as before.  Everything else (Metal, gfx906, ROCm older than 6): no virtual memory, the elastic K/V stays off.
#if defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)
#include <hip/hip_runtime.h>
#include <hip/hip_version.h>

#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#endif

#if defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && defined(HIP_VERSION_MAJOR) && HIP_VERSION_MAJOR >= 6
namespace strata::core {
namespace {
struct DeviceCap {
    bool ok = false;
    uint64_t gran = 0;
};
void fail(const char* what, hipError_t e) {
    static int n = 0;
    if (n++ < 8) std::fprintf(stderr, "strata: HIP virtual memory: %s failed: %s\n", what, hipGetErrorString(e));
    (void) hipGetLastError();
}
hipMemAllocationProp alloc_prop(int dev) {
    hipMemAllocationProp p{};
    p.type = hipMemAllocationTypePinned;
    p.requestedHandleType = hipMemHandleTypeNone;
    p.location.type = hipMemLocationTypeDevice;
    p.location.id = dev;
    return p;
}
/// STRATA_HIP_VMM=0 turns the whole feature off (read once, like the CUDA branch's resolved entry points).
bool vmm_disabled() {
    static const bool off = [] {
        const char* v = std::getenv("STRATA_HIP_VMM");
        return v != nullptr && v[0] == '0';
    }();
    return off;
}
int current_device() {
    int d = -1;
    if (hipGetDevice(&d) != hipSuccess) { (void) hipGetLastError(); return -1; }
    return d;
}
// The device's VMM support and granularity, cached per device like the CUDA branch's: the second prefill stage is a
// different device, and capturing either once (from GPU 0) would be wrong.  A device with no VMM - or an invalid
// index, such as the -1 of a range that was never reserved - answers 0 from vmm_chunk_new(device), the header's
// "no VMM there", instead of creating a chunk with another device's property.
DeviceCap device_cap(int dev) {
    static std::mutex mu;
    static std::map<int, DeviceCap> cache;
    std::lock_guard<std::mutex> lk(mu);
    const auto it = cache.find(dev);
    if (it != cache.end()) return it->second;
    DeviceCap c;
    if (!vmm_disabled() && dev >= 0) {
        int vmm = 0;
        if (hipDeviceGetAttribute(&vmm, hipDeviceAttributeVirtualMemoryManagementSupported, dev) == hipSuccess && vmm) {
            const hipMemAllocationProp prop = alloc_prop(dev);
            size_t g = 0;
            if (hipMemGetAllocationGranularity(&g, &prop, hipMemAllocationGranularityMinimum) == hipSuccess && g != 0) {
                // gfx1201 reports a 4 KiB minimum granularity; the elastic range works in 2 MiB chunks like CUDA's
                // (any multiple of the minimum)
                const uint64_t want = 2ull << 20;
                c.gran = g >= want ? g : (want + g - 1) / g * g;
                c.ok = true;
            }
        }
        if (!c.ok) (void) hipGetLastError();
    }
    cache.emplace(dev, c);
    return c;
}
}  // namespace

bool vmm_available() { return device_cap(current_device()).ok; }
uint64_t vmm_granularity() { return device_cap(current_device()).gran; }

VmmChunk vmm_chunk_new(int device) {
    const DeviceCap c = device_cap(device);
    if (!c.ok) return 0;
    const hipMemAllocationProp prop = alloc_prop(device);
    hipMemGenericAllocationHandle_t h = nullptr;
    if (const hipError_t e = hipMemCreate(&h, (size_t) c.gran, &prop, 0); e != hipSuccess) { fail("hipMemCreate", e); return 0; }
    return (VmmChunk) (uintptr_t) h;
}

VmmChunk vmm_chunk_new() { return vmm_chunk_new(current_device()); }

void vmm_chunk_free(VmmChunk h) {
    if (h != 0) (void) hipMemRelease((hipMemGenericAllocationHandle_t) (uintptr_t) h);
}

bool VmmRange::reserve(uint64_t bytes) {
    release();
    const int dev = current_device();
    const DeviceCap c = device_cap(dev);
    if (!c.ok || bytes == 0) return false;
    const uint64_t n = (bytes + c.gran - 1) / c.gran;
    void* p = nullptr;
    if (const hipError_t e = hipMemAddressReserve(&p, (size_t) (n * c.gran), 0, nullptr, 0); e != hipSuccess) { fail("hipMemAddressReserve", e); return false; }
    base_ = (unsigned long long) (uintptr_t) p;
    dev_ = dev;          // every later map/access/unmap/release uses THIS device, not the ambient one
    gran_ = c.gran;
    h_.assign((size_t) n, 0);
    return true;
}

void VmmRange::release() {
    if (base_ == 0) return;
    for (int64_t i = 0; i < chunks(); ++i) vmm_chunk_free(unmap(i));
    (void) hipMemAddressFree((void*) (uintptr_t) base_, (size_t) ((uint64_t) h_.size() * gran_));
    base_ = 0;
    dev_ = -1;
    gran_ = 0;
    h_.clear();
}

int64_t VmmRange::mapped_count() const {
    int64_t n = 0;
    for (const VmmChunk h : h_) n += h != 0;
    return n;
}

bool VmmRange::map_one(int64_t i, VmmChunk h) {
    if (const hipError_t e = hipMemMap((void*) (uintptr_t) (base_ + (uint64_t) i * gran_), (size_t) gran_, 0,
                                      (hipMemGenericAllocationHandle_t) (uintptr_t) h, 0);
        e != hipSuccess) {
        fail("hipMemMap", e);
        return false;
    }
    h_[(size_t) i] = h;
    return true;
}

bool VmmRange::set_access(int64_t lo, int64_t hi) {
    if (hi <= lo) return true;
    hipMemAccessDesc d{};
    d.location.type = hipMemLocationTypeDevice;
    d.location.id = dev_;
    d.flags = hipMemAccessFlagsProtReadWrite;
    const hipError_t e = hipMemSetAccess((void*) (uintptr_t) (base_ + (uint64_t) lo * gran_), (size_t) ((uint64_t) (hi - lo) * gran_),
                                         &d, 1);
    const bool ok = e == hipSuccess;
    if (!ok) fail("hipMemSetAccess", e);
    return ok;
}

bool VmmRange::commit_run(int64_t lo, int64_t hi) {
    if (set_access(lo, hi)) return true;
    for (int64_t c = lo; c < hi; ++c) vmm_chunk_free(unmap(c));
    return false;
}

VmmChunk VmmRange::unmap(int64_t i) {
    if (!mapped(i)) return 0;
    const VmmChunk h = h_[(size_t) i];
    if (hipMemUnmap((void*) (uintptr_t) (base_ + (uint64_t) i * gran_), (size_t) gran_) != hipSuccess) { (void) hipGetLastError(); return 0; }
    h_[(size_t) i] = 0;
    return h;
}
}  // namespace strata::core
#else


namespace strata::core {
bool vmm_available() { return false; }
uint64_t vmm_granularity() { return 0; }
VmmChunk vmm_chunk_new() { return 0; }
VmmChunk vmm_chunk_new(int) { return 0; }
void vmm_chunk_free(VmmChunk) {}
bool VmmRange::reserve(uint64_t) { return false; }
void VmmRange::release() {}
int64_t VmmRange::mapped_count() const { return 0; }
bool VmmRange::map_one(int64_t, VmmChunk) { return false; }
bool VmmRange::set_access(int64_t, int64_t) { return false; }
bool VmmRange::commit_run(int64_t, int64_t) { return false; }
VmmChunk VmmRange::unmap(int64_t) { return 0; }
}  // namespace strata::core

#endif
#endif
