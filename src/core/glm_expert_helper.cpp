
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>

#include "strata/core/glm_expert_helper.hpp"


// Small enough to copy as one pinned buffer per layer. The kernels read the
// individual arrays through pointers into the same device allocation.
struct RemoteMeta {
    unsigned long long ptr[CAP];
    int32_t start[CAP + 1];
    int32_t dst[CAP];
    int32_t tok[CAP];
    int32_t count;
};

struct DeviceScope {
    int previous = -1;
    bool ok = false;
    cudaError_t status = cudaSuccess;
    const char* failed_step = nullptr;
    explicit DeviceScope(int device) {
        status = cudaGetDevice(&previous);
        if (status != cudaSuccess) {
            previous = -1;
            failed_step = "cudaGetDevice";
            return;
        }
        status = cudaSetDevice(device);
        if (status != cudaSuccess) {
            failed_step = "cudaSetDevice";
            return;
        }
        ok = true;
    }
    ~DeviceScope() { if (previous >= 0) cudaSetDevice(previous); }
    std::string error(int device) const {
        return std::string("CUDA") + std::to_string(device) + " experts: " +
               (failed_step ? failed_step : "device switch") +
               "(" + std::to_string(device) + ") failed: " + cudaGetErrorString(status) +
               " (CUDA error " + std::to_string((int) status) + ")";
    }
};

bool check(cudaError_t result, const char* what, std::string& err, int device) {
    if (result == cudaSuccess) return true;
    err = "CUDA" + std::to_string(device) + " experts: " + what + ": " + cudaGetErrorString(result);
    return false;
}
} // namespace

// (nothing may leave a destructor: caught here, so GCC 14+ emits no __cxa_call_terminate for it - a build whose

namespace strata::core {
GlmExpertHelper::~GlmExpertHelper() {
    try {
        close();
    } catch (...) {
    }
}

bool GlmExpertHelper::open(int device, int n_embd, int n_ff,
                           const std::vector<std::pair<void*, size_t>>& regions, std::string& err) {
    close();
    DeviceScope scope(device);
    if (!scope.ok) { err = scope.error(device); return false; }
    int integrated = 0;
    if (!check(cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, device), "integrated memory", err, device))
        return false;
    if (n_embd <= 0 || n_embd > 4096 || n_embd % 32 || n_ff <= 0 || n_ff % 32) {
        err = "GLM expert helper: unsupported expert dimensions";
        return false;
    }
    device_ = device;
    integrated_ = integrated != 0;
    n_embd_ = n_embd;
    n_ff_ = n_ff;
    if (integrated_) for (const auto& [host, bytes] : regions) {
        void* mapped = nullptr;
        if (!check(cudaHostGetDevicePointer(&mapped, host, 0), "map RAM tier", err, device)) return false;
        regions_.push_back({(const uint8_t*) host, (uint8_t*) mapped, bytes});
    }
    const size_t need = 2 * (size_t) n_embd * sizeof(float) + (size_t) (n_embd + 8 * n_ff) / 32 * 36 + 1024;
    if (!check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "helper stream", err, device) ||
        !check(cudaMalloc((void**) &arena_, need), "helper scratch", err, device)) return false;
    uint8_t* at = arena_;
    const auto take = [&](size_t bytes) {
        void* p = at;
        at += (bytes + 63u) & ~(size_t) 63u;
        return p;
    };
    x_ = (float*) take((size_t) n_embd * sizeof(float));
    out_ = (float*) take((size_t) n_embd * sizeof(float));
    xq_ = take((size_t) n_embd / 32 * 36);
    hq_ = take((size_t) 8 * n_ff / 32 * 36);
    plan_.plan_ptr = (unsigned long long*) take(8 * sizeof(unsigned long long));
    plan_.plan_w = (float*) take(8 * sizeof(float));
    plan_.cpu_flag = (int*) take(64);
    return check(cudaMemset(plan_.cpu_flag, 0, 64), "helper flags", err, device);
}

bool GlmExpertHelper::reserve(int keys, size_t requested_bytes, std::string& err) {
    DeviceScope scope(device_);
    if (!scope.ok) { err = scope.error(device_); return false; }
    slots_.assign((size_t) keys, {});
    owned_.assign((size_t) keys, 0);
    capacity_ = requested_bytes;
    if (!integrated_) {
        size_t free = 0, total = 0;
        if (!check(cudaMemGetInfo(&free, &total), "helper capacity", err, device_)) return false;
        constexpr size_t headroom = 512ull << 20;   // PR #578's automatic VRAM budget
        capacity_ = std::min(capacity_, free > headroom ? free - headroom : 0);
        capacity_ &= ~(size_t) 255;
        if (capacity_ && !check(cudaMalloc((void**) &pool_, capacity_), "helper cache", err, device_)) return false;
    }
    return true;
}

bool GlmExpertHelper::insert(int key, const uint8_t* host, size_t bytes, size_t stride, std::string& err) {
    if (key < 0 || (size_t) key >= slots_.size() || owns(key) || bytes > stride || stride > available()) {
        err = "GLM expert helper: invalid cache insertion"; return false;
    }
    Slot s{const_cast<uint8_t*>(host), nullptr, bytes};
    if (integrated_) {
        for (const auto& r : regions_) {
            const uintptr_t offset = (uintptr_t) host - (uintptr_t) r.host;
            if ((uintptr_t) host >= (uintptr_t) r.host && offset <= r.bytes && bytes <= r.bytes - offset) {
                s.device = r.mapped + offset; break;
            }
        }
        if (!s.device) { err = "GLM expert helper: expert outside the mapped RAM tier"; return false; }
    } else {
        DeviceScope scope(device_);
        if (!scope.ok) { err = scope.error(device_); return false; }
        s.device = pool_ + used_;
        if (!check(cudaMemcpyAsync(s.device, host, bytes, cudaMemcpyHostToDevice, stream_), "cache upload", err, device_) ||
            !check(cudaStreamSynchronize(stream_), "cache upload finish", err, device_)) return false;
    }
    slots_[(size_t) key] = s;
    owned_[(size_t) key] = 1;
    used_ += stride;
    return true;
}

bool GlmExpertHelper::replace(int out, int in, const uint8_t* host, std::string& err) {
    if (out < 0 || (size_t) out >= slots_.size() || in < 0 || (size_t) in >= slots_.size() || !owns(out) || owns(in)) {
        err = "GLM expert helper: invalid cache replacement"; return false;
    }
    Slot s = slots_[(size_t) out];
    if (integrated_) {
        uint8_t* mapped = nullptr;
        for (const auto& r : regions_) {
            const uintptr_t offset = (uintptr_t) host - (uintptr_t) r.host;
            if ((uintptr_t) host >= (uintptr_t) r.host && offset <= r.bytes && s.bytes <= r.bytes - offset) {
                mapped = r.mapped + offset; break;
            }
        }
        // Already in shared RAM: exchange ownership, not bytes. The old expert
        // stays in the host tier, including in --glm-ram-resident mode.
        if (mapped) s = {const_cast<uint8_t*>(host), mapped, s.bytes};
        else std::memcpy(s.host, host, s.bytes);   // a disk-only candidate uses the existing slot
    }
    else {
        DeviceScope scope(device_);
        if (!scope.ok) { err = scope.error(device_); return false; }
        if (!check(cudaMemcpyAsync(s.device, host, s.bytes, cudaMemcpyHostToDevice, stream_), "cache replace", err, device_) ||
            !check(cudaStreamSynchronize(stream_), "cache replace finish", err, device_)) return false;
    }
    slots_[(size_t) in] = s;
    slots_[(size_t) out] = {};
    owned_[(size_t) out] = 0;
    owned_[(size_t) in] = 1;
    return true;
}

// PR #578: complement the primary cache, replacing a cold helper expert only
// with a hotter unowned expert of the SAME layer. Pending primary admissions
// are included in `primary` by the caller before this idle-boundary operation.
std::vector<GlmExpertHelper::Swap> GlmExpertHelper::swaps(const std::vector<uint32_t>& usage,
        const std::vector<int>& primary, int ne, int max_swaps) const {
    std::vector<Swap> result;
    for (size_t base = 0; base < owned_.size(); base += ne) {
        std::vector<std::pair<uint32_t, int>> candidates, victims;
        uint32_t primary_cold = UINT32_MAX;
        for (int e = 0; e < ne; ++e) {
            const int key = (int) base + e;
            if (primary[(size_t) key] >= 0) primary_cold = std::min(primary_cold, usage[(size_t) key]);
            if (owns(key)) victims.emplace_back(usage[(size_t) key], key);
            else if (primary[(size_t) key] < 0 && usage[(size_t) key] >= 2) candidates.emplace_back(usage[(size_t) key], key);
        }
        std::sort(candidates.begin(), candidates.end(), [](auto a, auto b) { return a.first > b.first; });
        std::sort(victims.begin(), victims.end());
        for (size_t i = 0; i < std::min(candidates.size(), victims.size()); ++i) {
            if (victims[i].first > primary_cold || candidates[i].first < victims[i].first + 2u) break;
            result.push_back({candidates[i].first - victims[i].first, candidates[i].second, victims[i].second});
        }
    }
    std::sort(result.begin(), result.end(), [](auto a, auto b) { return a.gain > b.gain; });
    if ((int) result.size() > max_swaps) result.resize((size_t) max_swaps);
    return result;
}

bool GlmExpertHelper::compute(int gu_type, int d_type, size_t down_off, float limit, int ne,
                              const int* keys, const float* weights, const float* x,
                              float* out, std::string& err) {
    namespace gf = strata::kernels::glmf;
    if (device_ < 0 || ne < 1 || ne > 8 || !gf::moe_supported(gu_type) || !gf::moe_supported(d_type)) {
        err = "GLM expert helper: unsupported expert types or route width";
        return false;
    }
    unsigned long long ptr[8] = {};
    for (int i = 0; i < ne; ++i) {
        if (keys[i] < 0 || (size_t) keys[i] >= slots_.size() || !owns(keys[i])) {
            err = "GLM expert helper: expert not owned by helper"; return false;
        }
        ptr[i] = (unsigned long long) slots_[(size_t) keys[i]].device;
    }
    DeviceScope scope(device_);
    if (!scope.ok) { err = scope.error(device_); return false; }
    if (!check(cudaMemcpyAsync(x_, x, (size_t) n_embd_ * sizeof(float), cudaMemcpyHostToDevice, stream_),
               "helper input", err, device_) ||
        !check(cudaMemcpyAsync(plan_.plan_ptr, ptr, (size_t) ne * sizeof(ptr[0]), cudaMemcpyHostToDevice, stream_),
               "helper pointers", err, device_) ||
        !check(cudaMemcpyAsync(plan_.plan_w, weights, (size_t) ne * sizeof(float), cudaMemcpyHostToDevice, stream_),
               "helper weights", err, device_)) return false;
    gf::quantize_q8_1(x_, xq_, n_embd_, stream_);
    gf::moe_gate_up(gu_type, plan_, ne, n_embd_, n_ff_, limit, xq_, hq_, nullptr, 0, nullptr, 0, nullptr, stream_);
    gf::moe_down(d_type, plan_, ne, n_embd_, n_ff_, down_off, hq_, nullptr, out_, stream_);
    return check(cudaMemcpyAsync(out, out_, (size_t) n_embd_ * sizeof(float), cudaMemcpyDeviceToHost, stream_),
                 "helper result", err, device_) &&
           check(cudaStreamSynchronize(stream_), "helper finish", err, device_);
}

void GlmExpertHelper::close() {
    if (device_ < 0) return;
    DeviceScope scope(device_);
    if (scope.ok) {
        if (stream_) cudaStreamSynchronize(stream_);
        if (arena_) cudaFree(arena_);
        if (pool_) cudaFree(pool_);
        if (stream_) cudaStreamDestroy(stream_);
    }
    device_ = -1;
    arena_ = nullptr;
    pool_ = nullptr;
    capacity_ = used_ = 0;
    stream_ = nullptr;
    regions_.clear();
    slots_.clear();
    owned_.clear();
}  // namespace strata::core
