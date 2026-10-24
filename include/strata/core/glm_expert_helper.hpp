#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/kernels/glm_fast.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

/// GLM helper-owned experts. UMA aliases pinned RAM; discrete helpers cache in
/// their own VRAM. Neither backend changes the legacy Qwen RemoteExperts path.
class GlmExpertHelper {
public:
    GlmExpertHelper() = default;
    ~GlmExpertHelper();
    GlmExpertHelper(const GlmExpertHelper&) = delete;
    GlmExpertHelper& operator=(const GlmExpertHelper&) = delete;
    bool open(int device, int n_embd, int n_ff,
              const std::vector<std::pair<void*, size_t>>& regions, std::string& err);
    bool reserve(int keys, size_t requested_bytes, std::string& err);
    bool insert(int key, const uint8_t* host, size_t bytes, size_t stride, std::string& err);
    bool replace(int out, int in, const uint8_t* host, std::string& err);
    struct Swap { uint32_t gain; int in, out; };
    std::vector<Swap> swaps(const std::vector<uint32_t>& usage, const std::vector<int>& primary,
                            int experts_per_layer, int max_swaps) const;
    const std::vector<uint8_t>& owners() const { return owned_; }
    bool owns(int key) const { return !owned_.empty() && owned_[(size_t) key] != 0; }
    bool integrated() const { return integrated_; }
    size_t available() const { return capacity_ - used_; }
    size_t used() const { return used_; }
    bool compute(int gu_type, int d_type, size_t down_off, float limit, int ne,
                 const int* keys, const float* weights, const float* x,
                 float* out, std::string& err);
    void close();
    int device() const { return device_; }

private:
    struct Region { const uint8_t* host; uint8_t* mapped; size_t bytes; };
    std::vector<Region> regions_;
    struct Slot { uint8_t* host = nullptr; uint8_t* device = nullptr; size_t bytes = 0; };
    std::vector<Slot> slots_;
    std::vector<uint8_t> owned_;
    bool integrated_ = false;
    uint8_t* pool_ = nullptr;
    size_t capacity_ = 0, used_ = 0;
    int device_ = -1, n_embd_ = 0, n_ff_ = 0;
    cudaStream_t stream_ = nullptr;
    uint8_t* arena_ = nullptr;
    float *x_ = nullptr, *out_ = nullptr;
    void *xq_ = nullptr, *hq_ = nullptr;
    strata::kernels::glmf::MoeDev plan_;
};

}  // namespace strata::core
