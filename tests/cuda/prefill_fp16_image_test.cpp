// prefill_fp16_image_test - the hyper-connection 16-bit activation images of the four writers (gr_norm,
// gr_norm_rs, gr_write_norm_rs, gr_write_cvec_norm_rs) when the device's images are FP16 bits
// (set_fp16_bits(true): Volta, no native BF16) *and* the writer is told f16img=true.
//
// f16img asks a writer for the FP16 values the Turing tensor-core path would convert its BF16 image to
// (bf_to_hf_conv).  When the images already ARE FP16 bits (act16 = hf_sat) there is nothing to convert, so
// f16img must be a no-op: the stored image is act16(value) = fp16(value) either way.
//
// The regression this pins: the writers applied bf_to_hf_conv() to the already-FP16 image, i.e. reinterpreted
// FP16 bits as BF16 and converted again (double conversion), so xn16 with f16img=true was a wrong image.
//
// Every writer is run on deterministic inputs and its image is compared bit for bit with (a) a reference built
// from the FP32 value the kernel itself produced (xn for gr_norm; (R * rs) * w for the others - the kernels'
// own operand order, no FMA) through the host replicas of the device funnels, and (b) the same writer's other
// f16img arm on the same FP16-bits device.  The BF16 arms (fp16 bits off) are checked too, so the conversion
// f16img performs on a BF16 image is still required to happen.
// Exit 77 without a CUDA device.
#include "strata/prefill/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int N = 2560, HC = 4, D = N * HC;   // kernels.cu
constexpr float EPS = 1e-6f;                  // prefill.cpp EPS

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// deterministic uniform RNG, bounded to the caller's range
struct Rng {
    uint64_t s = 0x243f6a8885a308d3ULL;
    float next(float lo, float hi) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        const float u = (float) ((s >> 11) * (1.0 / 9007199254740992.0));
        return lo + (hi - lo) * u;
    }
};

// ---- the device funnels of src/prefill/kernels.cu, replicated on the host (CUDA build: no __HIPCC__) ----
uint32_t f32_bits(float f) { uint32_t u; std::memcpy(&u, &f, sizeof u); return u; }
float bits_f32(uint32_t u) { float f; std::memcpy(&f, &u, sizeof f); return f; }

uint16_t bf_cpu(float f) {   // bf()
    uint32_t u = f32_bits(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
uint16_t hf_cpu(float f) { return __half_as_ushort(__float2half_rn(f)); }   // hf()
uint16_t hf_sat_cpu(float f) {   // hf_sat()
    const float c = std::isnan(f) ? f : std::fmin(std::fmax(f, -65504.0f), 65504.0f);
    return hf_cpu(c);
}
uint16_t bf_to_hf_cpu(uint16_t b) {   // bf_to_hf_conv()
    float f = bits_f32((uint32_t) b << 16);
    if (f > 65504.0f && !std::isinf(f)) f = 65504.0f;
    else if (f < -65504.0f && !std::isinf(f)) f = -65504.0f;
    return hf_cpu(f);
}
// act16(value) followed by the writer's f16img step
uint16_t image_cpu(float value, bool fp16bits, bool f16img) {
    const uint16_t h = fp16bits ? hf_sat_cpu(value) : bf_cpu(value);
    return (f16img && !fp16bits) ? bf_to_hf_cpu(h) : h;
}

struct Dev {
    void* p = nullptr;
    Dev() = default;
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    ~Dev() { if (p) cudaFree(p); }
    void alloc(size_t bytes, const char* what) { ck(cudaMalloc(&p, bytes), what); }
    template <class T> T* ptr() { return (T*) p; }
};

template <class T> void up(Dev& d, const std::vector<T>& v, const char* what) {
    d.alloc(v.size() * sizeof(T), what);
    ck(cudaMemcpy(d.p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), what);
}
template <class T> std::vector<T> down(const Dev& d, size_t n, const char* what) {
    std::vector<T> v(n);
    ck(cudaMemcpy(v.data(), d.p, n * sizeof(T), cudaMemcpyDeviceToHost), what);
    return v;
}

std::string mismatch(const char* what, bool fp16bits, bool f16img, size_t i, size_t n, uint16_t got,
                     uint16_t want, float value) {
    char b[320];
    std::snprintf(b, sizeof b,
                  "%s: fp16_bits=%d f16img=%d element %zu of %zu: got 0x%04x (%g) want 0x%04x (%g), value %g",
                  what, fp16bits ? 1 : 0, f16img ? 1 : 0, i, n, (unsigned) got,
                  (double) __half2float(__ushort_as_half(got)), (unsigned) want,
                  (double) __half2float(__ushort_as_half(want)), (double) value);
    return b;
}

// the image must be image_cpu(value) for every element
void expect_image(const char* what, const std::vector<uint16_t>& img, const std::vector<float>& value,
                  bool fp16bits, bool f16img) {
    for (size_t i = 0; i < img.size(); ++i) {
        const uint16_t want = image_cpu(value[i], fp16bits, f16img);
        if (img[i] != want) throw std::runtime_error(mismatch(what, fp16bits, f16img, i, img.size(), img[i], want, value[i]));
    }
}

void expect_same(const char* what, const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) {
            char buf[256];
            std::snprintf(buf, sizeof buf, "%s: element %zu of %zu: 0x%04x vs 0x%04x", what, i, a.size(),
                          (unsigned) a[i], (unsigned) b[i]);
            throw std::runtime_error(buf);
        }
}

// The FP32 value a writer's 16-bit image is of.  gr_norm's FP32 output already holds it; the others store the
// pre-norm value in R and the row scale in rs, and the kernel's value is (x * rs) * w (same order, no FMA).
std::vector<float> image_values(const std::vector<float>& base, const std::vector<float>& rs,
                                const std::vector<float>& w, bool base_is_value) {
    std::vector<float> value(base.size());
    for (size_t i = 0; i < base.size(); ++i) {
        if (base_is_value) { value[i] = base[i]; continue; }
        const size_t row = i / (size_t) N;
        const int c = (int) (row % HC), d = (int) (i % (size_t) N);
        value[i] = (base[i] * rs[row]) * w[(size_t) c * N + d];
    }
    return value;
}

struct Run {
    std::vector<float> base;   // xn (gr_norm) or R after the call
    std::vector<float> rs;     // row scales (empty for gr_norm)
    std::vector<uint16_t> img;
};

void check_gr_norm(cudaStream_t stream, int64_t T) {
    const size_t n = (size_t) T * D;
    std::vector<float> R(n), w((size_t) HC * N);
    Rng rng;
    for (float& v : R) v = rng.next(-1.0f, 1.0f);
    for (float& v : w) v = rng.next(-1.0f, 1.0f);
    Dev dR, dw, dxn, dx16;
    up(dR, R, "gr_norm R");
    up(dw, w, "gr_norm w");
    dxn.alloc(n * 4, "gr_norm xn");
    dx16.alloc(n * 2, "gr_norm xn16");

    const auto run = [&](bool fp16bits, bool f16img) {
        strata::prefill::set_fp16_bits(fp16bits);
        strata::prefill::gr_norm(dR.ptr<const float>(), dw.ptr<const float>(), EPS, dxn.ptr<float>(),
                                 dx16.ptr<uint16_t>(), T, stream, nullptr, f16img);
        ck(cudaStreamSynchronize(stream), "gr_norm sync");
        Run r;
        r.base = down<float>(dxn, n, "gr_norm xn");
        r.img = down<uint16_t>(dx16, n, "gr_norm xn16");
        return r;
    };

    const Run bb = run(false, false), bt = run(false, true), hb = run(true, false), ht = run(true, true);
    expect_image("gr_norm bf16 image", bb.img, bb.base, false, false);
    expect_image("gr_norm bf16 image, f16img", bt.img, bt.base, false, true);
    expect_image("gr_norm fp16-bits image", hb.img, hb.base, true, false);
    expect_image("gr_norm fp16-bits image, f16img", ht.img, ht.base, true, true);
    expect_same("gr_norm: f16img must not reconvert an FP16 image", ht.img, hb.img);
}

void check_gr_norm_rs(cudaStream_t stream, int64_t T) {
    const size_t n = (size_t) T * D, nr = (size_t) T * HC;
    std::vector<float> R(n), w((size_t) HC * N);
    Rng rng;
    for (float& v : R) v = rng.next(-1.0f, 1.0f);
    for (float& v : w) v = rng.next(-1.0f, 1.0f);
    Dev dR, dw, drs, dx16;
    up(dR, R, "gr_norm_rs R");
    up(dw, w, "gr_norm_rs w");
    drs.alloc(nr * 4, "gr_norm_rs rs");
    dx16.alloc(n * 2, "gr_norm_rs xn16");

    const auto run = [&](bool fp16bits, bool f16img) {
        strata::prefill::set_fp16_bits(fp16bits);
        strata::prefill::gr_norm_rs(dR.ptr<const float>(), dw.ptr<const float>(), EPS, drs.ptr<float>(),
                                    dx16.ptr<uint16_t>(), T, stream, nullptr, D, f16img);
        ck(cudaStreamSynchronize(stream), "gr_norm_rs sync");
        Run r;
        r.base = R;   // the kernel only reads it
        r.rs = down<float>(drs, nr, "gr_norm_rs rs");
        r.img = down<uint16_t>(dx16, n, "gr_norm_rs xn16");
        return r;
    };

    const Run bb = run(false, false), bt = run(false, true), hb = run(true, false), ht = run(true, true);
    expect_image("gr_norm_rs bf16 image", bb.img, image_values(bb.base, bb.rs, w, false), false, false);
    expect_image("gr_norm_rs bf16 image, f16img", bt.img, image_values(bt.base, bt.rs, w, false), false, true);
    expect_image("gr_norm_rs fp16-bits image", hb.img, image_values(hb.base, hb.rs, w, false), true, false);
    expect_image("gr_norm_rs fp16-bits image, f16img", ht.img, image_values(ht.base, ht.rs, w, false), true, true);
    expect_same("gr_norm_rs: f16img must not reconvert an FP16 image", ht.img, hb.img);
}

void check_gr_write_norm_rs(cudaStream_t stream, int64_t T) {
    const size_t n = (size_t) T * D, nr = (size_t) T * HC, nb = (size_t) T * N;
    std::vector<float> R(n), bo(nb), inj(nr), w((size_t) HC * N);
    Rng rng;
    for (float& v : R) v = rng.next(-1.0f, 1.0f);
    for (float& v : bo) v = rng.next(-1.0f, 1.0f);
    for (float& v : inj) v = rng.next(-4.0f, 4.0f);
    for (float& v : w) v = rng.next(-1.0f, 1.0f);
    Dev dR, dbo, dinj, dw, drs, dx16;
    up(dR, R, "gr_write_norm_rs R");
    up(dbo, bo, "gr_write_norm_rs bo");
    up(dinj, inj, "gr_write_norm_rs inj");
    up(dw, w, "gr_write_norm_rs w");
    drs.alloc(nr * 4, "gr_write_norm_rs rs");
    dx16.alloc(n * 2, "gr_write_norm_rs xn16");

    const auto run = [&](bool fp16bits, bool f16img) {
        ck(cudaMemcpy(dR.p, R.data(), n * 4, cudaMemcpyHostToDevice), "gr_write_norm_rs reset R");
        strata::prefill::set_fp16_bits(fp16bits);
        strata::prefill::gr_write_norm_rs(dR.ptr<float>(), dbo.ptr<const float>(), dinj.ptr<const float>(), HC,
                                          dw.ptr<const float>(), EPS, drs.ptr<float>(), dx16.ptr<uint16_t>(), T,
                                          stream, nullptr, D, f16img);
        ck(cudaStreamSynchronize(stream), "gr_write_norm_rs sync");
        Run r;
        r.base = down<float>(dR, n, "gr_write_norm_rs R");
        r.rs = down<float>(drs, nr, "gr_write_norm_rs rs");
        r.img = down<uint16_t>(dx16, n, "gr_write_norm_rs xn16");
        return r;
    };

    const Run bb = run(false, false), bt = run(false, true), hb = run(true, false), ht = run(true, true);
    expect_image("gr_write_norm_rs bf16 image", bb.img, image_values(bb.base, bb.rs, w, false), false, false);
    expect_image("gr_write_norm_rs bf16 image, f16img", bt.img, image_values(bt.base, bt.rs, w, false), false, true);
    expect_image("gr_write_norm_rs fp16-bits image", hb.img, image_values(hb.base, hb.rs, w, false), true, false);
    expect_image("gr_write_norm_rs fp16-bits image, f16img", ht.img, image_values(ht.base, ht.rs, w, false), true, true);
    expect_same("gr_write_norm_rs: f16img must not reconvert an FP16 image", ht.img, hb.img);
}

void check_gr_write_cvec_norm_rs(cudaStream_t stream, int64_t T) {
    const size_t n = (size_t) T * D, nr = (size_t) T * HC, nb = (size_t) T * N;
    std::vector<float> R(n), bo(nb), inj(nr), vl(N), w((size_t) HC * N);
    Rng rng;
    for (float& v : R) v = rng.next(-1.0f, 1.0f);
    for (float& v : bo) v = rng.next(-1.0f, 1.0f);
    for (float& v : inj) v = rng.next(-4.0f, 4.0f);
    for (float& v : vl) v = rng.next(-1.0f, 1.0f);
    for (float& v : w) v = rng.next(-1.0f, 1.0f);
    const std::vector<float> sl{0.5f};
    const std::vector<int> on{1};
    Dev dR, dbo, dinj, dvl, dsl, don, dw, drs, dx16;
    up(dR, R, "gr_write_cvec_norm_rs R");
    up(dbo, bo, "gr_write_cvec_norm_rs bo");
    up(dinj, inj, "gr_write_cvec_norm_rs inj");
    up(dvl, vl, "gr_write_cvec_norm_rs v_l");
    up(dsl, sl, "gr_write_cvec_norm_rs s_l");
    up(don, on, "gr_write_cvec_norm_rs on");
    up(dw, w, "gr_write_cvec_norm_rs w");
    drs.alloc(nr * 4, "gr_write_cvec_norm_rs rs");
    dx16.alloc(n * 2, "gr_write_cvec_norm_rs xn16");

    const auto run = [&](bool fp16bits, bool f16img) {
        ck(cudaMemcpy(dR.p, R.data(), n * 4, cudaMemcpyHostToDevice), "gr_write_cvec_norm_rs reset R");
        strata::prefill::set_fp16_bits(fp16bits);
        strata::prefill::gr_write_cvec_norm_rs(dR.ptr<float>(), dbo.ptr<const float>(), dinj.ptr<const float>(), HC,
                                               dvl.ptr<const float>(), dsl.ptr<const float>(), don.ptr<const int>(),
                                               0 /* project */, dw.ptr<const float>(), EPS, drs.ptr<float>(),
                                               dx16.ptr<uint16_t>(), T, stream, nullptr, D, f16img);
        ck(cudaStreamSynchronize(stream), "gr_write_cvec_norm_rs sync");
        Run r;
        r.base = down<float>(dR, n, "gr_write_cvec_norm_rs R");
        r.rs = down<float>(drs, nr, "gr_write_cvec_norm_rs rs");
        r.img = down<uint16_t>(dx16, n, "gr_write_cvec_norm_rs xn16");
        return r;
    };

    const Run bb = run(false, false), bt = run(false, true), hb = run(true, false), ht = run(true, true);
    expect_image("gr_write_cvec_norm_rs bf16 image", bb.img, image_values(bb.base, bb.rs, w, false), false, false);
    expect_image("gr_write_cvec_norm_rs bf16 image, f16img", bt.img, image_values(bt.base, bt.rs, w, false), false, true);
    expect_image("gr_write_cvec_norm_rs fp16-bits image", hb.img, image_values(hb.base, hb.rs, w, false), true, false);
    expect_image("gr_write_cvec_norm_rs fp16-bits image, f16img", ht.img, image_values(ht.base, ht.rs, w, false), true, true);
    expect_same("gr_write_cvec_norm_rs: f16img must not reconvert an FP16 image", ht.img, hb.img);
}

}  // namespace

int main() {
    try {
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) { std::printf("no CUDA device: skipped\n"); return 77; }
        cudaStream_t stream = nullptr;
        ck(cudaStreamCreate(&stream), "stream");
        const int64_t Ts[] = {1, 2};
        for (int64_t T : Ts) {
            check_gr_norm(stream, T);
            check_gr_norm_rs(stream, T);
            check_gr_write_norm_rs(stream, T);
            check_gr_write_cvec_norm_rs(stream, T);
        }
        strata::prefill::set_fp16_bits(false);
        ck(cudaStreamDestroy(stream), "destroy");
        std::printf("prefill FP16 activation images (f16img on an FP16-bits device) passed\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "prefill FP16 activation image test failed: %s\n", e.what());
        return 1;
    }
}
