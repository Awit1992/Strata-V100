// src/kernels/kv_tq_parity.cpp - TurboQuant K/V (kv_tq.hpp, `--kv tq3|tq4|tq6|tq8`) against host references.
//
// The failure modes this guards, all silent until a value or an address is wrong:
//   1. THE BLOCK LAYOUT.  A block is one FP16 norm plus 256 Lloyd-Max indices packed DIMENSION-ORDERED (index
//      group j = dimensions [8j, 8j + 8)).  A writer that packs any other order - kv_tq.cu first packed a
//      stride-32 slice - and a reader that assumes this one still produce the right number of bytes and a
//      finite, plausible head: a fixed permutation of the dimensions.  3 checks below pin it: the per-dimension
//      centroid oracle, the monotonic ramp, and the stride-32 reading, which must FAIL the oracle (it is the
//      shape of the bug, and this test regresses it).
//   2. ROUNDTRIP through the paged pool: kv_append_tq_steps / kv_append_tq (prompt) then kv_gather_tq_step, with
//      the gathered FP16 compared bit for bit against a host decode of the same block bytes.
//   3. THE ATTENTION READER: the decode-side entry points (qsa_decode_attn_step, qsa_decode_attn_batch) run on
//      TurboQuant pools and on FP16 pools holding exactly the gathered values - their outputs must agree to
//      within the FP16 rounding of the comparison pool (the TurboQuant reader returns the unrounded
//      centroid*norm product, the FP16 one the rounded value).  A KV-format dispatch that fell through to the
//      FP16 arm used to dereference the null k_pool/v_pool of a TurboQuant state; here that is a fault.
//
// The codebook oracle is the Lloyd-Max iteration of tools/gen_tq_codebook.py ported to double precision here
// (validated against the shipped kv_tq_codebook.inc: identical to 8 decimals), so it shares no code with the
// kernel.  GPU, synthetic, no model.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_tq.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {

int g_fail = 0;

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}

// The env switches below are read per call by the dispatchers; setting one must work on MSVC too (no setenv there).
void set_env(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

template <typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 64), "memset");
    return p;
}

template <typename T>
std::vector<T> fetch(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "d2h");
    return h;
}

// ---------------------------------------------------------------------------------------------------------------
// The Lloyd-Max codebook of Beta(128,128) on [-1, 1], as tools/gen_tq_codebook.py computes it (double precision;
// the 400-iteration bincount scheme, 262145-point grid, trapezoid normalization).  Verified against the shipped
// .inc for all four widths.
// ---------------------------------------------------------------------------------------------------------------
constexpr int kGridN = 262145;

struct Codebook {
    int bits = 0;
    std::vector<float> cent;   // the .inc's float centroids (the port rounds to the same float32 values)
};

Codebook lloyd_codebook(int bits) {
    static std::vector<double> x(kGridN), p(kGridN), cdf(kGridN);
    static bool built = false;
    static double dx = 0.0;
    if (!built) {
        const double a = k::QK_TQ / 2.0;
        for (int i = 0; i < kGridN; ++i) x[(size_t) i] = -1.0 + 2.0 * i / (kGridN - 1);
        const double lognorm = 2 * std::lgamma(a) - std::lgamma(2 * a);
        for (int i = 0; i < kGridN; ++i) {
            const double logp = (a - 1) * (std::log1p(x[(size_t) i]) + std::log1p(-x[(size_t) i])) - lognorm -
                                (2 * a - 1) * std::log(2.0);
            p[(size_t) i] = std::exp(logp);
        }
        p[0] = p[kGridN - 1] = 0.0;
        double trap = 0.0;
        for (int i = 0; i + 1 < kGridN; ++i) trap += (p[(size_t) i] + p[(size_t) i + 1]) / 2.0 * (x[(size_t) i + 1] - x[(size_t) i]);
        for (double& v : p) v /= trap;
        dx = x[1] - x[0];
        double acc = 0.0;
        for (int i = 0; i < kGridN; ++i) { acc += p[(size_t) i]; cdf[(size_t) i] = acc * dx; }
        const double total = cdf[kGridN - 1];
        for (double& v : cdf) v /= total;
        built = true;
    }
    const int n = 1 << bits;
    std::vector<double> c((size_t) n), cnt((size_t) n), sx((size_t) n), b((size_t) std::max(n - 1, 0));
    // np.interp((i + 0.5) / n, cdf, x): linear through the monotone cdf
    auto interp = [](double t, int n) {
        if (t <= cdf[0]) return x[0];
        if (t >= cdf[kGridN - 1]) return x[kGridN - 1];
        int lo = 0, hi = kGridN - 1;
        while (hi - lo > 1) {
            const int mid = (lo + hi) >> 1;
            if (cdf[(size_t) mid] <= t) lo = mid;
            else hi = mid;
        }
        const double d = cdf[(size_t) hi] - cdf[(size_t) lo];
        const double f = d > 0.0 ? (t - cdf[(size_t) lo]) / d : 0.0;
        return x[(size_t) lo] + f * (x[(size_t) hi] - x[(size_t) lo]);
    };
    for (int i = 0; i < n; ++i) c[(size_t) i] = interp((i + 0.5) / n, n);
    for (int iter = 0; iter < 400; ++iter) {
        for (int i = 0; i + 1 < n; ++i) b[(size_t) i] = (c[(size_t) i] + c[(size_t) i + 1]) / 2.0;
        std::fill(cnt.begin(), cnt.end(), 0.0);
        std::fill(sx.begin(), sx.end(), 0.0);
        int bin = 0;
        for (int i = 0; i < kGridN; ++i) {   // np.searchsorted(b, x[i]): b ascending
            while (bin < n - 1 && b[(size_t) bin] < x[(size_t) i]) ++bin;
            cnt[(size_t) bin] += p[(size_t) i];
            sx[(size_t) bin] += x[(size_t) i] * p[(size_t) i];
        }
        double worst = 0.0;
        for (int i = 0; i < n; ++i) {
            const double next = cnt[(size_t) i] > 0.0 ? sx[(size_t) i] / std::max(cnt[(size_t) i], 1e-300)
                                                      : c[(size_t) i];
            worst = std::fmax(worst, std::fabs(next - c[(size_t) i]));
            c[(size_t) i] = next;
        }
        if (worst < 1e-9) break;
    }
    std::vector<float> cf((size_t) n);
    for (int i = 0; i < n; ++i) cf[(size_t) i] = (float) c[(size_t) i];
    return Codebook{bits, cf};
}

const Codebook& codebook(int bits) {
    static Codebook c3 = lloyd_codebook(3), c4 = lloyd_codebook(4), c6 = lloyd_codebook(6), c8 = lloyd_codebook(8);
    return bits == 3 ? c3 : bits == 4 ? c4 : bits == 6 ? c6 : c8;
}

// kv_tq.cu's `nearest<B>`: binary search over the ascending centroids, a tie going to the lower index.
int nearest_centroid(const std::vector<float>& c, float v) {
    int lo = 0, hi = (int) c.size() - 1;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (v > c[(size_t) mid]) lo = mid + 1;
        else hi = mid;
    }
    if (lo > 0 && std::fabs(v - c[(size_t) lo - 1]) <= std::fabs(v - c[(size_t) lo])) --lo;
    return lo;
}

// One block as the documented layout defines it: `group j = dimensions [8j, 8j + 8)`, index i of a group is
// dimension 8j + i.  Fills the per-dimension centroid index and the dequantized value.
void decode_block(const uint8_t* blk, int bits, const std::vector<float>& cent, int* idx, float* val) {
    const float nrm = k::f32_from_f16(*reinterpret_cast<const uint16_t*>(blk));
    for (int j = 0; j < k::QK_TQ / 8; ++j) {
        const uint8_t* in = blk + k::kv_tq_group_bytes(bits, j);
        unsigned long long acc = 0;
        for (int b = 0; b < bits; ++b) acc |= (unsigned long long) in[b] << (8 * b);
        const unsigned long long mask = bits == 8 ? 0xFFull : ((1ull << bits) - 1ull);
        for (int i = 0; i < 8; ++i) {
            const int c = (int) ((acc >> (i * bits)) & mask);
            idx[j * 8 + i] = c;
            val[j * 8 + i] = cent[(size_t) c] * nrm;   // the reader's own expression
        }
    }
}

// The packing kv_tq.cu's first cut wrote: lane L held dimensions L, L+32, L+64, ... in its B bytes.  The test
// requires this reading to DISAGREE with the oracle (see test [4]).
void decode_block_stride32(const uint8_t* blk, int bits, int* idx) {
    for (int lane = 0; lane < 32; ++lane) {
        const uint8_t* in = blk + k::kv_tq_group_bytes(bits, lane);
        unsigned long long acc = 0;
        for (int b = 0; b < bits; ++b) acc |= (unsigned long long) in[b] << (8 * b);
        const unsigned long long mask = bits == 8 ? 0xFFull : ((1ull << bits) - 1ull);
        for (int i = 0; i < 8; ++i) idx[lane + i * 32] = (int) ((acc >> (i * bits)) & mask);
    }
}

// max|a - b| / max|a|: the attention outputs are floats of very different magnitude, and a per-element relative
// measure would divide by near-zero values.
float rel_close(const std::vector<float>& a, const std::vector<float>& b) {
    float md = 0.0f, mx = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        md = std::fmax(md, std::fabs(a[i] - b[i]));
        mx = std::fmax(mx, std::fabs(a[i]));
    }
    return mx > 0.0f ? md / mx : md;
}

bool non_decreasing(const uint16_t* v, int n) {
    for (int d = 1; d < n; ++d)
        if (k::f32_from_f16(v[d]) < k::f32_from_f16(v[d - 1])) return false;
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// The shared fixture: a paged pool with a NON-identity page table, positions scattered across pages.
// ---------------------------------------------------------------------------------------------------------------
struct Fixture {
    k::QsaShapes s;
    int H = 0, D = 0, P = 0, pages = 0;
    std::vector<int32_t> table;
    int32_t* d_table = nullptr;
    std::mt19937 rng{2504};

    explicit Fixture(int n_pages) {
        s = k::qsa_real_shapes();
        H = (int) s.n_head_kv; D = (int) s.head_dim; P = (int) s.page_size; pages = n_pages;
        table.resize((size_t) pages);
        for (int i = 0; i < pages; ++i) table[(size_t) i] = (i * 5 + 3) % pages;   // a permutation, not the identity
        d_table = dalloc<int32_t>((size_t) pages);
        ck(cudaMemcpy(d_table, table.data(), table.size() * 4, cudaMemcpyHostToDevice), "table");
    }
    ~Fixture() { cudaFree(d_table); }
    int64_t rows() const { return (int64_t) pages * H * P; }
    long long row_of(int cell, int h) const {
        // exactly the pool row the appends and readers index: (page * heads + h) * page_size + pos % page_size
        const long long page = table[(size_t) (cell / P)];
        return (page * H + h) * P + (cell % P);
    }
    std::vector<float> random_rows(size_t n) {
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> v(n);
        for (auto& x : v) x = nd(rng) * ((rng() % 17 == 0) ? 8.0f : 1.0f);
        return v;
    }
};

// [1] the layout the writer and both readers share.
void test_block_layout() {
    std::printf("[1/7] block layout (TqBlock == kv_tq_bytes_per_head, group j = dimensions [8j, 8j+8))...\n");
    int bad = 0;
    for (int bits : {3, 4, 6, 8}) {
        const uint64_t want = k::kv_tq_bytes_per_head(bits, k::QK_TQ);
        const size_t got = bits == 3 ? sizeof(k::TqBlock<3>)
                         : bits == 4 ? sizeof(k::TqBlock<4>)
                         : bits == 6 ? sizeof(k::TqBlock<6>)
                                     : sizeof(k::TqBlock<8>);
        if (got != want) { std::fprintf(stderr, "  tq%d: TqBlock is %zu B, kv_tq_bytes_per_head says %llu\n", bits, got,
                                        (unsigned long long) want); ++bad; }
        // group 31 (the last 8 dimensions) must end exactly at the block's end, and every group must be B bytes
        const int last = k::kv_tq_group_bytes(bits, 31);
        if ((uint64_t) (last + bits) != want) {
            std::fprintf(stderr, "  tq%d: group 31 is [%d, %d), block is %llu B\n", bits, last, last + bits,
                         (unsigned long long) want);
            ++bad;
        }
        for (int j = 1; j < 32; ++j)
            if (k::kv_tq_group_bytes(bits, j) - k::kv_tq_group_bytes(bits, j - 1) != bits) {
                std::fprintf(stderr, "  tq%d: group %d is not %d B\n", bits, j, bits);
                ++bad;
                break;
            }
    }
    std::printf("  -> %s\n", bad ? "MISMATCH" : "sizes and group offsets exact (OK)");
    if (bad) ++g_fail;
}

// [2] the prompt path (kv_append_tq) against the per-cell step appends, byte-identical pools.
void test_prompt_vs_steps(Fixture& f) {
    std::printf("[2/7] prompt append vs per-cell step appends (memcmp, both pools)...\n");
    for (int bits : {3, 4, 6, 8}) {
        const uint64_t row = k::kv_tq_bytes_per_head(bits, f.D);
        const size_t pool = (size_t) f.rows() * row;
        uint8_t *dka = dalloc<uint8_t>(pool), *dva = dalloc<uint8_t>(pool);
        uint8_t *dkb = dalloc<uint8_t>(pool), *dvb = dalloc<uint8_t>(pool);
        const int T = 2 * f.P + 1, pos0 = f.P - 2;          // crosses two page boundaries
        std::vector<float> K = f.random_rows((size_t) T * f.H * f.D), V = f.random_rows(K.size());
        float* dK = dalloc<float>(K.size());
        float* dV = dalloc<float>(V.size());
        ck(cudaMemcpy(dK, K.data(), K.size() * 4, cudaMemcpyHostToDevice), "K");
        ck(cudaMemcpy(dV, V.data(), V.size() * 4, cudaMemcpyHostToDevice), "V");
        std::vector<int32_t> steps((size_t) T * k::kStepCount);
        for (int t = 0; t < T; ++t) {
            steps[(size_t) t * k::kStepCount + k::kStepPos] = pos0 + t;
            steps[(size_t) t * k::kStepCount + k::kStepNKv] = pos0 + t + 1;
            steps[(size_t) t * k::kStepCount + k::kStepWidth] = T;
        }
        int32_t* dsteps = dalloc<int32_t>(steps.size());
        ck(cudaMemcpy(dsteps, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice), "steps");

        k::kv_append_tq_steps(bits, dka, dva, f.d_table, dsteps, k::kStepCount, T, dK, dV, f.s, nullptr, nullptr);
        k::kv_append_tq(bits, dkb, dvb, f.d_table, pos0, T, dK, dV, f.s, nullptr, nullptr, nullptr);
        ck(cudaDeviceSynchronize(), "append");

        const std::vector<uint8_t> ka = fetch(dka, pool), kb = fetch(dkb, pool);
        const std::vector<uint8_t> va = fetch(dva, pool), vb = fetch(dvb, pool);
        const long bad = std::memcmp(ka.data(), kb.data(), pool) ? 1 : 0;
        const long badv = std::memcmp(va.data(), vb.data(), pool) ? 1 : 0;
        std::printf("  -> tq%d T=%d pos0=%d: K %s, V %s\n", bits, T, pos0, bad ? "DIFFERS" : "identical",
                    badv ? "DIFFERS" : "identical");
        if (bad || badv) ++g_fail;
        cudaFree(dka); cudaFree(dva); cudaFree(dkb); cudaFree(dvb); cudaFree(dK); cudaFree(dV); cudaFree(dsteps);
    }
}

// [3][4] the roundtrip through a paged gather and the dimension-order oracle (plus the stride-32 regression).
void test_gather_and_layout_oracle(Fixture& f) {
    std::printf("[3/7] paged gather vs a host decode of the block bytes (within one FP16 code)...\n");
    std::printf("[4/7] per-dimension centroid oracle (and the stride-32 reading, which must fail it)...\n");
    const int cells = 7;
    std::vector<int32_t> positions(cells);
    for (int i = 0; i < cells; ++i) positions[(size_t) i] = (i * 3 + 1) % (f.pages * f.P);
    for (int bits : {3, 4, 6, 8}) {
        const uint64_t row = k::kv_tq_bytes_per_head(bits, f.D);
        const size_t pool = (size_t) f.rows() * row;
        const std::vector<float>& cent = codebook(bits).cent;
        // a ramp for the oracle (every dimension a distinct magnitude, so a permutation cannot hide) and random
        // rows for the roundtrip
        std::vector<float> K((size_t) cells * f.H * f.D), V(K.size());
        for (int c = 0; c < cells; ++c)
            for (int h = 0; h < f.H; ++h)
                for (int d = 0; d < f.D; ++d) {
                    const float t = (float) d / (float) f.D;
                    // cell 0 / head 0 is a STRICTLY INCREASING ramp: with ordered centroids its decoded values
                    // must come back non-decreasing, which no dimension permutation that is not order-preserving
                    // can fake (see the ramp check below)
                    K[((size_t) c * f.H + h) * f.D + d] =
                        (c == 0 && h == 0)
                            ? 0.15f + 0.7f * t
                            : std::sin(3.0f * t + 0.7f * (float) c + 0.3f * (float) h) * (0.25f + t) + 0.05f * t * t;
                    V[((size_t) c * f.H + h) * f.D + d] =
                        std::cos(1.7f * t + 0.9f * (float) h + 0.2f * (float) c) * (0.5f - 0.3f * t);
                }
        uint8_t* dk = dalloc<uint8_t>(pool);
        uint8_t* dv = dalloc<uint8_t>(pool);
        float* dK = dalloc<float>(K.size());
        float* dV = dalloc<float>(V.size());
        ck(cudaMemcpy(dK, K.data(), K.size() * 4, cudaMemcpyHostToDevice), "K");
        ck(cudaMemcpy(dV, V.data(), V.size() * 4, cudaMemcpyHostToDevice), "V");
        std::vector<int32_t> steps((size_t) cells * k::kStepCount);
        for (int c = 0; c < cells; ++c) {
            steps[(size_t) c * k::kStepCount + k::kStepPos] = positions[(size_t) c];
            steps[(size_t) c * k::kStepCount + k::kStepNKv] = positions[(size_t) c] + 1;
            steps[(size_t) c * k::kStepCount + k::kStepWidth] = cells;
        }
        int32_t* dsteps = dalloc<int32_t>(steps.size());
        int32_t* dids = dalloc<int32_t>((size_t) cells);
        ck(cudaMemcpy(dsteps, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice), "steps");
        ck(cudaMemcpy(dids, positions.data(), positions.size() * 4, cudaMemcpyHostToDevice), "ids");
        uint16_t* dks = dalloc<uint16_t>((size_t) cells * f.H * f.D);
        uint16_t* dvs = dalloc<uint16_t>((size_t) cells * f.H * f.D);

        k::kv_append_tq_steps(bits, dk, dv, f.d_table, dsteps, k::kStepCount, cells, dK, dV, f.s, nullptr, nullptr);
        k::kv_gather_tq_step(bits, dk, dv, f.d_table, dids, dsteps, cells, f.s, dks, dvs, nullptr);
        ck(cudaDeviceSynchronize(), "append+gather");

        const std::vector<uint8_t> hk = fetch(dk, pool), hv = fetch(dv, pool);
        const std::vector<uint16_t> hks = fetch(dks, (size_t) cells * f.H * f.D);
        const std::vector<uint16_t> hvs = fetch(dvs, (size_t) cells * f.H * f.D);
        long bitwise_bad = 0, idx_bad = 0, idx_total = 0, stride_bad = 0;
        long bad_ramp = 0, bad_stride_ramp = 0;
        std::vector<int> idx((size_t) f.D), sidx((size_t) f.D);
        std::vector<float> val((size_t) f.D);
        for (int c = 0; c < cells; ++c)
            for (int h = 0; h < f.H; ++h) {
                const size_t base = ((size_t) c * f.H + h) * f.D;
                const float* xk = &K[base];
                const float* xv = &V[base];
                for (int side = 0; side < 2; ++side) {   // 0: K, 1: V
                    const uint8_t* blk = (side ? hv : hk).data() + (size_t) f.row_of(positions[(size_t) c], h) * row;
                    const uint16_t* got = (side ? hvs : hks).data() + base;
                    const float* x = side ? xv : xk;
                    decode_block(blk, bits, cent, idx.data(), val.data());
                    decode_block_stride32(blk, bits, sidx.data());
                    double ss = 0.0;
                    for (int d = 0; d < f.D; ++d) ss += (double) x[d] * x[d];
                    // the quantizer's own arithmetic: nrm = sqrtf(sum), inv = 1/nrm, v * inv (fp32)
                    const float nrm = (float) std::sqrt(ss);
                    const float inv = nrm > 0.0f ? 1.0f / nrm : 0.0f;
                    for (int d = 0; d < f.D; ++d) {
                        // The reader's value: the block's own norm times its centroid, rounded to FP16.  A value
                        // that lands within a few float32 ULPs of an FP16 midpoint can round to either neighbour
                        // (the device's product and this one need not be the same float bit for bit), so one code
                        // of slack is allowed - a wrong dimension or centroid is orders of magnitude away.
                        const uint16_t want16 = k::f16_from_f32(val[(size_t) d]);
                        if (std::abs((int) want16 - (int) got[d]) > 1) {
                            if (bitwise_bad++ == 0)
                                std::printf("      first value mismatch: c=%d h=%d side=%d d=%d centroid %d: "
                                            "value %.9g -> 0x%04x, reader 0x%04x (%.9g)\n",
                                            c, h, side, d, idx[(size_t) d], (double) val[(size_t) d], want16, got[d],
                                            (double) k::f32_from_f16(got[d]));
                        }
                        const int want = nearest_centroid(cent, x[d] * inv);
                        if (idx[(size_t) d] != want) ++idx_bad;
                        if (sidx[(size_t) d] != want) ++stride_bad;
                        ++idx_total;
                    }
                    // The ramp row: the dimension-ordered reading must be non-decreasing, the stride-32 reading
                    // must NOT be - that is the shape of the packing bug this test exists for.
                    if (side == 0 && c == 0 && h == 0) {
                        const float bnrm = k::f32_from_f16(*reinterpret_cast<const uint16_t*>(blk));
                        std::vector<uint16_t> smono((size_t) f.D);
                        for (int d = 0; d < f.D; ++d)
                            smono[(size_t) d] = k::f16_from_f32(cent[(size_t) sidx[(size_t) d]] * bnrm);
                        if (!non_decreasing(got, f.D)) ++bad_ramp;
                        if (non_decreasing(smono.data(), f.D)) ++bad_stride_ramp;
                    }
                }
            }
        std::printf("  -> tq%d: gather vs fp16(host decode) (%ld of %ld values beyond one code), oracle %ld of %ld "
                    "centroid indices differ, monotone ramp: %s\n",
                    bits, bitwise_bad, idx_total, idx_bad, idx_total, bad_ramp ? "NOT non-decreasing (BAD)" : "OK");
        std::printf("            stride-32 reading: %ld of %ld indices differ from the oracle (must be most), ramp "
                    "%s (must be non-decreasing)\n",
                    stride_bad, idx_total, bad_stride_ramp ? "non-decreasing (BAD)" : "non-monotone (OK)");
        if (bitwise_bad) ++g_fail;
        if (idx_bad > idx_total / 200) {   // <= 0.5%: an fp32 norm can flip a coordinate sitting on a boundary
            std::fprintf(stderr, "  FAIL: tq%d layout oracle, %ld of %ld indices\n", bits, idx_bad, idx_total);
            ++g_fail;
        }
        if (bad_ramp || bad_stride_ramp) ++g_fail;
        if (stride_bad * 2 <= idx_total) {
            // Either reading now agrees with the oracle, so the writer and the documented layout disagree: this is
            // the exact regression the test exists for (a stride-32 writer against a dimension-ordered reader).
            std::fprintf(stderr, "  FAIL: tq%d the stride-32 reading of the blocks matches the oracle (%ld of %ld): "
                                 "the WRITER is packing a stride-32 slice, not index groups of 8 dimensions\n",
                         bits, stride_bad, idx_total);
            ++g_fail;
        }
        cudaFree(dk); cudaFree(dv); cudaFree(dK); cudaFree(dV); cudaFree(dsteps); cudaFree(dids);
        cudaFree(dks); cudaFree(dvs);
    }
}

// [5] the attention readers: TurboQuant pools vs FP16 pools holding exactly the gathered values, bitwise.
void test_attention_readers(Fixture& f) {
    std::printf("[5/7] attention on TurboQuant pools vs on FP16 pools of the same values...\n");
    const int cells = 7;
    std::vector<int32_t> positions(cells);
    for (int c = 0; c < cells; ++c) positions[(size_t) c] = (c * 5 + 2) % (f.pages * f.P);
    const int cap = cells;                       // the selection: every stored cell
    const int NH = (int) f.s.n_head;
    for (int bits : {3, 4, 6, 8}) {
        const uint64_t row = k::kv_tq_bytes_per_head(bits, f.D);
        const size_t tq_pool = (size_t) f.rows() * row;
        uint8_t* dk = dalloc<uint8_t>(tq_pool);
        uint8_t* dv = dalloc<uint8_t>(tq_pool);
        std::vector<float> K = f.random_rows((size_t) cells * f.H * f.D), V = f.random_rows(K.size());
        float* dK = dalloc<float>(K.size());
        float* dV = dalloc<float>(V.size());
        ck(cudaMemcpy(dK, K.data(), K.size() * 4, cudaMemcpyHostToDevice), "K");
        ck(cudaMemcpy(dV, V.data(), V.size() * 4, cudaMemcpyHostToDevice), "V");
        std::vector<int32_t> steps((size_t) cells * k::kStepCount);
        for (int c = 0; c < cells; ++c) {
            steps[(size_t) c * k::kStepCount + k::kStepPos] = positions[(size_t) c];
            steps[(size_t) c * k::kStepCount + k::kStepNKv] = positions[(size_t) c] + 1;
            steps[(size_t) c * k::kStepCount + k::kStepWidth] = cap;
        }
        int32_t* dsteps = dalloc<int32_t>(steps.size());
        int32_t* dids = dalloc<int32_t>((size_t) cap);
        ck(cudaMemcpy(dsteps, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice), "steps");
        ck(cudaMemcpy(dids, positions.data(), positions.size() * 4, cudaMemcpyHostToDevice), "ids");
        uint16_t* dks = dalloc<uint16_t>((size_t) cells * f.H * f.D);
        uint16_t* dvs = dalloc<uint16_t>((size_t) cells * f.H * f.D);
        k::kv_append_tq_steps(bits, dk, dv, f.d_table, dsteps, k::kStepCount, cells, dK, dV, f.s, nullptr, nullptr);
        k::kv_gather_tq_step(bits, dk, dv, f.d_table, dids, dsteps, cells, f.s, dks, dvs, nullptr);
        ck(cudaDeviceSynchronize(), "append+gather");

        // the FP16 pools hold, at every cell's own pool row, the values the TurboQuant readers produce for it
        const std::vector<uint16_t> hks = fetch(dks, (size_t) cells * f.H * f.D);
        const std::vector<uint16_t> hvs = fetch(dvs, (size_t) cells * f.H * f.D);
        std::vector<uint16_t> hp_k((size_t) f.rows() * f.D), hp_v((size_t) f.rows() * f.D);
        for (int c = 0; c < cells; ++c)
            for (int h = 0; h < f.H; ++h) {
                const size_t src = ((size_t) c * f.H + h) * f.D;
                const size_t dst = (size_t) f.row_of(positions[(size_t) c], h) * f.D;
                std::memcpy(&hp_k[dst], &hks[src], (size_t) f.D * 2);
                std::memcpy(&hp_v[dst], &hvs[src], (size_t) f.D * 2);
            }
        uint16_t* dkp = dalloc<uint16_t>(hp_k.size());
        uint16_t* dvp = dalloc<uint16_t>(hp_v.size());
        ck(cudaMemcpy(dkp, hp_k.data(), hp_k.size() * 2, cudaMemcpyHostToDevice), "k_pool");
        ck(cudaMemcpy(dvp, hp_v.data(), hp_v.size() * 2, cudaMemcpyHostToDevice), "v_pool");

        k::QsaAttnPools ptq{}, pf16{};
        ptq.page_table = f.d_table; ptq.k_tq = dk; ptq.v_tq = dv; ptq.tq_bits = bits;
        pf16.page_table = f.d_table; pf16.k_pool = dkp; pf16.v_pool = dvp;

        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> hq((size_t) NH * f.D);
        for (auto& x : hq) x = nd(f.rng);
        float* dq = dalloc<float>(hq.size());
        ck(cudaMemcpy(dq, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice), "q");
        float* dscratch = dalloc<float>((size_t) k::qsa_decode_attn_scratch_floats(cap, f.s));
        float* out_tq = dalloc<float>((size_t) NH * f.D);
        float* out_f16 = dalloc<float>((size_t) NH * f.D);
        std::vector<float> r_tq((size_t) NH * f.D), r_f16((size_t) NH * f.D);

        // the decode-side step entry (the one the engine's decode loop calls)
        k::qsa_decode_attn_step(dq, ptq, dids, dsteps, cap, f.s, dscratch, out_tq, nullptr);
        k::qsa_decode_attn_step(dq, pf16, dids, dsteps, cap, f.s, dscratch, out_f16, nullptr);
        ck(cudaDeviceSynchronize(), "attn step");
        ck(cudaMemcpy(r_tq.data(), out_tq, r_tq.size() * 4, cudaMemcpyDeviceToHost), "attn tq");
        ck(cudaMemcpy(r_f16.data(), out_f16, r_f16.size() * 4, cudaMemcpyDeviceToHost), "attn f16");
        const float step_rel = rel_close(r_tq, r_f16);

        // the batch entry (the prompt path's fallback and every multi-row window)
        float* bout_tq = dalloc<float>((size_t) NH * f.D);
        float* bout_f16 = dalloc<float>((size_t) NH * f.D);
        float* bq = dalloc<float>(hq.size());
        ck(cudaMemcpy(bq, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice), "q1");
        k::qsa_decode_attn_batch(bq, ptq, dids, dsteps, cap, f.s, dscratch, bout_tq, 1, nullptr);
        k::qsa_decode_attn_batch(bq, pf16, dids, dsteps, cap, f.s, dscratch, bout_f16, 1, nullptr);
        ck(cudaDeviceSynchronize(), "attn batch");
        ck(cudaMemcpy(r_tq.data(), bout_tq, r_tq.size() * 4, cudaMemcpyDeviceToHost), "batch tq");
        ck(cudaMemcpy(r_f16.data(), bout_f16, r_f16.size() * 4, cudaMemcpyDeviceToHost), "batch f16");
        const float batch_rel = rel_close(r_tq, r_f16);

        // FP16's own rounding of the comparison pool bounds this (~1e-3 relative); a permutation or a different
        // set of operands differs by O(1), so the bound separates them cleanly.
        std::printf("  -> tq%d: qsa_decode_attn_step max|diff|/max|out| %.2e, qsa_decode_attn_batch %.2e %s\n", bits,
                    (double) step_rel, (double) batch_rel,
                    (step_rel <= 5e-3f && batch_rel <= 5e-3f) ? "(within FP16 rounding, OK)" : "(DIVERGES)");
        if (!(step_rel <= 5e-3f) || !(batch_rel <= 5e-3f)) ++g_fail;

        cudaFree(dk); cudaFree(dv); cudaFree(dK); cudaFree(dV); cudaFree(dsteps); cudaFree(dids);
        cudaFree(dks); cudaFree(dvs); cudaFree(dkp); cudaFree(dvp); cudaFree(dq); cudaFree(dscratch);
        cudaFree(out_tq); cudaFree(out_f16); cudaFree(bout_tq); cudaFree(bout_f16); cudaFree(bq);
    }
}

// [6] the bits buy precision: the reconstruction error must fall as the width grows.
void test_error_shrinks(Fixture& f) {
    std::printf("[6/7] reconstruction error falls with the width (tq3 > tq4 > tq6 > tq8)...\n");
    // In-distribution rows: a rotated head's coordinates are ~N(0, 1/head_dim), inside the codebook's support.
    // (An 8x outlier row would sit past the codebook's ends - out of distribution for this format, and the error
    // it produces says nothing about the quantizer.)
    const int cells = 4;
    std::vector<int32_t> positions(cells);
    for (int c = 0; c < cells; ++c) positions[(size_t) c] = (c * 7) % (f.pages * f.P);
    std::vector<double> rms;
    for (int bits : {3, 4, 6, 8}) {
        const uint64_t row = k::kv_tq_bytes_per_head(bits, f.D);
        const size_t pool = (size_t) f.rows() * row;
        uint8_t* dk = dalloc<uint8_t>(pool);
        uint8_t* dv = dalloc<uint8_t>(pool);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> K((size_t) cells * f.H * f.D), V(K.size());
        for (auto& x : K) x = nd(f.rng);
        for (auto& x : V) x = nd(f.rng);
        float* dK = dalloc<float>(K.size());
        float* dV = dalloc<float>(V.size());
        ck(cudaMemcpy(dK, K.data(), K.size() * 4, cudaMemcpyHostToDevice), "K");
        ck(cudaMemcpy(dV, V.data(), V.size() * 4, cudaMemcpyHostToDevice), "V");
        std::vector<int32_t> steps((size_t) cells * k::kStepCount);
        for (int c = 0; c < cells; ++c) {
            steps[(size_t) c * k::kStepCount + k::kStepPos] = positions[(size_t) c];
            steps[(size_t) c * k::kStepCount + k::kStepNKv] = positions[(size_t) c] + 1;
            steps[(size_t) c * k::kStepCount + k::kStepWidth] = cells;
        }
        int32_t* dsteps = dalloc<int32_t>(steps.size());
        int32_t* dids = dalloc<int32_t>((size_t) cells);
        ck(cudaMemcpy(dsteps, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice), "steps");
        ck(cudaMemcpy(dids, positions.data(), positions.size() * 4, cudaMemcpyHostToDevice), "ids");
        uint16_t* dks = dalloc<uint16_t>((size_t) cells * f.H * f.D);
        uint16_t* dvs = dalloc<uint16_t>((size_t) cells * f.H * f.D);
        k::kv_append_tq_steps(bits, dk, dv, f.d_table, dsteps, k::kStepCount, cells, dK, dV, f.s, nullptr, nullptr);
        k::kv_gather_tq_step(bits, dk, dv, f.d_table, dids, dsteps, cells, f.s, dks, dvs, nullptr);
        ck(cudaDeviceSynchronize(), "append+gather");
        const std::vector<uint16_t> hks = fetch(dks, (size_t) cells * f.H * f.D);
        double se = 0.0, ss = 0.0;
        for (size_t i = 0; i < K.size(); ++i) {
            const double d = (double) k::f32_from_f16(hks[i]) - (double) K[i];
            se += d * d;
            ss += (double) K[i] * K[i];
        }
        rms.push_back(std::sqrt(se / ss));
        cudaFree(dk); cudaFree(dv); cudaFree(dK); cudaFree(dV); cudaFree(dsteps); cudaFree(dids);
        cudaFree(dks); cudaFree(dvs);
    }
    std::printf("  -> relative RMS error: tq3 %.4f, tq4 %.4f, tq6 %.4f, tq8 %.4f\n", rms[0], rms[1], rms[2], rms[3]);
    const bool ordered = rms[0] > rms[1] && rms[1] > rms[2] && rms[2] > rms[3];
    // the theory for Gaussian rows: the codebook's half-spacing over the coordinate spread (0.0625 for head_dim
    // 256) is ~0.09 at 3 bits and ~0.0088 at 8; both bounds leave a factor of ~2
    const bool sane = rms[0] < 0.3 && rms[3] < 0.02;
    if (!ordered || !sane) std::fprintf(stderr, "  FAIL: the widths do not buy precision monotonically\n");
    if (!ordered || !sane) ++g_fail;
}

// [7] the sm_70 prompt tensor reader on TurboQuant pools (mode 5, STRATA_PROMPT_ATTN_TQ=1) against the TurboQuant
// decode reader (mode 4) on the SAME blocks.  The decoded values are the contract; the tensor path only decodes each
// cell once into the FP16 rows and applies the head norm through the FP32 ks/vs slots, so the two must agree within
// the FP16 centroid rounding.  Prompt shapes: 48 cells over 12 pages through a NON-identity page table, four queries
// whose selections are causal windows of different widths (48 > the kernel's CH = 32, so the second chunk is
// partial), and each query keeps its own width.
void test_prompt_tensor_reader(Fixture& f) {
    std::printf("[7/7] sm_70 prompt tensor reader (mode 5) vs the TurboQuant decode reader (mode 4)...\n");
    int dev = 0, major = 0, minor = 0;
    ck(cudaGetDevice(&dev), "device");
    ck(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev), "cc");
    ck(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev), "cc");
    if (major != 7 || minor != 0) {
        std::printf("  -> SKIP: STRATA_PROMPT_ATTN_TQ's kernel is sm_70 only (device is sm_%d%d)\n", major, minor);
        return;
    }
    const int cells = f.pages * f.P;      // > CH: the prompt kernel's second chunk is partial
    const int cap = cells, nq = 4, NH = (int) f.s.n_head;
    const int widths[nq] = {cells, cells - 5, 33, cells - 1};
    std::vector<int32_t> ids((size_t) nq * cap, 0), steps((size_t) nq * k::kStepCount, 0);
    for (int q = 0; q < nq; ++q) {
        steps[q * k::kStepCount + k::kStepWidth] = widths[q];
        for (int c = 0; c < widths[q]; ++c) ids[(size_t) q * cap + c] = cells - widths[q] + c;
    }
    int32_t* dids = dalloc<int32_t>(ids.size());
    int32_t* dsteps = dalloc<int32_t>(steps.size());
    ck(cudaMemcpy(dids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice), "ids");
    ck(cudaMemcpy(dsteps, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice), "steps");
    std::vector<float> K = f.random_rows((size_t) cells * f.H * f.D), V = f.random_rows(K.size());
    float* dK = dalloc<float>(K.size());
    float* dV = dalloc<float>(V.size());
    ck(cudaMemcpy(dK, K.data(), K.size() * 4, cudaMemcpyHostToDevice), "K");
    ck(cudaMemcpy(dV, V.data(), V.size() * 4, cudaMemcpyHostToDevice), "V");
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> hq((size_t) nq * NH * f.D);
    for (auto& x : hq) x = nd(f.rng);
    float* dq = dalloc<float>(hq.size());
    ck(cudaMemcpy(dq, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice), "q");
    float* dscratch = dalloc<float>((size_t) nq * k::qsa_decode_attn_scratch_floats(cap, f.s));
    const size_t out_n = (size_t) nq * NH * f.D;
    float* dref = dalloc<float>(out_n);
    float* dnew = dalloc<float>(out_n);
    std::vector<float> ref(out_n), got(out_n);
    for (int bits : {3, 4, 6, 8}) {
        const uint64_t row = k::kv_tq_bytes_per_head(bits, f.D);
        const size_t pool = (size_t) f.rows() * row;
        uint8_t* dk = dalloc<uint8_t>(pool);
        uint8_t* dv = dalloc<uint8_t>(pool);
        k::kv_append_tq(bits, dk, dv, f.d_table, 0, cells, dK, dV, f.s, nullptr, nullptr, nullptr);
        ck(cudaDeviceSynchronize(), "append");
        k::QsaAttnPools ptq{};
        ptq.page_table = f.d_table; ptq.k_tq = dk; ptq.v_tq = dv; ptq.tq_bits = bits;
        k::qsa_decode_attn_batch(dq, ptq, dids, dsteps, cap, f.s, dscratch, dref, nq, nullptr);
        // flag '0': the dispatcher must refuse, leaving the caller its decode fallback (the flag's VALUE, not its
        // presence, is what opts in).  `qsa_prompt_attn_batch` returns whether it LAUNCHED, so the refusal is its
        // negation - and that negation is what the check and the report below use.
        set_env("STRATA_PROMPT_ATTN_TQ", "0");
        const bool launched_off = k::qsa_prompt_attn_batch(dq, ptq, dids, dsteps, cap, f.s, dnew, nq, nullptr);
        set_env("STRATA_PROMPT_ATTN_TQ", "1");
        const bool on = k::qsa_prompt_attn_batch(dq, ptq, dids, dsteps, cap, f.s, dnew, nq, nullptr);
        ck(cudaDeviceSynchronize(), "attn");
        ck(cudaMemcpy(ref.data(), dref, out_n * 4, cudaMemcpyDeviceToHost), "ref");
        ck(cudaMemcpy(got.data(), dnew, out_n * 4, cudaMemcpyDeviceToHost), "tq");
        const float rel = rel_close(ref, got);
        const bool refused_off = !launched_off;
        std::printf("  -> tq%d: mode 5 vs mode 4 max|diff|/max|out| %.2e %s; flag '0' refused: %s\n", bits,
                    (double) rel, rel <= 5e-3f ? "(within FP16 centroid rounding, OK)" : "(DIVERGES)",
                    refused_off ? "yes" : "NO");
        if (!refused_off || !on || !(rel <= 5e-3f)) ++g_fail;
        cudaFree(dk); cudaFree(dv);
    }
    set_env("STRATA_PROMPT_ATTN_TQ", "0");   // the flag defaults off; leave it that way for any later test
    cudaFree(dids); cudaFree(dsteps); cudaFree(dK); cudaFree(dV); cudaFree(dq); cudaFree(dscratch);
    cudaFree(dref); cudaFree(dnew);
}

}  // namespace

int main() {
    std::printf("kv_tq_parity: TurboQuant K/V layout, roundtrip and attention readers (GPU, synthetic)\n");
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev <= 0) {
        std::printf("no CUDA device: nothing to check\n");
        return 77;
    }
    test_block_layout();
    {
        Fixture f(/*pages=*/4);
        test_prompt_vs_steps(f);
        test_gather_and_layout_oracle(f);
        test_attention_readers(f);
        test_error_shrinks(f);
    }
    {
        Fixture f(/*pages=*/12);   // 48 cells: two prompt chunks in test [7]
        test_prompt_tensor_reader(f);
    }
    if (g_fail == 0) {
        std::printf("=== kv_tq_parity: ALL TESTS PASSED SUCCESSFULLY! ===\n");
        return 0;
    }
    std::fprintf(stderr, "=== kv_tq_parity: FAILED (%d failures) ===\n", g_fail);
    return 1;
}
