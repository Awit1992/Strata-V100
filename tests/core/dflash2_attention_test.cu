// tests/core/dflash2_attention_test.cu - the output race in the DFlash2 block attention
// (attn_kernel, src/core/dflash2.cu).
//
// The kernel splits the keys over the block's eight warps for the query-key dot products, and the
// softmax statistics are block-wide through shared memory.  A value pass that keeps that split and
// lets every warp write its OWN partial mixture to `out` has no single right answer: the warps
// store to the same elements, so one warp's partial survives - the row becomes the mixture of the
// keys one warp happened to visit, whichever warp that was.  Nothing crashes and nothing is NaN, so
// the consumer only sees a draft whose acceptance rate quietly collapses.  The kernel as written has
// one writer per output dimension: the cooperative QK pass fills `scores` in shared memory, and the
// thread that owns an output dimension sums it over ALL keys.  This test pins that.
//
// It reaches the private kernel by including the implementation.  There is no public testing API for
// the attention alone, and routing through DFlash2Backend::propose would need a checkpoint plus the
// whole GEMM chain to isolate one attention row.  The launch is the backend's own: 256 threads, grid
// (block rows x q heads), the same dynamic shared memory, the same argument order.
//
// What makes a partial-sum regression fail here instead of passing by luck:
//   - the reference is a CPU double softmax over every key, written from its definition, never from
//     the kernel;
//   - the ring's unread slots hold NaN, so touching a slot outside the `ctx` rows shows up as a
//     non-finite output rather than as a plausible number;
//   - the output buffer is a guard band, the region the kernel is given, then another guard band, so
//     a write past the row block is caught before the numbers are read;
//   - the cases cover an empty ring (ctx 0) and a full one (2047), head_dim 8/16/128/256, a ring
//     that wraps, the 4:1 GQA share at 1, 2 and 8 kv heads, and block positions whose delta to the
//     oldest keys is exactly 2048: the sliding window is strict (|delta| < 2048), so both sides of
//     that bound are checked and not assumed;
//   - two cases give every query and block-K/V row a PITCH wider than its head blocks - the layout the
//     backend's fused QKV GEMM produces - and leave the slack NaN, so a kernel that ignores the stride
//     it is handed reads NaN (or another row) instead of the numbers the reference uses.
// The measured error is ~1e-6 relative against the 5e-5 / 1e-4 bounds below.
// Needs a CUDA device (exits 77 without one).
#include "../../src/core/dflash2.cu"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr int kRingCap = 2048;    ///< the context K/V ring's rows
constexpr int kWindow = 2048;     ///< a key is visible iff |q_pos - k_pos| < kWindow
constexpr int kGuard = 16;        ///< untouched floats before and after the kernel's output
constexpr float kGuardValue = 1234567.0f;
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr double kMaxRelL2 = 5e-5;      ///< ||got - ref|| / ||ref||
constexpr double kMaxScaledErr = 1e-4;  ///< max |got - ref| / max |ref|

/// One attention geometry.  The positions are the ones the backend passes: the `ctx` context rows
/// sit at [ctx_pos0, ctx_pos0 + ctx) in ring slots starting at `ring_start`, and the block's own
/// `block` rows at [q_pos0, q_pos0 + block).
struct Case {
    int ctx;             ///< context rows in the ring
    int block;           ///< block rows (the grid's x)
    int head_dim;
    int q_heads;
    int kv_heads;        ///< q_heads / kv_heads is the GQA share
    int ring_start;      ///< the slot the first context row lands in
    int64_t ctx_pos0;
    int64_t q_pos0;
    /// Extra floats per row past the head blocks.  The backend's fused QKV GEMM leaves q, k and v in ONE
    /// buffer, so the kernel reads each row at a pitch wider than a head block; 0 is the packed layout.
    /// Every pitch stays a multiple of 4 (the kernel's float4 head-block loads).
    int slack = 0;
};

/// The geometries under test, as {ctx, block, head_dim, q_heads, kv_heads, ring_start, ctx_pos0,
/// q_pos0, slack}:
///   1  no ring: the keys are the block's own rows
///   2  a 3-row context at slots 2046, 2047, 0 - the ring wraps between two context rows
///   3  a full 2047-row ring that wraps after 3 rows: the newest context key sits 2047 away, one
///      inside the window, and each later row masks its r oldest context keys (delta 2048+)
///   4  the same positions with 8 rows, so up to 7 of the oldest context keys fall outside the
///      window
///   5  a narrow head (16) with one kv head, so all four q heads share the same K/V rows
///   6  the widest head (256): one output dimension per thread
///   7  the narrowest head (8): two float4 slots per row
///   8  the block sits 2048 past the context, so the oldest keys land exactly ON the bound (masked)
///      and one inside it (visible)
///   9  a full ring with a slack of 8: every row of the query and of the block K/V sits at a pitch the
///      packed layout would not produce, so a kernel that ignores the stride it is given reads the
///      slack (NaN) or another row instead of the numbers the reference uses
///  10  the narrow-head case with a slack of 4
constexpr Case kCases[] = {
    {0, 4, 128, 32, 8, 0, 100, 100},
    {3, 8, 128, 32, 8, 2046, 97, 100},
    {2047, 4, 128, 32, 8, 2045, 7953, 10000},
    {2047, 8, 128, 32, 8, 37, 7953, 10000},
    {3, 4, 16, 4, 1, 2046, 97, 100},
    {3, 8, 256, 8, 2, 2046, 97, 100},
    {3, 4, 8, 4, 1, 2046, 97, 100},
    {3, 8, 128, 8, 2, 2046, 12048, 10000},
    {2047, 8, 128, 32, 8, 37, 7953, 10000, 8},
    {3, 4, 16, 4, 1, 2046, 97, 100, 4},
};

/// The row pitch of the fixture's query and block K/V buffers: the head blocks plus `slack` floats.
int q_pitch(const Case& c) { return c.q_heads * c.head_dim + c.slack; }
int kv_pitch(const Case& c) { return c.kv_heads * c.head_dim + c.slack; }

/// The K/V fixture.  Only the scale of the numbers is deliberate: V is 1e5 so a relative tolerance
/// cannot be met by a result that lost the tail of the mixture, and the ring outside the `ctx` rows
/// stays NaN.  So does every row's slack, so a read at the wrong pitch cannot land on a number.
struct Fixture {
    std::vector<float> query;     ///< [block][q_pitch], the head blocks then the slack
    std::vector<float> ring_k;    ///< [ring_cap][kv_heads][head_dim], NaN outside the ctx rows
    std::vector<float> ring_v;    ///< ditto
    std::vector<float> block_k;   ///< [block][kv_pitch]
    std::vector<float> block_v;   ///< ditto
};

Fixture make_fixture(const Case& c) {
    const int qp = q_pitch(c), kvp = kv_pitch(c);
    Fixture f;
    f.query.assign((size_t) c.block * qp, kNaN);
    f.ring_k.assign((size_t) kRingCap * c.kv_heads * c.head_dim, kNaN);
    f.ring_v.assign((size_t) kRingCap * c.kv_heads * c.head_dim, kNaN);
    f.block_k.assign((size_t) c.block * kvp, kNaN);
    f.block_v.assign((size_t) c.block * kvp, kNaN);

    for (int r = 0; r < c.block; ++r)
        for (int h = 0; h < c.q_heads; ++h)
            for (int d = 0; d < c.head_dim; ++d) {
                const size_t i = (size_t) r * qp + (size_t) h * c.head_dim + d;
                f.query[i] = std::sin((float) i * 0.017f) * 0.7f;
            }
    for (int r = 0; r < c.block; ++r)
        for (int h = 0; h < c.kv_heads; ++h)
            for (int d = 0; d < c.head_dim; ++d) {
                const size_t i = (size_t) r * kvp + (size_t) h * c.head_dim + d;
                f.block_k[i] = std::cos((float) i * 0.019f) * 0.9f;
                f.block_v[i] = std::sin((float) i * 0.031f) * 100000.0f;
            }
    // the context rows land in slots (ring_start + t) % cap, which is where the kernel looks
    for (int t = 0; t < c.ctx; ++t)
        for (int h = 0; h < c.kv_heads; ++h)
            for (int d = 0; d < c.head_dim; ++d) {
                const size_t i = ((size_t) ((c.ring_start + t) % kRingCap) * c.kv_heads + h) * c.head_dim + d;
                const float x = (float) (t * 13 + h * 31 + d) * 0.023f;
                f.ring_k[i] = std::cos(x) * 0.9f;
                f.ring_v[i] = std::sin(x + 0.3f) * 100000.0f;
            }
    return f;
}

/// The absolute position of key `t`: the `ctx` context rows first, then the block's own rows.
int64_t key_pos(const Case& c, int t) {
    return t < c.ctx ? c.ctx_pos0 + t : c.q_pos0 + (t - c.ctx);
}

/// Where key `t` of kv head `kv_head` lives: the ring slot (ring_start + t) % cap for a context row,
/// the block's own row t - ctx for one of the block's keys (at the block buffer's row pitch).
const float* key_value_row(const float* ring, const float* block, const Case& c, int t, int kv_head) {
    if (t < c.ctx)
        return ring + ((size_t) ((c.ring_start + t) % kRingCap) * c.kv_heads + kv_head) * c.head_dim;
    return block + (size_t) (t - c.ctx) * kv_pitch(c) + (size_t) kv_head * c.head_dim;
}

bool check_cuda(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string("dflash2 attention test: ") + what + ": " + cudaGetErrorString(e);
    return false;
}

/// A printf-formatted failure message.
template <typename... A>
std::string message(const char* fmt, A... a) {
    char buf[256];
    std::snprintf(buf, sizeof buf, fmt, a...);
    return buf;
}

/// A cudaMalloc'd float buffer that frees itself, so no exit path strands device memory.
struct DeviceBuf {
    float* p = nullptr;
    size_t n = 0;

    DeviceBuf() = default;
    DeviceBuf(const DeviceBuf&) = delete;
    DeviceBuf& operator=(const DeviceBuf&) = delete;
    ~DeviceBuf() { cudaFree(p); }

    bool alloc(size_t count, std::string& err) {
        n = count;
        return check_cuda(cudaMalloc(&p, count * sizeof(float)), "cudaMalloc", err);
    }
    bool upload(const float* src, const char* what, std::string& err) const {
        return check_cuda(cudaMemcpy(p, src, n * sizeof(float), cudaMemcpyHostToDevice), what, err);
    }
    bool download(float* dst, std::string& err) const {
        return check_cuda(cudaMemcpy(dst, p, n * sizeof(float), cudaMemcpyDeviceToHost), "readback", err);
    }
};

/// One case: the fixture to the device, the backend's launch, every output element against the CPU
/// double reference.  False is a failure, already described in `err`.
bool run_case(const Case& c, std::string& err) {
    const int n_keys = c.ctx + c.block;
    const int group = c.q_heads / c.kv_heads;
    const Fixture f = make_fixture(c);
    const size_t n_out = (size_t) c.block * c.q_heads * c.head_dim;

    // guard band | the kernel's output | guard band
    std::vector<float> got(n_out + 2 * (size_t) kGuard, kGuardValue);

    DeviceBuf query, ring_k, ring_v, block_k, block_v, out;
    if (!query.alloc(f.query.size(), err) || !ring_k.alloc(f.ring_k.size(), err) ||
        !ring_v.alloc(f.ring_v.size(), err) || !block_k.alloc(f.block_k.size(), err) ||
        !block_v.alloc(f.block_v.size(), err) || !out.alloc(got.size(), err))
        return false;
    if (!query.upload(f.query.data(), "query", err) || !ring_k.upload(f.ring_k.data(), "ring K", err) ||
        !ring_v.upload(f.ring_v.data(), "ring V", err) || !block_k.upload(f.block_k.data(), "block K", err) ||
        !block_v.upload(f.block_v.data(), "block V", err) || !out.upload(got.data(), "guard bands", err))
        return false;

    // The launch is the backend's: dim3(rows, q heads) x 256 threads, and the shared memory it
    // reserves - n_keys scores, the per-warp reduction scratch, and the tail it always allows for.
    const float scale = 1.0f / std::sqrt((float) c.head_dim);
    const int threads = strata::core::kThreads;
    const size_t shmem =
        ((size_t) n_keys + 32 + (size_t) (threads / c.head_dim) * c.head_dim) * sizeof(float);
    strata::core::attn_kernel<<<dim3((unsigned) c.block, (unsigned) c.q_heads), threads, shmem>>>(
        query.p, ring_k.p, ring_v.p, block_k.p, block_v.p, out.p + kGuard, c.block, c.q_heads, c.kv_heads,
        c.head_dim, q_pitch(c), kv_pitch(c), c.ring_start, kRingCap, c.ctx, c.ctx_pos0, c.q_pos0, kWindow,
        scale);
    if (!check_cuda(cudaGetLastError(), "attn_kernel launch", err) ||
        !check_cuda(cudaDeviceSynchronize(), "attn_kernel", err))
        return false;
    if (!out.download(got.data(), err)) return false;

    // ---- nothing outside the region the kernel was given may move
    for (int i = 0; i < kGuard; ++i)
        if (got[(size_t) i] != kGuardValue || got[(size_t) kGuard + n_out + (size_t) i] != kGuardValue) {
            err = "a write landed outside the output region the kernel was given";
            return false;
        }

    // ---- the reference: softmax(q k^T / sqrt(head_dim)) v over every key the window leaves, in
    // double, from the same host arrays the device was fed
    std::vector<double> weight((size_t) n_keys);
    double sq_err = 0.0, sq_ref = 0.0, max_err = 0.0, max_ref = 0.0;
    for (int row = 0; row < c.block; ++row) {
        for (int h = 0; h < c.q_heads; ++h) {
            const int kv_head = h / group;
            const float* q = f.query.data() + (size_t) row * q_pitch(c) + (size_t) h * c.head_dim;

            double top = -std::numeric_limits<double>::infinity();
            for (int t = 0; t < n_keys; ++t) {
                const int64_t delta = c.q_pos0 + row - key_pos(c, t);
                double s = -std::numeric_limits<double>::infinity();
                if (delta < kWindow && delta > -kWindow) {
                    const float* k = key_value_row(f.ring_k.data(), f.block_k.data(), c, t, kv_head);
                    s = 0.0;
                    for (int d = 0; d < c.head_dim; ++d) s += (double) q[d] * (double) k[d];
                    s *= (double) scale;
                }
                weight[(size_t) t] = s;
                top = std::max(top, s);
            }
            double denom = 0.0;
            for (int t = 0; t < n_keys; ++t) {
                weight[(size_t) t] = std::exp(weight[(size_t) t] - top);
                denom += weight[(size_t) t];
            }

            for (int d = 0; d < c.head_dim; ++d) {
                double ref = 0.0;
                for (int t = 0; t < n_keys; ++t) {
                    const float* v = key_value_row(f.ring_v.data(), f.block_v.data(), c, t, kv_head);
                    ref += (weight[(size_t) t] / denom) * (double) v[d];
                }
                const double y = got[(size_t) kGuard + ((size_t) row * c.q_heads + h) * c.head_dim + d];
                if (!std::isfinite(y)) {
                    err = message("non-finite output at row %d head %d dim %d", row, h, d);
                    return false;
                }
                const double e = y - ref;
                sq_err += e * e;
                sq_ref += ref * ref;
                max_err = std::max(max_err, std::fabs(e));
                max_ref = std::max(max_ref, std::fabs(ref));
            }
        }
    }

    const double rel_l2 = std::sqrt(sq_err / std::max(sq_ref, 1e-30));
    const double max_scaled = max_err / std::max(max_ref, 1e-30);
    std::printf("ctx=%d block=%d head=%d ring=%d slack=%d relL2=%.9g maxScaled=%.9g\n", c.ctx, c.block,
                c.head_dim, c.ring_start, c.slack, rel_l2, max_scaled);
    if (rel_l2 > kMaxRelL2) {
        err = message("relative L2 %.9g > %.9g: the output is not the mixture over every key", rel_l2,
                      kMaxRelL2);
        return false;
    }
    if (max_scaled > kMaxScaledErr) {
        err = message("max scaled error %.9g > %.9g", max_scaled, kMaxScaledErr);
        return false;
    }
    return true;
}

}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("no CUDA device: skipped");
        return 77;
    }

    for (const Case& c : kCases) {
        std::string err;
        if (!run_case(c, err)) {
            std::fprintf(stderr, "ctx=%d block=%d head=%d ring=%d: %s\n", c.ctx, c.block, c.head_dim,
                         c.ring_start, err.c_str());
            return 1;
        }
    }
    std::puts("dflash2 attention regression: ok");
    return 0;
}
