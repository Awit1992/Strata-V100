// src/kernels/cpu_pool_quant_rows_test.cpp - ExpertPool::quant_act_rows, the native prefill stage's row batch.
//
// A native CPU prefill stage quantizes each activation row through the same kernel the submitting thread's serial
// loop would call - `act_quant_any` for a Q2_0 gate/up layer Strata runs itself (`q2_native_kernels`), ggml-cpu's
// `native_quant_act` otherwise - and hands the rows to `quant_act_rows` so the pool spreads them over its
// threads.  The claim is small and exact, so it is checked byte for byte rather than through a timing number:
//
//   1. THE SAME BYTES AS SERIAL.  Every needed row must quantize to exactly what the serial per-row kernel
//      produces, for both activation paths (IQ3_S's native ggml quantization and Q2_0's own kernels), over
//      257 / 17 / 1 / 0 rows (shrinking, empty last), one and four workers, with and without host participation.
//   2. SKIPPED ROWS, PADDING AND EVERYTHING PAST THE BATCH ARE UNTOUCHED.  A row whose `need` flag is clear
//      must not be written, the bytes past a row's activation inside its `kNativeActBytes` slot must not be
//      either, and neither must the guard row after the last one.  Every buffer starts at a sentinel and the
//      whole allocation is compared, so a skipped row, a padding byte or a write past `n_rows` shows up as a
//      difference rather than as a plausible value.
//   3. THE MASK SHAPES THE ENGINE ACTUALLY SEES: dense, empty, the last row only, and every 31st row.
//   4. THE COMPACT FEED THE PREFILL STAGE SENDS.  The stage deduplicates the rows its CPU experts read, gathers
//      them on the device and passes the compact buffer with an all-ones mask, so each token's row must land on
//      exactly the bytes the sparse feed left in that token's own destination.
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {

constexpr int kN = 2560;       // n_embd of the probed layer
constexpr int kMaxRows = 257;  // the largest row count checked
constexpr uint8_t kSentinel = 0xa5;

// One allocation set reused by every case: a row's inputs depend only on (row, column), so the first `rows` rows
// of the max-size buffers are already the right activations for a smaller case.
struct Fixture {
    std::vector<float> x;                   // kMaxRows * kN
    std::vector<char> need;                 // kMaxRows
    std::vector<cpu::ActQ> actq, ref_actq;  // kMaxRows + 1 (one guard row)
    std::vector<uint8_t> nact, ref_nact;    // (kMaxRows + 1) * kNativeActBytes (one guard row)
    // The compact feed's own buffers: the needed rows in ascending token order, the all-ones mask, the token ->
    // compact row map, and the compact destinations.
    std::vector<float> cx;                  // kMaxRows * kN
    std::vector<char> cneed;                // kMaxRows
    std::vector<int32_t> pos;               // kMaxRows: token -> compact row, or -1
    std::vector<cpu::ActQ> cactq;           // kMaxRows + 1 (one guard row)
    std::vector<uint8_t> cnact;             // (kMaxRows + 1) * kNativeActBytes (one guard row)

    Fixture()
        : x((size_t) kMaxRows * kN), need(kMaxRows), actq(kMaxRows + 1), ref_actq(kMaxRows + 1),
          nact((size_t) (kMaxRows + 1) * cpu::kNativeActBytes),
          ref_nact((size_t) (kMaxRows + 1) * cpu::kNativeActBytes),
          cx((size_t) kMaxRows * kN), cneed(kMaxRows), pos(kMaxRows), cactq(kMaxRows + 1),
          cnact((size_t) (kMaxRows + 1) * cpu::kNativeActBytes) {
        for (int t = 0; t < kMaxRows; ++t)
            for (int j = 0; j < kN; ++j) {
                const float v = (float) ((t * 19 + j * 31) % 257 - 128) * 0.015625f;
                // Row 0 is all zero; row 2 is subnormal (a value in [-2, 2] x 1e-39) so the scale search sees
                // the bottom of the float range as well.
                x[(size_t) t * kN + j] = t == 0 ? 0.0f : t == 2 ? v * 1.0e-39f : v;
            }
    }
};

size_t first_diff(const uint8_t* got, const uint8_t* want, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (got[i] != want[i]) return i;
    return n;
}

}  // namespace

int main() {
    Fixture fx;
    int cases = 0, failures = 0;

    for (int host = 0; host < 2; ++host)
        for (int workers : {1, 4}) {
            cpu::ExpertPool pool(workers, true, host != 0);
            for (int gu : {21, 42}) {  // GGML_TYPE_IQ3_S (ggml-cpu activation) and GGML_TYPE_Q2_0 (Strata's kernels)
                cpu::NativeFmt f;
                std::string err;
                if (!cpu::native_fmt(gu, 20 /*GGML_TYPE_IQ4_NL*/, kN, 640, f, err)) {
                    std::fprintf(stderr, "cpu_pool_quant_rows_test: native_fmt(gu=%d): %s\n", gu, err.c_str());
                    return 2;
                }
                const bool q2 = cpu::q2_native_kernels(f.gu_type);
                // Shrinking row counts, empty last: at 1 row "last only" and "every 31st" are the dense mask
                // again, and at 0 rows every mask is the empty one, so those duplicates are not repeated.
                for (int rows : {257, 17, 1, 0}) {
                    const int patterns = rows == 0 ? 1 : rows == 1 ? 2 : 4;
                    for (int pattern = 0; pattern < patterns; ++pattern) {
                        for (int t = 0; t < rows; ++t)
                            fx.need[(size_t) t] = (char) (pattern == 0 || (pattern == 2 && t == rows - 1) ||
                                                          (pattern == 3 && t % 31 == 0));
                        // The whole allocation, guard row included, starts at the sentinel every case.
                        std::memset(fx.actq.data(), kSentinel, fx.actq.size() * sizeof(cpu::ActQ));
                        std::memset(fx.ref_actq.data(), kSentinel, fx.ref_actq.size() * sizeof(cpu::ActQ));
                        std::memset(fx.nact.data(), kSentinel, fx.nact.size());
                        std::memset(fx.ref_nact.data(), kSentinel, fx.ref_nact.size());

                        // The serial reference: the same per-row kernel the submitting thread would call.
                        for (int t = 0; t < rows; ++t) {
                            if (fx.need[(size_t) t] == 0) continue;
                            const float* xr = fx.x.data() + (size_t) t * kN;
                            if (q2)
                                cpu::act_quant_any(xr, kN, fx.ref_actq[(size_t) t]);
                            else
                                cpu::native_quant_act(f, xr, fx.ref_nact.data() + (size_t) t * cpu::kNativeActBytes);
                        }

                        const cpu::ExpertPool::ActRows batch{fx.x.data(), fx.need.data(), rows, kN,
                                                             q2 ? fx.actq.data() : nullptr,
                                                             q2 ? nullptr : fx.nact.data()};
                        pool.quant_act_rows(f, batch);

                        const uint8_t* got = q2 ? (const uint8_t*) fx.actq.data() : fx.nact.data();
                        const uint8_t* want = q2 ? (const uint8_t*) fx.ref_actq.data() : fx.ref_nact.data();
                        const size_t bytes = q2 ? fx.actq.size() * sizeof(cpu::ActQ) : fx.nact.size();
                        const size_t at = first_diff(got, want, bytes);
                        if (at != bytes) {
                            ++failures;
                            const size_t row_bytes = q2 ? sizeof(cpu::ActQ) : cpu::kNativeActBytes;
                            const size_t row = at / row_bytes;
                            const bool in_batch = row < (size_t) rows;
                            std::fprintf(stderr,
                                         "quant_act_rows mismatch: host=%d workers=%d gu=%d rows=%d pattern=%d "
                                         "(%s path) row %zu (%s) byte %zu: got 0x%02x want 0x%02x\n",
                                         host, workers, gu, rows, pattern, q2 ? "Q2_0" : "native", row,
                                         in_batch ? (fx.need[row] ? "need" : "skip") : "past batch",
                                         at % row_bytes, got[at], want[at]);
                        }
                        // 4. The compact feed the prefill stage sends: the needed rows in ascending token order,
                        // an all-ones mask, destinations indexed by the compact row.  Each token's row must land
                        // on exactly the bytes the sparse feed left in that token's own destination.
                        int64_t n_need = 0;
                        for (int t = 0; t < rows; ++t) {
                            if (fx.need[(size_t) t] == 0) { fx.pos[(size_t) t] = -1; continue; }
                            fx.pos[(size_t) t] = (int32_t) n_need;
                            std::memcpy(fx.cx.data() + (size_t) n_need * kN, fx.x.data() + (size_t) t * kN,
                                        (size_t) kN * sizeof(float));
                            fx.cneed[(size_t) n_need] = 1;
                            ++n_need;
                        }
                        std::memset(fx.cactq.data(), kSentinel, fx.cactq.size() * sizeof(cpu::ActQ));
                        std::memset(fx.cnact.data(), kSentinel, fx.cnact.size());
                        const cpu::ExpertPool::ActRows compact{fx.cx.data(), fx.cneed.data(), n_need, kN,
                                                               q2 ? fx.cactq.data() : nullptr,
                                                               q2 ? nullptr : fx.cnact.data()};
                        pool.quant_act_rows(f, compact);
                        const size_t crow_bytes = q2 ? sizeof(cpu::ActQ) : cpu::kNativeActBytes;
                        const uint8_t* cbuf = q2 ? (const uint8_t*) fx.cactq.data() : fx.cnact.data();
                        for (int t = 0; t < rows; ++t) {
                            if (fx.need[(size_t) t] == 0) continue;
                            const size_t c = (size_t) fx.pos[(size_t) t];
                            const uint8_t* a = q2 ? (const uint8_t*) &fx.actq[(size_t) t]
                                                  : fx.nact.data() + (size_t) t * crow_bytes;
                            const size_t at = first_diff(a, cbuf + c * crow_bytes, crow_bytes);
                            if (at != crow_bytes) {
                                ++failures;
                                std::fprintf(stderr,
                                             "compact feed mismatch: host=%d workers=%d gu=%d rows=%d pattern=%d "
                                             "(%s path) token %d -> compact row %zu, byte %zu: sparse 0x%02x compact 0x%02x\n",
                                             host, workers, gu, rows, pattern, q2 ? "Q2_0" : "native", t, c, at,
                                             a[at], cbuf[c * crow_bytes + at]);
                            }
                        }
                        // ... and it writes nothing past its own rows (the guard row included)
                        const size_t cbytes = q2 ? fx.cactq.size() * sizeof(cpu::ActQ) : fx.cnact.size();
                        for (size_t i = (size_t) n_need * crow_bytes; i < cbytes; ++i)
                            if (cbuf[i] != kSentinel) {
                                ++failures;
                                std::fprintf(stderr, "compact feed wrote past its rows: host=%d workers=%d gu=%d "
                                                     "rows=%d pattern=%d byte %zu\n",
                                             host, workers, gu, rows, pattern, i);
                                break;
                            }
                        ++cases;
                    }
                }
            }
        }

    if (failures) {
        std::printf("cpu_pool_quant_rows_test: %d of %d cases differ from the reference\n", failures, cases);
        return 1;
    }
    std::printf("cpu_pool_quant_rows_test: %d cases byte-exact (native/Q2_0, 257/17/1/0 rows, dense/empty/last/"
                "sparse masks, host on/off, 1/4 workers); skipped rows, native padding and the guard row untouched; "
                "the compact feed lands each token's row on the sparse feed's bytes\n",
                cases);
    return 0;
}
