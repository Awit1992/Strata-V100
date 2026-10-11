// src/kernels/router_chain_bench.cpp - what the FFN-side router chain of one layer costs in a captured graph:
// the router GEMV (bf16 [512][2560], T rows), the top-10, the doorbell publish; 48 layers' worth with distinct
// weights (cold), as one graph, per-layer microseconds.  Bench only (no parity: the kernels are the engine's own).
//
// The chain modes call the plain top-10 on a fully resident table, so no rank ever swaps: the STRATA_ROUTE_RESIDENT
// section at the end is separate and times the warp-per-token swap against the serial thread-per-token form it
// replaced, on a deliberately partial table where the swap path is the one exercised.
//
//     build/router_chain_bench [T=2] [reps=20]
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace K = strata::kernels;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); std::exit(1); } } while (0)


namespace {
// The pre-port form of route_resident_k, kept here (and NOT in the library) so this bench can time the serial
// thread-per-token scan against the shipped warp-per-token one on the same input in one run.  The semantics are
// the same for finite logits: a strictly greater logit wins, the smallest expert index breaks a tie, the
// already-picked test runs against the ids as they are at that rank, and the softmax is the same expression.
// They differ on a NaN logit (this form's `l[f] <= bl` skip does not skip NaN; the warp form's `l[f] > bv` does),
// which is outside the router's finite-F32 contract.  route_resident_parity is what proves the warp form against
// the spec on finite inputs; this is only the timing.
__global__ void route_resident_serial_k(const float* __restrict__ logits, int32_t* __restrict__ ids,
                                        float* __restrict__ weights, const int32_t* __restrict__ res, int n_tok,
                                        float margin, int lo, int hi, unsigned long long* __restrict__ stats) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tok) return;
    const float* l = logits + (size_t) t * 512;
    int32_t* id = ids + (size_t) t * 10;
    int swaps = 0, tail = 0, before = 0;
    for (int r = 0; r < 10; ++r) before += res[id[r]] < 0;
    for (int r = lo; r <= hi && r < 10; ++r) {
        const int e = id[r];
        if (res[e] >= 0) continue;
        ++tail;
        int best = -1;
        float bl = -INFINITY;
        for (int f = 0; f < 512; ++f) {
            if (res[f] < 0 || l[f] <= bl) continue;
            bool used = false;
            for (int q = 0; q < 10; ++q) used |= id[q] == f;
            if (!used) { best = f; bl = l[f]; }
        }
        if (best >= 0 && l[e] - bl <= margin) { id[r] = best; ++swaps; }
    }
    int after = 0;
    for (int r = 0; r < 10; ++r) after += res[id[r]] < 0;
    if (swaps) {
        float m = -INFINITY;
        for (int r = 0; r < 10; ++r) m = fmaxf(m, l[id[r]]);
        float ex[10], sum = 0.0f;
        for (int r = 0; r < 10; ++r) { ex[r] = expf(l[id[r]] - m); sum += ex[r]; }
        for (int r = 0; r < 10; ++r) weights[(size_t) t * 10 + r] = ex[r] / sum;
    }
    if (stats) {
        atomicAdd(stats + 0, (unsigned long long) tail);
        atomicAdd(stats + 1, (unsigned long long) swaps);
        atomicAdd(stats + 2, (unsigned long long) before);
        atomicAdd(stats + 3, (unsigned long long) after);
    }
}
}  // namespace
int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const int T = argc > 1 ? std::atoi(argv[1]) : 2;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 20;
    const int L = 48, N = 2560, NE = 512, KK = 10;
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint16_t> w((size_t) NE * N);
    for (auto& x : w) {
        const float f = 0.05f * nd(rng);
        uint32_t u;
        std::memcpy(&u, &f, 4);
        x = (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16);
    }
    std::vector<float> xh((size_t) T * N);
    for (auto& x : xh) x = nd(rng);
    std::vector<uint16_t*> dw(L);
    for (int l = 0; l < L; ++l) {
        CK(cudaMalloc((void**) &dw[l], w.size() * 2));
        CK(cudaMemcpy(dw[l], w.data(), w.size() * 2, cudaMemcpyHostToDevice));
    }
    float *dx, *dlog, *dwt, *dxo;
    int32_t *dids, *dids_o, *dres;
    CK(cudaMalloc((void**) &dx, xh.size() * 4));
    CK(cudaMemcpy(dx, xh.data(), xh.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc((void**) &dlog, (size_t) T * NE * 4));
    CK(cudaMalloc((void**) &dwt, (size_t) T * KK * 4));
    CK(cudaMalloc((void**) &dids, (size_t) T * KK * 4));
    CK(cudaMalloc((void**) &dids_o, (size_t) T * KK * 4));
    CK(cudaMalloc((void**) &dxo, xh.size() * 4));
    CK(cudaMalloc((void**) &dres, NE * 4));
    std::vector<int32_t> res(NE);
    for (int i = 0; i < NE; ++i) res[i] = i;
    CK(cudaMemcpy(dres, res.data(), NE * 4, cudaMemcpyHostToDevice));
    uint32_t* hseq;
    CK(cudaHostAlloc((void**) &hseq, 64, cudaHostAllocMapped | cudaHostAllocPortable));
    *hseq = 0;
    uint32_t* dseq;
    CK(cudaHostGetDevicePointer((void**) &dseq, hseq, 0));
    int* dz;
    CK(cudaMalloc((void**) &dz, 16));
    CK(cudaMemset(dz, 0, 16));
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    {   // outputs of the router GEMV for several shapes and token counts as FNV hashes: compare across STRATA_MMVF_PF=0/1
        const int shapes[][2] = {{2560, 512}, {2560, 256}, {2560, 64}, {1280, 513}, {640, 100}, {4096, 50}, {2048, 7}, {256, 33}, {8192, 90}};
        for (const auto& sh : shapes) {
            const int ni = sh[0], no = sh[1];
            std::vector<uint16_t> wh((size_t) no * ni);
            for (auto& v : wh) {
                const float f = 0.05f * nd(rng);
                uint32_t u;
                std::memcpy(&u, &f, 4);
                v = (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16);
            }
            std::vector<float> xs((size_t) 8 * ni);
            for (auto& v : xs) v = nd(rng);
            uint16_t* dwp;
            float *dxp, *dyp;
            CK(cudaMalloc((void**) &dwp, wh.size() * 2));
            CK(cudaMemcpy(dwp, wh.data(), wh.size() * 2, cudaMemcpyHostToDevice));
            CK(cudaMalloc((void**) &dxp, xs.size() * 4));
            CK(cudaMemcpy(dxp, xs.data(), xs.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMalloc((void**) &dyp, (size_t) 8 * no * 4));
            for (int tk = 1; tk <= 8; ++tk) {
                CK(cudaMemset(dyp, 0xff, (size_t) 8 * no * 4));
                K::bf16_gemv_fp32_mmvf_multi(dxp, ni, dwp, dyp, no, ni, no, tk, s);
                CK(cudaStreamSynchronize(s));
                std::vector<float> yh((size_t) 8 * no);
                CK(cudaMemcpy(yh.data(), dyp, yh.size() * 4, cudaMemcpyDeviceToHost));
                unsigned long long h = 1469598103934665603ull;
                for (size_t b = 0; b < yh.size() * 4; ++b) h = (h ^ ((const unsigned char*) yh.data())[b]) * 1099511628211ull;
                std::printf("PARITY n_in %d n_out %d T %d hash %016llx\n", ni, no, tk, h);
            }
            cudaFree(dwp); cudaFree(dxp); cudaFree(dyp);
        }
    }
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    {   // the BF16 GEMV by shape and token count, 48 distinct weight copies in one graph: us per call (STRATA_MMVF_PF=0 / 1 to compare)
        const int outs[] = {512, 256, 128, 64, 48, 8, 1};
        const int ins[] = {2560, 2560, 2560, 2560, 2560, 2560, 2560};
        for (int si = 0; si < 7; ++si) {
            const int no = outs[si], ni = ins[si];
            std::vector<uint16_t*> dws(L);
            for (int l = 0; l < L; ++l) {
                CK(cudaMalloc((void**) &dws[l], (size_t) no * ni * 2));
                CK(cudaMemcpy(dws[l], w.data(), (size_t) no * ni * 2 < w.size() * 2 ? (size_t) no * ni * 2 : w.size() * 2, cudaMemcpyHostToDevice));
            }
            for (int tk : {1, 2, 3, 4, 6}) {
                cudaGraph_t g;
                cudaGraphExec_t ge;
                CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
                for (int l = 0; l < L; ++l) K::bf16_gemv_fp32_mmvf_multi(dx, N, dws[l], dlog, NE, ni, no, tk, s);
                CK(cudaStreamEndCapture(s, &g));
                CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
                for (int i = 0; i < 5; ++i) CK(cudaGraphLaunch(ge, s));
                CK(cudaStreamSynchronize(s));
                std::vector<float> ms;
                for (int r = 0; r < reps; ++r) {
                    CK(cudaEventRecord(e0, s));
                    CK(cudaGraphLaunch(ge, s));
                    CK(cudaEventRecord(e1, s));
                    CK(cudaEventSynchronize(e1));
                    float m = 0;
                    CK(cudaEventElapsedTime(&m, e0, e1));
                    ms.push_back(m);
                }
                std::sort(ms.begin(), ms.end());
                std::printf("SHAPE n_in %d n_out %3d T %d : %6.2f us per call\n", ni, no, tk, 1e3 * ms[ms.size() / 2] / L);
                cudaGraphExecDestroy(ge);
                cudaGraphDestroy(g);
            }
            for (int l = 0; l < L; ++l) cudaFree(dws[l]);
        }
    }
    // mode bits: 1 mmvf, 2 route, 4 doorbell, 8 an empty kernel per layer (the node floor)
    const char* names[] = {"empty node", "mmvf", "route", "mmvf+route", "doorbell", "mmvf+doorbell", "route+doorbell", "mmvf+route+doorbell"};
    for (int mode = 0; mode < 8; ++mode) {
        cudaGraph_t g;
        cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        for (int l = 0; l < L; ++l) {
            if (mode == 0) K::gpu_stamp((unsigned long long*) dz, 0, s);
            if (mode & 1) K::bf16_gemv_fp32_mmvf_multi(dx, N, dw[l], dlog, NE, N, NE, T, s);
            if (mode & 2) K::native_router_top10_multi(dlog, dids, dwt, T, s);
            if (mode & 4) K::doorbell_publish_res(dx, dids, dres, NE, (int64_t) T * N, (int64_t) T * KK, dxo, dids_o, dseq, s);
        }
        CK(cudaStreamEndCapture(s, &g));
        CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
        for (int i = 0; i < 5; ++i) CK(cudaGraphLaunch(ge, s));
        CK(cudaStreamSynchronize(s));
        std::vector<float> ms;
        for (int r = 0; r < reps; ++r) {
            CK(cudaEventRecord(e0, s));
            CK(cudaGraphLaunch(ge, s));
            CK(cudaEventRecord(e1, s));
            CK(cudaEventSynchronize(e1));
            float m = 0;
            CK(cudaEventElapsedTime(&m, e0, e1));
            ms.push_back(m);
        }
        std::sort(ms.begin(), ms.end());
        std::printf("T %d %-20s : %6.2f us per layer (median of %d graphs of %d layers)\n", T, names[mode], 1e3 * ms[ms.size() / 2] / L, reps, L);
        cudaGraphExecDestroy(ge);
        cudaGraphDestroy(g);
    }
    // STRATA_ROUTE_RESIDENT: the warp-per-token swap against the serial thread-per-token form the port replaced.
    // The chain modes above call the plain top-10 on a fully resident table, so no rank ever swaps; here 40% of
    // the experts are resident and the margin is wide, so every rank in [0, 9] scans all 512 - the swap path is
    // the one measured.  Each form is captured as ONE graph of 48 layers, so the timed span is the graph's own
    // execution rather than 48 host launches (their feed gaps and per-call argument validation are not the
    // kernel's cost).  The ids are restored from dids_src before every timed graph because a swap rewrites them,
    // both graphs are warmed, and the passes alternate which form runs first so clock/thermal drift is shared
    // rather than owned by one side.  T = 1 is the single-token decode case.
    {
        std::vector<float> lg((size_t) T * NE);
        for (auto& v : lg) v = nd(rng);
        std::vector<int32_t> idh((size_t) T * KK);
        for (int t = 0; t < T; ++t) {
            const float* row = lg.data() + (size_t) t * NE;
            int32_t* out = idh.data() + (size_t) t * KK;
            for (int r = 0; r < KK; ++r) {                       // the router's top-10, ties to the smallest index
                float bv = -INFINITY;
                int bi = -1;
                for (int f = 0; f < NE; ++f) {
                    bool used = false;
                    for (int q = 0; q < r; ++q) used |= out[q] == f;
                    if (used) continue;
                    if (row[f] > bv) { bv = row[f]; bi = f; }
                }
                out[r] = bi;
            }
        }
        std::vector<int32_t> resh(NE);
        for (int i = 0; i < NE; ++i) resh[i] = (i % 5) < 2 ? (i % 7) : -1;   // 40% of the experts resident
        std::vector<int32_t> src((size_t) L * T * KK);                       // one fresh id copy per layer
        for (int l = 0; l < L; ++l) std::memcpy(src.data() + (size_t) l * T * KK, idh.data(), (size_t) T * KK * 4);
        int nonresident = 0;                                                 // ids in the window that are not resident
        for (int t = 0; t < T; ++t)
            for (int r = 0; r < KK; ++r) nonresident += resh[idh[(size_t) t * KK + r]] < 0;
        float* dlg;
        float* dwts;
        int32_t* dresh;
        int32_t* dids_rs;
        int32_t* dids_src;
        unsigned long long* dstats;
        CK(cudaMalloc((void**) &dlg, lg.size() * 4));
        CK(cudaMemcpy(dlg, lg.data(), lg.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc((void**) &dwts, (size_t) T * KK * 4));
        CK(cudaMalloc((void**) &dresh, NE * 4));
        CK(cudaMemcpy(dresh, resh.data(), NE * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc((void**) &dids_rs, src.size() * 4));
        CK(cudaMalloc((void**) &dids_src, src.size() * 4));
        CK(cudaMemcpy(dids_src, src.data(), src.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc((void**) &dstats, 4 * sizeof(unsigned long long)));
        const float margin = 4.0f;   // wide: every non-resident rank swaps, so the softmax branch runs in both forms
        // One 48-layer graph per form, each layer on its own id slice of dids_rs, captured once so the measured
        // span is just cudaGraphLaunch.
        auto capture = [&](bool warp) {
            cudaGraph_t g;
            cudaGraphExec_t ge;
            CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
            for (int l = 0; l < L; ++l) {
                int32_t* id = dids_rs + (size_t) l * T * KK;
                if (warp) K::native_route_resident(dlg, id, dwts, dresh, T, margin, 0, KK - 1, nullptr, s);
                else route_resident_serial_k<<<(T + 31) / 32, 32, 0, s>>>(dlg, id, dwts, dresh, T, margin, 0, KK - 1, nullptr);
            }
            CK(cudaStreamEndCapture(s, &g));
            CK(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0));
            cudaGraphDestroy(g);
            return ge;
        };
        cudaGraphExec_t ge_serial = capture(false);
        cudaGraphExec_t ge_warp = capture(true);
        for (int i = 0; i < 5; ++i) {   // warm both graphs: the first launch of an exec pays its own setup
            CK(cudaMemcpyAsync(dids_rs, dids_src, src.size() * 4, cudaMemcpyDeviceToDevice, s));
            CK(cudaGraphLaunch(ge_serial, s));
            CK(cudaMemcpyAsync(dids_rs, dids_src, src.size() * 4, cudaMemcpyDeviceToDevice, s));
            CK(cudaGraphLaunch(ge_warp, s));
        }
        CK(cudaStreamSynchronize(s));
        std::vector<float> ms_serial, ms_warp;
        for (int r = 0; r < reps; ++r) {
            for (int pass = 0; pass < 2; ++pass) {
                const bool warp = (pass == (r & 1));    // even passes: serial first; odd passes: warp first
                cudaGraphExec_t ge = warp ? ge_warp : ge_serial;
                CK(cudaMemcpyAsync(dids_rs, dids_src, src.size() * 4, cudaMemcpyDeviceToDevice, s));   // untimed restore
                CK(cudaEventRecord(e0, s));
                CK(cudaGraphLaunch(ge, s));
                CK(cudaEventRecord(e1, s));
                CK(cudaEventSynchronize(e1));
                float m = 0;
                CK(cudaEventElapsedTime(&m, e0, e1));
                (warp ? ms_warp : ms_serial).push_back(m);
            }
        }
        std::sort(ms_serial.begin(), ms_serial.end());
        std::sort(ms_warp.begin(), ms_warp.end());
        const float us_serial = 1e3 * ms_serial[ms_serial.size() / 2] / L;
        const float us_warp = 1e3 * ms_warp[ms_warp.size() / 2] / L;
        // One probe launch with the counters: proof that the swap path ran, not just the scan.
        CK(cudaMemcpyAsync(dids_rs, dids_src, src.size() * 4, cudaMemcpyDeviceToDevice, s));
        CK(cudaMemsetAsync(dstats, 0, 4 * sizeof(unsigned long long), s));
        K::native_route_resident(dlg, dids_rs, dwts, dresh, T, margin, 0, KK - 1, dstats, s);
        unsigned long long hst[4] = {0, 0, 0, 0};
        CK(cudaMemcpy(hst, dstats, sizeof(hst), cudaMemcpyDeviceToHost));
        std::printf("T %d route_resident: %d of %d ids non-resident, the warp form swapped %llu\n", T, nonresident, T * KK,
                    hst[1]);
        std::printf("T %d route_resident serial : %6.2f us per layer (median of %d graphs of %d layers)\n", T, us_serial, reps, L);
        std::printf("T %d route_resident warp   : %6.2f us per layer (median of %d graphs of %d layers), %.2fx faster\n",
                    T, us_warp, reps, L, us_warp > 0.0f ? us_serial / us_warp : 0.0f);
        cudaGraphExecDestroy(ge_serial);
        cudaGraphExecDestroy(ge_warp);
        cudaFree(dlg); cudaFree(dwts); cudaFree(dresh); cudaFree(dids_rs); cudaFree(dids_src); cudaFree(dstats);
    }
    return 0;
}
