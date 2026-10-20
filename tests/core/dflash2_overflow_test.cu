// tests/core/dflash2_overflow_test.cu - the fp16-output-overflow regression for the native DFlash2 backend.
//
// The real checkpoint's residual reaches RMS ~4e4 and its sublayer outputs are larger still, so a
// backend that keeps ACTIVATIONS (or a projection's OUTPUT) in fp16 turns the draft rows into inf/NaN
// while the reference - fp16 matmul inputs, fp32 accumulation, fp32 outputs - stays finite.  That is
// a silent, consumer-visible failure: the proposal's q goes NaN and the draft is useless.
//
// This test builds a fixture whose numbers make that edge exact and checkable, with no model download:
//   hidden 32, 1 layer of "real" work and 4 identity layers, 1 q / 1 kv head x 32, vocab 32, rank 256,
//   top_k 16, one context tap over 2 hyper-connection streams, block 2 (1 slot), 2 context rows.
//   fc = I, every norm = ones, the convs are the identity (base tap0 = 1, tap1 = 0, projection = 0),
//   layer 0's v_proj = o_proj = 1000*I with q_proj = k_proj = 0 (uniform, non-causal attention), and
//   every other attention/MLP weight is 0.
// v and the whole step reduces to
//   attn = 1000 * [1..32]/rms  ->  o_proj  ->  1000*attn = 1e6 * [1..32]/rms   (above the FP16 limit)
//   res  = [1..32] + that                                                   (fp16 saturates)
//   out  = rmsnorm(res) = [1..32]/rms                                       (exact, finite)
// The test checks the direction and scale of the draft row. FP16 saturation produces inf/NaN.
// Clamping changes the channel ratios and fails the 1e-3 tolerance.
// Needs a CUDA device (exits 77 without one) and writes its fixture under the system temp directory.
#include "strata/core/dflash2.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <chrono>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kH = 32;        // draft hidden
constexpr int kTH = 32;       // target hidden per hyper-connection stream
constexpr int kHC = 2;        // hyper-connection streams
constexpr int kHeads = 1;     // q heads (head_dim 32)
constexpr int kKv = 1;
constexpr int kHd = 32;
constexpr int kInter = 32;
constexpr int kVocab = 32;
constexpr int kRank = 256;
constexpr int kTopK = 16;
constexpr int kTaps = 1;
constexpr int kCtaps = 2;
constexpr int kGroup = 16;
constexpr int kGroups = kH / kGroup;
constexpr int kLayers = 5;
constexpr int kBlock = 2;     // -> one slot
constexpr int kCtx = 2;       // context rows appended

/// bf16 bits of `f` (the fixture uses only values that are exact in bf16).
uint16_t bf16(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof u);
    return (uint16_t) (u >> 16);
}

struct Tensor {
    std::string name;
    std::vector<int64_t> dims;
    std::vector<uint16_t> data;
};

void add(std::vector<Tensor>& ts, const char* name, std::vector<int64_t> dims, const std::vector<float>& v) {
    Tensor t;
    t.name = name;
    t.dims = std::move(dims);
    t.data.reserve(v.size());
    for (float f : v) t.data.push_back(bf16(f));
    ts.push_back(std::move(t));
}

void add_zeros(std::vector<Tensor>& ts, const std::string& name, std::vector<int64_t> dims) {
    int64_t n = 1;
    for (int64_t d : dims) n *= d;
    add(ts, name.c_str(), std::move(dims), std::vector<float>((size_t) n, 0.0f));
}

void add_ones(std::vector<Tensor>& ts, const std::string& name, int64_t n) {
    add(ts, name.c_str(), {n}, std::vector<float>((size_t) n, 1.0f));
}

/// Identity [n, n].
void add_identity(std::vector<Tensor>& ts, const std::string& name, int n, float scale) {
    std::vector<float> v((size_t) n * n, 0.0f);
    for (int i = 0; i < n; ++i) v[(size_t) i * n + i] = scale;
    add(ts, name.c_str(), {n, n}, v);
}

bool write_file(const std::filesystem::path& path, const std::string& bytes, std::string& err) {
    std::FILE* f = std::fopen(path.string().c_str(), "wb");
    if (f == nullptr) {
        err = "cannot create " + path.string();
        return false;
    }
    const size_t n = std::fwrite(bytes.data(), 1, bytes.size(), f);
    std::fclose(f);
    if (n != bytes.size()) {
        err = "short write to " + path.string();
        return false;
    }
    return true;
}

/// A minimal safetensors file (BF16, contiguous) plus the config.json the loader validates.
bool write_fixture(const std::filesystem::path& dir, std::string& err) {
    std::vector<Tensor> ts;
    add_identity(ts, "fc.weight", kH, 1.0f);
    add_ones(ts, "hidden_norm.weight", kH);
    add_ones(ts, "norm.weight", kH);
    add_zeros(ts, "candidate_selector.hidden_projection.weight", {kRank, kH});
    add_zeros(ts, "candidate_selector.predecessor_codebook", {kVocab, kRank});
    add_zeros(ts, "candidate_selector.successor_codebook", {kVocab, kRank});
    for (int li = 0; li < kLayers; ++li) {
        const std::string p = "layers." + std::to_string(li) + ".";
        add_ones(ts, p + "input_layernorm.weight", kH);
        add_ones(ts, p + "post_attention_layernorm.weight", kH);
        add_ones(ts, p + "self_attn.q_norm.weight", kHd);
        add_ones(ts, p + "self_attn.k_norm.weight", kHd);
        add_zeros(ts, p + "self_attn.q_proj.weight", {kHeads * kHd, kH});
        add_zeros(ts, p + "self_attn.k_proj.weight", {kKv * kHd, kH});
        // the one layer that does the work: a uniform attention whose value is 1000x the input
        add_identity(ts, p + "self_attn.v_proj.weight", kH, li == 0 ? 1000.0f : 0.0f);
        add_identity(ts, p + "self_attn.o_proj.weight", kH, li == 0 ? 1000.0f : 0.0f);
        add_zeros(ts, p + "mlp.gate_proj.weight", {kInter, kH});
        add_zeros(ts, p + "mlp.up_proj.weight", {kInter, kH});
        add_zeros(ts, p + "mlp.down_proj.weight", {kH, kInter});
        for (const char* conv : {"attention_conv", "mlp_conv"}) {
            const std::string c = p + conv + ".";
            std::vector<float> base((size_t) 2 * kCtaps * kH, 0.0f);   // [side, tap, channel]
            for (int side = 0; side < 2; ++side)
                for (int ch = 0; ch < kH; ++ch) base[(size_t) side * kCtaps * kH + ch] = 1.0f;   // tap 0
            add(ts, (c + "base_kernel").c_str(), {2, kCtaps, kH}, base);
            add_zeros(ts, c + "kernel_projection.weight", {2 * kCtaps * kGroups, kH});
        }
    }

    std::string hdr = "{";
    std::string blob;
    uint64_t off = 0;
    for (size_t i = 0; i < ts.size(); ++i) {
        const Tensor& t = ts[i];
        const uint64_t bytes = (uint64_t) t.data.size() * 2;
        if (i != 0) hdr += ",";
        hdr += "\"" + t.name + "\":{\"dtype\":\"BF16\",\"shape\":[";
        for (size_t d = 0; d < t.dims.size(); ++d) hdr += (d ? "," : "") + std::to_string(t.dims[d]);
        hdr += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + bytes) + "]}";
        off += bytes;
        blob.append(reinterpret_cast<const char*>(t.data.data()), (size_t) bytes);
    }
    hdr += "}";

    std::string file;
    const uint64_t n = hdr.size();
    file.append(reinterpret_cast<const char*>(&n), 8);
    file += hdr;
    file += blob;
    if (!write_file(dir / "model.safetensors", file, err)) return false;

    const std::string cfg =
        "{\"architectures\":[\"DFlash2DraftModel\"],\"attention_bias\":false,"
        "\"dflash_config\":{\"block_size\":" + std::to_string(kBlock) +
        ",\"conv_group_size\":" + std::to_string(kGroup) + ",\"conv_kernel_size\":" + std::to_string(kCtaps) +
        ",\"mask_token_id\":1,\"selector_rank\":" + std::to_string(kRank) +
        ",\"selector_top_k\":" + std::to_string(kTopK) + ",\"target_layer_ids\":[1]},"
        "\"dtype\":\"bfloat16\",\"head_dim\":" + std::to_string(kHd) + ",\"hidden_act\":\"silu\","
        "\"hidden_size\":" + std::to_string(kH) + ",\"intermediate_size\":" + std::to_string(kInter) +
        ",\"is_causal\":false,\"layer_types\":[\"sliding_attention\",\"sliding_attention\","
        "\"sliding_attention\",\"sliding_attention\",\"sliding_attention\"],\"model_type\":\"qwen3\","
        "\"num_attention_heads\":" + std::to_string(kHeads) + ",\"num_hidden_layers\":" +
        std::to_string(kLayers) + ",\"num_key_value_heads\":" + std::to_string(kKv) + ","
        "\"num_target_layers\":48,\"rms_norm_eps\":1e-06,"
        "\"rope_parameters\":{\"rope_theta\":10000,\"rope_type\":\"default\"},"
        "\"sliding_window\":2048,\"use_sliding_window\":true,\"vocab_size\":" + std::to_string(kVocab) + "}";
    return write_file(dir / "config.json", cfg, err);
}

/// The fixture's target head: logits[r][v] = (v * 13 + r) % 32 - 32 distinct values per row, so the
/// top-16 is unambiguous and recomputable here (nothing about the proposal is echoed from the backend).
float head_logit(int r, int v) { return (float) ((v * 13 + r) % kVocab); }

bool head_fn(void* ctx, const float* rows, int n_rows, float* logits, void* stream, std::string& err) {
    (void) ctx;
    (void) rows;
    std::vector<float> lg((size_t) n_rows * kVocab);
    for (int r = 0; r < n_rows; ++r)
        for (int v = 0; v < kVocab; ++v) lg[(size_t) r * kVocab + v] = head_logit(r, v);
    cudaError_t e = cudaMemcpyAsync(logits, lg.data(), lg.size() * sizeof(float), cudaMemcpyHostToDevice,
                                    (cudaStream_t) stream);
    if (e != cudaSuccess) {
        err = cudaGetErrorString(e);
        return false;
    }
    return cudaStreamSynchronize((cudaStream_t) stream) == cudaSuccess;
}
void expected_topk(std::vector<int>& ids, std::vector<float>& vals) {
    std::vector<int> order(kVocab);
    for (int v = 0; v < kVocab; ++v) order[(size_t) v] = v;
    for (int i = 0; i < kVocab; ++i)
        for (int j = i + 1; j < kVocab; ++j) {
            const float a = head_logit(0, order[(size_t) i]), b = head_logit(0, order[(size_t) j]);
            if (b > a || (b == a && order[(size_t) j] < order[(size_t) i]))
                std::swap(order[(size_t) i], order[(size_t) j]);
        }
    ids.assign(order.begin(), order.begin() + kTopK);
    vals.clear();
    for (int id : ids) vals.push_back(head_logit(0, id));
}

}  // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::puts("no CUDA device: skipped");
        return 77;
    }

    // The fixture lives only for this run (RAII: removed on every exit path).
    struct TempDir {
        std::filesystem::path path;
        ~TempDir() {
            std::error_code ec;
            std::filesystem::remove_all(path, ec);
        }
    } tmp;
    tmp.path = std::filesystem::temp_directory_path() /
               ("strata_dflash2_overflow_" +
                std::to_string((unsigned long long) std::chrono::steady_clock::now().time_since_epoch().count()));
    std::error_code ec;
    std::filesystem::create_directories(tmp.path, ec);
    std::string err;
    if (!write_fixture(tmp.path, err)) {
        std::fprintf(stderr, "fixture: %s\n", err.c_str());
        return 1;
    }

    strata::core::DFlash2Config cfg;
    cfg.model_dir = tmp.path.string();
    cfg.device = 0;
    cfg.block_size = kBlock;
    cfg.n_taps = kTaps;
    cfg.tap_layers = {1};
    cfg.target_hidden = kTH;
    cfg.hc_streams = kHC;
    cfg.draft_hidden = kH;
    cfg.vocab = kVocab;
    cfg.max_context = 2047;
    cfg.sliding_window = 2048;
    cfg.top_k = kTopK;

    strata::core::DFlash2Backend b;
    if (!b.load(cfg, err)) {
        std::fprintf(stderr, "load: %s\n", err.c_str());
        return 1;
    }
    b.set_target_head(head_fn, nullptr);

    // block embeddings [B, H] and context features [taps][rows][hc * target_hidden]: [1..32] everywhere
    const int ctx_rows = kCtx;
    std::vector<float> emb((size_t) kBlock * kH);
    for (int r = 0; r < kBlock; ++r)
        for (int i = 0; i < kH; ++i) emb[(size_t) r * kH + i] = (float) (i + 1);
    std::vector<float> feats((size_t) kTaps * ctx_rows * kHC * kTH);
    for (size_t i = 0; i < feats.size(); ++i) feats[i] = (float) ((int) (i % kTH) + 1);

    float* d_emb = nullptr;
    float* d_feats = nullptr;
    if (cudaMalloc(&d_emb, emb.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_feats, feats.size() * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed\n");
        return 1;
    }
    cudaMemcpy(d_emb, emb.data(), emb.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_feats, feats.data(), feats.size() * sizeof(float), cudaMemcpyHostToDevice);

    int rc = 0;
    if (!b.append_features(d_feats, ctx_rows, 0, 0, err)) {
        std::fprintf(stderr, "append_features: %s\n", err.c_str());
        rc = 1;
    }

    strata::core::DFlash2Proposal p;
    if (rc == 0 && !b.propose(0, ctx_rows, d_emb, 1.0f, /*greedy=*/true, nullptr, p, err)) {
        std::fprintf(stderr, "propose: %s\n", err.c_str());
        rc = 1;
    }

    // ---- the draft row: exactly [1..32]/rms, finite.  fp16 saturation gives inf/NaN here, and a
    // clamped (or otherwise scaled) result misses the ratio by far more than the tolerance.
    if (rc == 0) {
        double sq = 0.0;
        for (int i = 1; i <= kH; ++i) sq += (double) i * i;
        const double rms = std::sqrt(sq / kH);
        std::vector<float> ph((size_t) b.slots() * kH);
        if (b.slots() != 1 || !b.copy_pred_hidden(ph.data(), err)) {
            std::fprintf(stderr, "copy_pred_hidden: %s\n", err.c_str());
            rc = 1;
        } else {
            double worst = 0.0;
            int worst_i = -1;
            bool finite = true;
            for (int i = 0; i < kH; ++i) {
                const double want = (double) (i + 1) / rms;
                const double got = ph[(size_t) i];
                if (!std::isfinite(got)) {   // fp16 saturation: inf/NaN, never a small difference
                    finite = false;
                    worst_i = i;
                    break;
                }
                const double d = std::fabs(got - want);
                if (d > worst) {
                    worst = d;
                    worst_i = i;
                }
            }
            if (!finite || worst > 1e-3) {
                std::fprintf(stderr,
                             "draft row is not [1..%d]/rms: %s at %d (got %.6g, want %.6g)\n", kH,
                             finite ? "worst |diff|" : "non-finite", worst_i,
                             worst_i >= 0 ? (double) ph[(size_t) worst_i] : 0.0,
                             worst_i >= 0 ? (double) (worst_i + 1) / rms : 0.0);
                rc = 1;
            }
        }
    }

    // ---- the proposal is the head's own top-16 (recomputed here), with q the selector's conditional
    // distribution: hidden_projection is 0 in this fixture, so score == unary and q == softmax(unary).
    if (rc == 0) {
        std::vector<int> want_ids;
        std::vector<float> want_vals;
        expected_topk(want_ids, want_vals);
        if (p.n_slots != 1 || p.n_cand != kTopK) {
            std::fprintf(stderr, "proposal shape %d slots x %d cand\n", p.n_slots, p.n_cand);
            rc = 1;
        }
        for (int c = 0; c < p.n_cand && rc == 0; ++c) {
            if (p.cand[c] != want_ids[(size_t) c] || !std::isfinite(p.unary[c]) ||
                std::fabs(p.unary[c] - want_vals[(size_t) c]) > 1e-5f) {
                std::fprintf(stderr, "candidate %d: got id %d val %g, want id %d val %g\n", c, p.cand[c],
                             (double) p.unary[c], want_ids[(size_t) c], (double) want_vals[(size_t) c]);
                rc = 1;
            }
        }
        // greedy: a point mass on the argmax of the row
        int best = 0;
        for (int c = 1; c < p.n_cand; ++c)
            if (p.unary[c] > p.unary[best]) best = c;
        if (rc == 0) {
            for (int c = 0; c < p.n_cand; ++c)
                if (!std::isfinite(p.q[c]) || std::fabs(p.q[c] - (c == best ? 1.0f : 0.0f)) > 1e-6f) {
                    std::fprintf(stderr, "greedy q[%d] = %g, expected a point mass at %d\n", c,
                                 (double) p.q[c], best);
                    rc = 1;
                }
            if (p.tokens[0] != p.cand[best]) {
                std::fprintf(stderr, "greedy token %d is not cand[%d] = %d\n", p.tokens[0], best, p.cand[best]);
                rc = 1;
            }
        }
        // sampling: q must be the softmax of the row (temperature 1), and the token the drawn index
        if (rc == 0) {
            const float u = 0.37f;
            strata::core::DFlash2Proposal s;
            if (!b.propose(0, ctx_rows, d_emb, 1.0f, /*greedy=*/false, &u, s, err)) {
                std::fprintf(stderr, "propose (sampling): %s\n", err.c_str());
                rc = 1;
            } else {
                float sum = 0.0f;
                for (int c = 0; c < s.n_cand; ++c) sum += std::exp(want_vals[(size_t) c] - want_vals[0]);
                int drawn = s.n_cand - 1;
                float cum = 0.0f;
                for (int c = 0; c < s.n_cand; ++c) {
                    const float q = std::exp(want_vals[(size_t) c] - want_vals[0]) / sum;
                    if (!std::isfinite(s.q[c]) || std::fabs(s.q[c] - q) > 1e-5f) {
                        std::fprintf(stderr, "sampling q[%d] = %g, expected softmax %g\n", c, (double) s.q[c],
                                     (double) q);
                        rc = 1;
                    }
                    cum += q;
                    if (drawn == s.n_cand - 1 && u < cum) drawn = c;
                }
                if (s.tokens[0] != s.cand[drawn]) {
                    std::fprintf(stderr, "sampling token %d is not cand[%d] = %d\n", s.tokens[0], drawn,
                                 s.cand[drawn]);
                    rc = 1;
                }
            }
        }
    }


    cudaFree(d_emb);
    cudaFree(d_feats);
    if (rc == 0) std::puts("dflash2 overflow regression: ok");
    return rc;
}
