// src/core/dflash2.cu - the native DFlash2 draft backend (see include/strata/core/dflash2.hpp).
//
// Ported from the reference implementation, not from a description of it:
//   sgl-project/sglang @3c560089 python/sglang/srt/models/dflash.py
//     DFlashAttention / DFlashMLP / _grouped_conv / DFlashGroupedConv / DFlashDecoderLayer /
//     DFlashDraftModel.forward / _score_edges / CandidateSelector.sample_path / DFlash2DraftModel
//   sgl-project/sglang @3c560089 python/sglang/srt/speculative/dflash_worker_v2.py
//     _append_target_hidden_sequential (context K/V = kv_proj_only(ctx_hidden) + k_norm + RoPE)
//
// What one step computes (all of it here; verification is the caller's):
//   ctx = hidden_norm(fc(concat_taps(mean_hc(target features))))     positions [start-ctx_len, start)
//   per layer, over the block rows only:
//     h        = input_layernorm(x)                                  (pre-norm, residual carried)
//     pre, co  = attention_conv.prepare(h)     coef = base + delta, delta = kernel_projection(h)
//     q,k,v    = q_proj|k_proj|v_proj(pre);  q,k: per-head RMSNorm then RoPE at the row's position
//     a        = softmax(q k^T * 128^-0.5) v   over [ctx K/V] ++ [block K/V], NON-CAUSAL, sliding window
//     a        = attention_conv.finish(a, co);  h2,res = post_attention_layernorm(a, res)
//     g,u      = gate_proj|up_proj(h2); h3 = down_proj(silu(g)*u)
//     y        = mlp_conv.finish(h3, mlp_conv.prepare(h2).co)
//   out = norm(y + res)[1:]  -> target lm_head -> top-k -> selector lattice -> one path -> tokens + q
//
// Native precision: BF16 checkpoint weights are rounded to FP16 at load time. Tiny values can round
// or underflow; the loader rejects non-finite weights and magnitudes above 65504. cuBLAS uses FP16
// operands with FP32 accumulation and FP32 outputs. Activations, residuals, context K/V, and
// elementwise steps use FP32. This is not bitwise equivalent to the reference's BF16 compute.
// FP16 activation outputs can overflow here: the real checkpoint's residual reaches RMS ~4e4.

#include "strata/core/dflash2.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {
namespace {

constexpr int kThreads = 256;
constexpr int kChunkRows = 256;         ///< rows of target features converted / projected at once
constexpr int kTopKThreads = 128;       ///< top-k scan width (the shared merge table is 128 * KMAX)
constexpr int kTopKMax = kDFlash2MaxTopK;
constexpr int kSelShards = 16;          ///< top-k shards per logits row (contiguous vocabulary spans)
constexpr int kMergeThreads = 512;      ///< top-k merge block: one thread per shard entry (>= kSelShards * kTopKMax)
constexpr int kMergeWarps = kMergeThreads / 32;

// --------------------------------------------------------------------------- small helpers

inline bool cuda_ok(cudaError_t e, const char* what, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string("dflash2: ") + what + ": " + cudaGetErrorString(e);
    return false;
}

inline bool blas_ok(cublasStatus_t s, const char* what, std::string& err) {
    if (s == CUBLAS_STATUS_SUCCESS) return true;
    err = std::string("dflash2: ") + what + ": cuBLAS status " + std::to_string((int) s);
    return false;
}

/// bf16 bits -> fp16 bits, through float (round to nearest even).
inline uint16_t bf16_to_f16(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, sizeof f);
    return (uint16_t) __half_as_ushort(__float2half_rn(f));
}

inline bool env_on(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
}

// --------------------------------------------------------------------------- config.json

/// A tiny JSON reader: objects, arrays, strings, numbers, bools, null - enough for config.json.
struct JVal {
    enum Kind { Nul, Bool, Num, Str, Arr, Obj } kind = Nul;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<JVal> arr;
    std::vector<std::pair<std::string, JVal>> obj;

    const JVal* find(const char* key) const {
        for (const auto& kv : obj)
            if (kv.first == key) return &kv.second;
        return nullptr;
    }
};

struct JParser {
    const char* p = nullptr;
    const char* e = nullptr;
    std::string err;

    void ws() {
        while (p < e && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
    }
    bool fail(const char* what) {
        if (err.empty()) err = std::string("config.json: ") + what;
        return false;
    }
    bool lit(const char* s) {
        const size_t n = std::strlen(s);
        if ((size_t) (e - p) < n || std::strncmp(p, s, n) != 0) return false;
        p += n;
        return true;
    }
    bool str(std::string& out) {
        if (p >= e || *p != '"') return fail("expected a string");
        ++p;
        out.clear();
        while (p < e && *p != '"') {
            if (*p == '\\') {
                ++p;
                if (p >= e) return fail("truncated escape");
                switch (*p) {
                    case 'n': out.push_back('\n'); break;
                    case 't': out.push_back('\t'); break;
                    case 'r': out.push_back('\r'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'u': {
                        if (e - p < 5) return fail("truncated \\u");
                        unsigned cp = 0;
                        for (int i = 1; i <= 4; ++i) {
                            const char c = p[i];
                            cp <<= 4;
                            if (c >= '0' && c <= '9') cp |= (unsigned) (c - '0');
                            else if (c >= 'a' && c <= 'f') cp |= (unsigned) (c - 'a' + 10);
                            else if (c >= 'A' && c <= 'F') cp |= (unsigned) (c - 'A' + 10);
                            else return fail("bad \\u escape");
                        }
                        p += 4;
                        if (cp < 0x80) out.push_back((char) cp);
                        else if (cp < 0x800) { out.push_back((char) (0xC0 | (cp >> 6))); out.push_back((char) (0x80 | (cp & 0x3F))); }
                        else { out.push_back((char) (0xE0 | (cp >> 12))); out.push_back((char) (0x80 | ((cp >> 6) & 0x3F))); out.push_back((char) (0x80 | (cp & 0x3F))); }
                        break;
                    }
                    default: out.push_back(*p); break;
                }
                ++p;
            } else {
                out.push_back(*p++);
            }
        }
        if (p >= e) return fail("unterminated string");
        ++p;
        return true;
    }
    bool value(JVal& v) {
        ws();
        if (p >= e) return fail("unexpected end");
        if (*p == '{') {
            ++p;
            v.kind = JVal::Obj;
            ws();
            if (p < e && *p == '}') { ++p; return true; }
            while (true) {
                ws();
                std::string key;
                if (!str(key)) return false;
                ws();
                if (p >= e || *p != ':') return fail("expected ':'");
                ++p;
                JVal child;
                if (!value(child)) return false;
                v.obj.emplace_back(std::move(key), std::move(child));
                ws();
                if (p < e && *p == ',') { ++p; continue; }
                if (p < e && *p == '}') { ++p; return true; }
                return fail("expected ',' or '}'");
            }
        }
        if (*p == '[') {
            ++p;
            v.kind = JVal::Arr;
            ws();
            if (p < e && *p == ']') { ++p; return true; }
            while (true) {
                JVal child;
                if (!value(child)) return false;
                v.arr.push_back(std::move(child));
                ws();
                if (p < e && *p == ',') { ++p; continue; }
                if (p < e && *p == ']') { ++p; return true; }
                return fail("expected ',' or ']'");
            }
        }
        if (*p == '"') {
            v.kind = JVal::Str;
            return str(v.str);
        }
        if (lit("true")) { v.kind = JVal::Bool; v.b = true; return true; }
        if (lit("false")) { v.kind = JVal::Bool; v.b = false; return true; }
        if (lit("null")) { v.kind = JVal::Nul; return true; }
        {
            char* end = nullptr;
            const double d = std::strtod(p, &end);
            if (end == p) return fail("expected a value");
            p = end;
            v.kind = JVal::Num;
            v.num = d;
            return true;
        }
    }
};

bool read_file(const std::string& path, std::string& out, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "dflash2: cannot open " + path; return false; }
    out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return true;
}

bool parse_json_file(const std::string& path, JVal& out, std::string& err) {
    std::string text;
    if (!read_file(path, text, err)) return false;
    JParser ps;
    ps.p = text.data();
    ps.e = text.data() + text.size();
    if (!ps.value(out)) { err = "dflash2: " + ps.err + " (" + path + ")"; return false; }
    return true;
}

/// The checkpoint's own geometry: the semantic fields are pinned (bidirectional sliding attention,
/// no attention bias, a 2-tap / 16-group grouped conv, a 256-rank top-16 selector, silu, default
/// RoPE), the sizes come from the file, and `load` then checks every tensor against them - so a
/// mismatched or unsupported checkpoint fails here instead of drafting wrongly.
struct Geom {
    int layers = 5, heads = 32, kv_heads = 8, hd = 128, inter = 7680, rank = 256, group = 16, taps = 2;
    int block = 8, window = 2048;
    int64_t vocab = 248320;
    float eps = 1e-6f, theta = 10000.0f;
    int32_t mask = 248077;
};

bool read_geometry(const std::string& dir, const DFlash2Config& cfg, Geom& g, std::string& err) {
    JVal root;
    if (!parse_json_file(dir + "/config.json", root, err)) return false;
    auto bad = [&](const std::string& what) {
        err = "dflash2: " + dir + "/config.json: " + what;
        return false;
    };
    auto num = [&](const JVal* o, const char* key, double& out) -> bool {
        const JVal* v = o != nullptr ? o->find(key) : nullptr;
        if (v == nullptr) return bad(std::string("missing '") + key + "'");
        if (v->kind != JVal::Num) return bad(std::string("'") + key + "' is not a number");
        out = v->num;
        return true;
    };
    auto flag = [&](const JVal* o, const char* key, bool& out) -> bool {
        const JVal* v = o != nullptr ? o->find(key) : nullptr;
        if (v == nullptr) return bad(std::string("missing '") + key + "'");
        if (v->kind != JVal::Bool) return bad(std::string("'") + key + "' is not a bool");
        out = v->b;
        return true;
    };
    auto same = [&](const char* key, double want, double got) -> bool {
        if (want == got) return true;
        return bad(std::string("'") + key + "' is " + std::to_string(got) + ", this build requires " +
                   std::to_string(want));
    };
    auto close_enough = [&](const char* key, double want, double got) -> bool {
        if (std::fabs(want - got) <= 1e-6 * std::max(1.0, std::fabs(want))) return true;
        return bad(std::string("'") + key + "' is " + std::to_string(got) + ", this build requires " +
                   std::to_string(want));
    };

    const JVal* dc = root.find("dflash_config");
    if (dc == nullptr || dc->kind != JVal::Obj)
        return bad("no 'dflash_config' object: this is not a DFlash2 draft checkpoint");

    double v = 0;
    if (!num(&root, "num_hidden_layers", v) || !same("num_hidden_layers", 5, v)) return false;
    if (!num(&root, "hidden_size", v) || !same("hidden_size", cfg.draft_hidden, v)) return false;
    if (!num(&root, "intermediate_size", v)) return false;
    if (v < 1 || v > (1 << 20)) return bad("'intermediate_size' is out of range");
    g.inter = (int) v;
    if (!num(&root, "num_attention_heads", v)) return false;
    g.heads = (int) v;
    if (!num(&root, "num_key_value_heads", v)) return false;
    g.kv_heads = (int) v;
    if (!num(&root, "head_dim", v)) return false;
    g.hd = (int) v;
    if (g.heads < 1 || g.kv_heads < 1 || g.heads % g.kv_heads != 0 || g.hd < 8 || g.hd > kThreads ||
        kThreads % g.hd != 0)
        return bad("attention geometry is unsupported (kv heads must divide q heads, head_dim a divisor of 256)");
    if (!num(&root, "vocab_size", v) || !same("vocab_size", (double) cfg.vocab, v)) return false;
    if (!num(&root, "sliding_window", v) || !same("sliding_window", cfg.sliding_window, v)) return false;
    if (!num(&root, "rms_norm_eps", v) || !close_enough("rms_norm_eps", (double) cfg.rms_eps, v)) return false;

    bool b = false;
    // Non-causal, sliding-window, bias-free attention is the whole point of DFlash2: a checkpoint
    // that says otherwise would be silently mis-drafted by this backend.
    if (!flag(&root, "is_causal", b) || b != false) return bad("'is_causal' must be false (DFlash2 drafts bidirectionally)");
    {
        const JVal* ab = root.find("attention_bias");
        if (ab != nullptr && (ab->kind != JVal::Bool || ab->b))
            return bad("'attention_bias' must be false (this architecture has no attention bias)");
    }
    if (!flag(&root, "use_sliding_window", b) || b != true) return bad("'use_sliding_window' must be true");

    const JVal* lt = root.find("layer_types");
    if (lt == nullptr || lt->kind != JVal::Arr || lt->arr.size() != (size_t) g.layers)
        return bad("'layer_types' must hold one attention type per draft layer");
    for (const JVal& t : lt->arr)
        if (t.kind != JVal::Str || t.str != "sliding_attention")
            return bad("every draft layer must be 'sliding_attention'");

    const JVal* act = root.find("hidden_act");
    if (act == nullptr || act->kind != JVal::Str || act->str != "silu")
        return bad("'hidden_act' must be 'silu'");

    const JVal* rp = root.find("rope_parameters");
    if (rp == nullptr) return bad("missing 'rope_parameters'");
    if (!num(rp, "rope_theta", v) || !close_enough("rope_parameters.rope_theta", (double) cfg.rope_theta, v)) return false;
    const JVal* rt = rp->find("rope_type");
    if (rt != nullptr && (rt->kind != JVal::Str || rt->str != "default"))
        return bad("'rope_parameters.rope_type' must be 'default'");

    // Domino projectors (prefix_gru / embed_proj) would change the whole input path.
    if (root.find("projector_type") != nullptr) {
        const JVal* pt = root.find("projector_type");
        if (pt->kind == JVal::Str && !pt->str.empty())
            return bad("unsupported 'projector_type' '" + pt->str + "' (this backend has no projector)");
    }

    if (!num(dc, "block_size", v)) return false;
    if (v < 2 || v > kDFlash2MaxSlots + 1)
        return bad("'dflash_config.block_size' is out of the supported range");
    g.block = cfg.block_size;   // upstream's set_block_size may override the checkpoint's value
    if (!num(dc, "conv_kernel_size", v) || !same("dflash_config.conv_kernel_size", 2, v)) return false;
    if (!num(dc, "conv_group_size", v) || !same("dflash_config.conv_group_size", 16, v)) return false;
    if (!num(dc, "selector_rank", v) || !same("dflash_config.selector_rank", 256, v)) return false;
    if (!num(dc, "selector_top_k", v) || !same("dflash_config.selector_top_k", cfg.top_k, v)) return false;
    if (!num(dc, "mask_token_id", v)) return false;
    g.mask = (int32_t) v;

    const JVal* ids = dc->find("target_layer_ids");
    if (ids == nullptr || ids->kind != JVal::Arr || (int) ids->arr.size() != cfg.n_taps)
        return bad("'dflash_config.target_layer_ids' must list the " + std::to_string(cfg.n_taps) + " context taps");
    for (size_t i = 0; i < ids->arr.size(); ++i) {
        if (ids->arr[i].kind != JVal::Num) return bad("'dflash_config.target_layer_ids' must be numbers");
        if ((int) ids->arr[i].num != cfg.tap_layers[i])
            return bad("'dflash_config.target_layer_ids' does not match the configured taps");
    }

    // Everything the kernels size their buffers with has to divide cleanly and stay 8-aligned for
    // the fp16 GEMMs.
    if (cfg.draft_hidden % g.group != 0)
        return bad("conv_group_size must divide hidden_size");
    if (cfg.draft_hidden % 8 != 0 || g.inter % 8 != 0 || g.rank % 8 != 0 || g.hd % 8 != 0)
        return bad("hidden_size, intermediate_size, selector_rank and head_dim must be multiples of 8");
    g.vocab = cfg.vocab;
    g.window = cfg.sliding_window;
    g.eps = cfg.rms_eps;
    g.theta = cfg.rope_theta;
    return true;
}

// --------------------------------------------------------------------------- safetensors

struct StTensor {
    std::string dtype;
    int64_t numel = 0;
    std::vector<int64_t> shape;
    uint64_t begin = 0, end = 0;
};

/// The safetensors header: `{"name":{"dtype":"BF16","shape":[...],"data_offsets":[a,b]}, ...}`.
class SafeTensors {
public:
    bool open(const std::string& path, std::string& err) {
        path_ = path;
        f_.open(path, std::ios::binary);
        if (!f_) { err = "dflash2: cannot open " + path; return false; }
        uint64_t n = 0;
        f_.read(reinterpret_cast<char*>(&n), 8);
        if (!f_ || n == 0 || n > (1ull << 28)) { err = "dflash2: " + path + ": bad header size"; return false; }
        std::string hdr(n, '\0');
        f_.read(hdr.data(), (std::streamsize) n);
        if (!f_) { err = "dflash2: " + path + ": truncated header"; return false; }
        data_off_ = 8 + n;
        return parse(hdr, err);
    }

    const StTensor* find(const std::string& name) const {
        const auto it = tensors_.find(name);
        return it == tensors_.end() ? nullptr : &it->second;
    }

    /// Reads `t.numel` bf16 values in chunks, calling `sink(values, count)` with each chunk.
    template <typename Sink>
    bool stream(const StTensor& t, Sink&& sink, std::string& err) {
        constexpr int64_t kChunk = 1 << 21;   // 4 MB of bf16 per read
        std::vector<uint16_t> buf((size_t) std::min<int64_t>(kChunk, std::max<int64_t>(t.numel, 1)));
        f_.clear();
        f_.seekg((std::streamoff) (data_off_ + t.begin), std::ios::beg);
        if (!f_) { err = "dflash2: " + path_ + ": seek failed"; return false; }
        int64_t left = t.numel;
        while (left > 0) {
            const int64_t n = std::min<int64_t>(left, kChunk);
            f_.read(reinterpret_cast<char*>(buf.data()), (std::streamsize) (n * 2));
            if (!f_) { err = "dflash2: " + path_ + ": short read"; return false; }
            if (!sink(buf.data(), n)) return false;
            left -= n;
        }
        return true;
    }

    const std::vector<std::string>& names() const { return names_; }

private:
    bool parse(const std::string& hdr, std::string& err) {
        size_t i = 0;
        auto skip_ws = [&] { while (i < hdr.size() && (hdr[i] == ' ' || hdr[i] == '\t' || hdr[i] == '\n' || hdr[i] == '\r')) ++i; };
        auto expect = [&](char c) { skip_ws(); return i < hdr.size() && hdr[i] == c ? (++i, true) : false; };
        auto read_str = [&](std::string& out) {
            skip_ws();
            if (i >= hdr.size() || hdr[i] != '"') return false;
            ++i;
            out.clear();
            while (i < hdr.size() && hdr[i] != '"') {
                if (hdr[i] == '\\' && i + 1 < hdr.size()) ++i;
                out.push_back(hdr[i++]);
            }
            if (i >= hdr.size()) return false;
            ++i;
            return true;
        };
        auto skip_value = [&](int depth) -> bool {   // leaves `i` past a JSON value
            skip_ws();
            if (i >= hdr.size()) return false;
            if (hdr[i] == '"') { std::string s; return read_str(s); }
            if (hdr[i] == '{' || hdr[i] == '[') {
                const char open = hdr[i], close = open == '{' ? '}' : ']';
                int d = 0;
                while (i < hdr.size()) {
                    if (hdr[i] == '"') { std::string s; if (!read_str(s)) return false; continue; }
                    if (hdr[i] == open) ++d;
                    else if (hdr[i] == close && --d == 0) { ++i; return true; }
                    ++i;
                }
                return false;
            }
            while (i < hdr.size() && hdr[i] != ',' && hdr[i] != '}' && hdr[i] != ']') ++i;
            (void) depth;
            return true;
        };
        if (!expect('{')) { err = "dflash2: " + path_ + ": bad header"; return false; }
        skip_ws();
        if (i < hdr.size() && hdr[i] == '}') return true;
        while (true) {
            std::string key;
            if (!read_str(key)) { err = "dflash2: " + path_ + ": bad header key"; return false; }
            if (!expect(':')) { err = "dflash2: " + path_ + ": bad header"; return false; }
            skip_ws();
            if (i < hdr.size() && hdr[i] == '{') {
                const size_t entry_end = match_object(hdr, i);
                if (entry_end == std::string::npos) { err = "dflash2: " + path_ + ": bad header object"; return false; }
                if (key != "__metadata__") {
                    StTensor t;
                    std::string body = hdr.substr(i, entry_end - i);
                    if (!parse_entry(body, t, err)) return false;
                    names_.push_back(key);
                    tensors_.emplace(key, std::move(t));
                }
                i = entry_end;
            } else if (!skip_value(0)) {
                err = "dflash2: " + path_ + ": bad header value";
                return false;
            }
            skip_ws();
            if (i < hdr.size() && hdr[i] == ',') { ++i; continue; }
            if (i < hdr.size() && hdr[i] == '}') { ++i; return true; }
            err = "dflash2: " + path_ + ": bad header";
            return false;
        }
    }

    static size_t match_object(const std::string& s, size_t i) {
        int d = 0;
        bool in_str = false;
        for (; i < s.size(); ++i) {
            const char c = s[i];
            if (in_str) {
                if (c == '\\') ++i;
                else if (c == '"') in_str = false;
                continue;
            }
            if (c == '"') in_str = true;
            else if (c == '{') ++d;
            else if (c == '}') { if (--d == 0) return i + 1; }
        }
        return std::string::npos;
    }

    static bool parse_entry(const std::string& body, StTensor& t, std::string& err) {
        auto after = [&](const char* key) -> size_t {
            const std::string pat = std::string("\"") + key + "\"";
            const size_t p = body.find(pat);
            if (p == std::string::npos) return std::string::npos;
            const size_t colon = body.find(':', p + pat.size());
            return colon == std::string::npos ? std::string::npos : colon + 1;
        };
        size_t p = after("dtype");
        if (p == std::string::npos) { err = "dflash2: safetensors entry has no dtype"; return false; }
        const size_t q = body.find('"', p);
        const size_t q2 = q == std::string::npos ? std::string::npos : body.find('"', q + 1);
        if (q == std::string::npos || q2 == std::string::npos) { err = "dflash2: bad dtype"; return false; }
        t.dtype = body.substr(q + 1, q2 - q - 1);

        p = after("shape");
        if (p == std::string::npos) { err = "dflash2: safetensors entry has no shape"; return false; }
        const size_t lb = body.find('[', p), rb = body.find(']', p);
        if (lb == std::string::npos || rb == std::string::npos) { err = "dflash2: bad shape"; return false; }
        t.numel = 1;
        size_t k = lb + 1;
        while (k < rb) {
            while (k < rb && (body[k] == ' ' || body[k] == ',')) ++k;
            if (k >= rb) break;
            int64_t d = 0;
            while (k < rb && body[k] >= '0' && body[k] <= '9') d = d * 10 + (body[k++] - '0');
            t.shape.push_back(d);
            t.numel *= d;
        }

        p = after("data_offsets");
        if (p == std::string::npos) { err = "dflash2: safetensors entry has no data_offsets"; return false; }
        const size_t ob = body.find('[', p), cb = body.find(']', p);
        if (ob == std::string::npos || cb == std::string::npos) { err = "dflash2: bad data_offsets"; return false; }
        size_t comma = body.find(',', ob);
        if (comma == std::string::npos || comma > cb) { err = "dflash2: bad data_offsets"; return false; }
        t.begin = std::strtoull(body.c_str() + ob + 1, nullptr, 10);
        t.end = std::strtoull(body.c_str() + comma + 1, nullptr, 10);
        if (t.end < t.begin || t.end - t.begin != (uint64_t) t.numel * 2) {
            err = "dflash2: safetensors data_offsets do not span the shape";
            return false;
        }
        return true;
    }

    std::ifstream f_;
    std::string path_;
    uint64_t data_off_ = 0;
    std::map<std::string, StTensor> tensors_;
    std::vector<std::string> names_;
};

// --------------------------------------------------------------------------- kernels

__device__ __forceinline__ float warp_max_f(float v) {
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
    return v;
}

__device__ __forceinline__ float warp_sum_f(float v) {
    for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

/// `sh` holds at least 32 floats; the result is in sh[0] for every thread on return.
__device__ __forceinline__ float block_max_f(float v, float* sh) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int nw = (blockDim.x + 31) >> 5;
    v = warp_max_f(v);
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = lane < nw ? sh[lane] : -INFINITY;
        v = warp_max_f(v);
        if (lane == 0) sh[0] = v;
    }
    __syncthreads();
    return sh[0];
}

__device__ __forceinline__ float block_sum_f(float v, float* sh) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int nw = (blockDim.x + 31) >> 5;
    v = warp_sum_f(v);
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = lane < nw ? sh[lane] : 0.0f;
        v = warp_sum_f(v);
        if (lane == 0) sh[0] = v;
    }
    __syncthreads();
    return sh[0];
}

/// The GEMM operand cast: the weights are fp16 and the activations fp32, and only the activation
/// that enters a matmul is rounded to fp16 - exactly the reference's `x.astype(fp16) @ w.T`.
__global__ void f32_to_f16_kernel(const float* __restrict__ in, __half* __restrict__ out, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (int64_t) gridDim.x * blockDim.x)
        out[i] = __float2half(in[i]);
}

/// RMSNorm over `cols` with the fused residual add the reference layers use: `res` is the residual
/// row (in/out, may be null), `v = x (+ res)`, `res = v`, `out = rmsnorm(v) * w`.  `out` may alias
/// `x` (the context path normalizes in place); `res` is read and written element-wise by one thread.
__global__ void rmsnorm_res_kernel(const float* x, float* res, const float* __restrict__ w, float* out,
                                   int cols, float eps, int add_res) {
    __shared__ float sh[32];
    const size_t off = (size_t) blockIdx.x * cols;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = x[off + i];
        if (add_res) v += res[off + i];
        if (res != nullptr) res[off + i] = v;
        sum = fmaf(v, v, sum);
    }
    const float rstd = rsqrtf(block_sum_f(sum, sh) / (float) cols + eps);
    __syncthreads();
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const float v = res != nullptr ? res[off + i] : x[off + i];
        out[off + i] = v * rstd * w[i];
    }
}

/// Per-head RMSNorm (Qwen3's q_norm / k_norm), applied over each `head_dim` slice.
__global__ void head_rmsnorm_kernel(float* __restrict__ x, const float* __restrict__ w, int n_heads,
                                    int head_dim, float eps) {
    __shared__ float sh[32];
    float* xr = x + ((size_t) blockIdx.x * n_heads + blockIdx.y) * head_dim;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float v = xr[i];
        sum = fmaf(v, v, sum);
    }
    const float rstd = rsqrtf(block_sum_f(sum, sh) / (float) head_dim + eps);
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) xr[i] = xr[i] * rstd * w[i];
}

/// Rotary (NeoX / HF Qwen3 half-split) at `pos0 + blockIdx.x`, in place.
__global__ void rope_kernel(float* __restrict__ x, int n_heads, int head_dim, int64_t pos0,
                            const float* __restrict__ inv_freq) {
    const int half = head_dim >> 1;
    if ((int) threadIdx.x >= half) return;
    float* xr = x + ((size_t) blockIdx.x * n_heads + blockIdx.y) * head_dim;
    const float a = xr[threadIdx.x];
    const float b = xr[threadIdx.x + half];
    const float ang = (float) (pos0 + blockIdx.x) * inv_freq[threadIdx.x];
    const float c = __cosf(ang), s = __sinf(ang);
    xr[threadIdx.x] = a * c - b * s;
    xr[threadIdx.x + half] = b * c + a * s;
}

__global__ void silu_mul_kernel(const float* __restrict__ g, const float* __restrict__ u,
                                float* __restrict__ out, int64_t n) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (int64_t) gridDim.x * blockDim.x) {
        const float x = g[i];
        out[i] = u[i] * (x / (1.0f + __expf(-x)));
    }
}

/// Grouped dynamic causal convolution (`_grouped_conv`): coefficients are ADDITIVE
/// (`base[side][tap][ch] + delta[row][side][tap][ch / group]`) and a tap > 0 term is dropped at
/// block position < tap.  `delta` is [rows][2][taps][groups], `base` is [2][taps][hidden].
__global__ void grouped_conv_kernel(const float* __restrict__ src, const float* __restrict__ delta,
                                    const float* __restrict__ base, float* __restrict__ out, int hidden,
                                    int taps, int group, int block_size, int side) {
    const int groups = hidden / group;
    const int row = blockIdx.x;
    const int pos = block_size > 0 ? row % block_size : row;
    const float* drow = delta + (size_t) row * 2 * taps * groups + (size_t) side * taps * groups;
    const float* brow = base + (size_t) side * taps * hidden;
    for (int ch = threadIdx.x; ch < hidden; ch += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < taps; ++t) {
            if (t > 0 && pos < t) continue;
            const float coef = brow[(size_t) t * hidden + ch] + drow[(size_t) t * groups + ch / group];
            acc = fmaf(coef, src[(size_t) (row - t) * hidden + ch], acc);
        }
        out[(size_t) row * hidden + ch] = acc;
    }
}

/// Non-causal sliding-window attention for the block rows.  Keys are the context ring
/// (`ctx_len` rows at positions ctx_pos0.., indexed from `ring_start`) followed by the block's own
/// rows (`n_blk` at q_pos0..).  Visible iff |q_pos - k_pos| < window.
///
/// One warp owns one key at a time. Its lanes load contiguous float4 segments and reduce the
/// query-key dot product across the warp. Value accumulation retains the existing partitioned
/// reduction: head-sized thread groups visit disjoint keys, then one thread combines their
/// partials and writes each output dimension.
__global__ void attn_kernel(const float* __restrict__ q, const float* __restrict__ k_ring,
                            const float* __restrict__ v_ring, const float* __restrict__ k_blk,
                            const float* __restrict__ v_blk, float* __restrict__ out, int n_blk,
                            int n_q_heads, int n_kv_heads, int head_dim, int ring_start, int ring_cap,
                            int ctx_len, int64_t ctx_pos0, int64_t q_pos0, int window, float scale) {
    extern __shared__ float sh[];
    const int row = blockIdx.x, h = blockIdx.y;
    const int tid = threadIdx.x;
    const int n_keys = ctx_len + n_blk;
    float* scores = sh;
    float* red = sh + n_keys;
    const int gqa = n_q_heads / n_kv_heads;
    const int kvh = h / gqa;
    const int64_t qpos = q_pos0 + row;
    const float* qrow = q + ((size_t) row * n_q_heads + h) * head_dim;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int nwarp = blockDim.x >> 5;
    const int nslot = head_dim >> 2;                    // float4 slots in a row (head_dim % 4 == 0)
    constexpr int kSlots = (kThreads / 4 + 31) / 32;    // head_dim <= kThreads -> at most 2 per lane

    float4 qv[kSlots];                                  // the query row, hoisted out of the key loop
#pragma unroll
    for (int k = 0; k < kSlots; ++k) {
        const int f = lane + 32 * k;
        if (f < nslot) qv[k] = *reinterpret_cast<const float4*>(qrow + 4 * f);
    }

    // ctx_len <= ring_cap and ring_start < ring_cap: each context index wraps at most once.
    for (int i = wid; i < n_keys; i += nwarp) {
        const float* krow;
        int64_t kpos;
        if (i < ctx_len) {
            kpos = ctx_pos0 + i;
            const int index = ring_start + i;
            const int slot = index < ring_cap ? index : index - ring_cap;
            krow = k_ring + ((size_t) slot * n_kv_heads + kvh) * head_dim;
        } else {
            const int j = i - ctx_len;
            kpos = q_pos0 + j;
            krow = k_blk + ((size_t) j * n_kv_heads + kvh) * head_dim;
        }
        float s = -INFINITY;
        const int64_t d = qpos - kpos;                  // uniform across the warp: i depends on wid
        if (d < (int64_t) window && d > -(int64_t) window) {
            float4 a[kSlots];
#pragma unroll
            for (int k = 0; k < kSlots; ++k) a[k] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
            for (int k = 0; k < kSlots; ++k) {
                const int f = lane + 32 * k;
                if (f < nslot) {
                    const float4 kv = *reinterpret_cast<const float4*>(krow + 4 * f);
                    a[k].x = fmaf(qv[k].x, kv.x, a[k].x);
                    a[k].y = fmaf(qv[k].y, kv.y, a[k].y);
                    a[k].z = fmaf(qv[k].z, kv.z, a[k].z);
                    a[k].w = fmaf(qv[k].w, kv.w, a[k].w);
                }
            }
            float acc = 0.0f;
#pragma unroll
            for (int k = 0; k < kSlots; ++k) acc += (a[k].x + a[k].y) + (a[k].z + a[k].w);
            s = warp_sum_f(acc) * scale;                // every lane takes this branch, so the
        }                                               // full mask is live; the total is in lane 0
        if (lane == 0) scores[i] = s;
    }
    __syncthreads();
    float m = -INFINITY;
    for (int i = tid; i < n_keys; i += blockDim.x) m = fmaxf(m, scores[i]);
    m = block_max_f(m, red);
    float sum = 0.0f;
    for (int i = tid; i < n_keys; i += blockDim.x) {
        const float e = __expf(scores[i] - m);
        scores[i] = e;
        sum += e;
    }
    sum = block_sum_f(sum, red);       // its last __syncthreads makes every write visible
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    const int per = blockDim.x / head_dim;
    const int d = tid % head_dim;
    const int part = tid / head_dim;
    float acc = 0.0f;
    for (int i = part; i < n_keys; i += per) {
        const float* vrow;
        if (i < ctx_len) {
            const int index = ring_start + i;
            const int slot = index < ring_cap ? index : index - ring_cap;
            vrow = v_ring + ((size_t) slot * n_kv_heads + kvh) * head_dim;
        } else
            vrow = v_blk + ((size_t) (i - ctx_len) * n_kv_heads + kvh) * head_dim;
        acc = fmaf(scores[i] * inv, vrow[d], acc);
    }
    float* vred = red + 32;
    vred[part * head_dim + d] = acc;
    __syncthreads();
    if (tid < head_dim) {
        float sum_value = 0.0f;
        for (int p = 0; p < per; ++p) sum_value += vred[p * head_dim + tid];
        out[((size_t) row * n_q_heads + h) * head_dim + tid] = sum_value;
    }
}

/// Native top-k order: NaN first, then descending value, ties to the lower id. A NaN logit must
/// still yield a REAL id - a sentinel here would index the codebooks out of bounds downstream.
__device__ __forceinline__ bool topk_better(float a, int32_t ai, float b, int32_t bi) {
    const bool an = isnan(a), bn = isnan(b);
    if (an != bn) return an;
    if (an) return ai < bi;
    return a > b || (a == b && ai < bi);
}

__device__ __forceinline__ void warp_best_f(float& v, int32_t& id) {
    for (int o = 16; o; o >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, v, o);
        const int32_t oi = __shfl_down_sync(0xffffffffu, id, o);
        if (topk_better(ov, oi, v, id)) { v = ov; id = oi; }
    }
}

/// One SHARD of a row's top-k: the same comparator and the same block merge as before, restricted to
/// the contiguous span [blockIdx.y * shard_len, +shard_len).  `gridDim.y` shards per row; each writes
/// its own `k` entries, shard-major, into the row's partial list.
/// The union's top-k is exactly the merge of the shards' top-k (topk_better is a total order on
/// (value, unique id) pairs), so the merged result is identical to one block scanning the whole row.
__global__ void topk_shard_kernel(const float* __restrict__ logits, int64_t vocab, int64_t limit,
                                  int64_t shard_len, int k, int32_t* __restrict__ part_ids,
                                  float* __restrict__ part_vals) {
    __shared__ float sv[kTopKThreads * kTopKMax];
    __shared__ int32_t si[kTopKThreads * kTopKMax];
    __shared__ int32_t cur[kTopKThreads];
    __shared__ float bsh[kMergeWarps];
    __shared__ int32_t bsi[kMergeWarps];
    const int tid = threadIdx.x;
    const int row = blockIdx.x, shard = blockIdx.y;
    const float* lg = logits + (int64_t) row * vocab;
    const int64_t lo = (int64_t) shard * shard_len;
    const int64_t hi = lo + shard_len < limit ? lo + shard_len : limit;
    const int kk = k < kTopKMax ? k : kTopKMax;

    float tv[kTopKMax];
    int32_t ti[kTopKMax];
    for (int j = 0; j < kTopKMax; ++j) { tv[j] = -INFINITY; ti[j] = INT32_MAX; }
    for (int64_t i = lo + tid; i < hi; i += blockDim.x) {
        const float v = lg[i];
        if (topk_better(v, (int32_t) i, tv[kk - 1], ti[kk - 1])) {
            int j = kk - 1;
            while (j > 0 && topk_better(v, (int32_t) i, tv[j - 1], ti[j - 1])) {
                tv[j] = tv[j - 1];
                ti[j] = ti[j - 1];
                --j;
            }
            tv[j] = v;
            ti[j] = (int32_t) i;
        }
    }
    for (int j = 0; j < kTopKMax; ++j) {
        sv[tid * kTopKMax + j] = tv[j];
        si[tid * kTopKMax + j] = ti[j];
    }
    cur[tid] = 0;
    __syncthreads();
    float* pv = part_vals + ((int64_t) row * gridDim.y + shard) * k;
    int32_t* pi = part_ids + ((int64_t) row * gridDim.y + shard) * k;
    for (int step = 0; step < kk; ++step) {
        const int c = cur[tid];
        float v = c < kk ? sv[tid * kTopKMax + c] : -INFINITY;
        int32_t id = c < kk ? si[tid * kTopKMax + c] : INT32_MAX;
        const int lane = tid & 31, wid = tid >> 5;
        const int nw = (blockDim.x + 31) >> 5;
        warp_best_f(v, id);
        if (lane == 0) { bsh[wid] = v; bsi[wid] = id; }
        __syncthreads();
        if (wid == 0) {
            float wv = lane < nw ? bsh[lane] : -INFINITY;
            int32_t wi = lane < nw ? bsi[lane] : INT32_MAX;
            warp_best_f(wv, wi);
            if (lane == 0) { bsh[0] = wv; bsi[0] = wi; }
        }
        __syncthreads();
        const float bv = bsh[0];
        const int32_t bid = bsi[0];
        if (tid == 0) {
            pv[step] = bv;
            pi[step] = bid;
        }
        if (c < kk && si[tid * kTopKMax + c] == bid) cur[tid] = c + 1;
        __syncthreads();
    }
}

/// Merges `g` shard lists of `k` entries each (shard-major, contiguous per row) into the row's final
/// top-k, with the same comparator: one thread owns one entry and offers it until it is emitted, so
/// the order, the NaN rule and the lower-id tie rule are exactly the single-block kernel's.
__global__ void topk_merge_kernel(const int32_t* __restrict__ part_ids,
                                  const float* __restrict__ part_vals, int g, int k,
                                  int32_t* __restrict__ ids_out, float* __restrict__ vals_out) {
    __shared__ float bsh[kMergeWarps];
    __shared__ int32_t bsi[kMergeWarps];
    const int tid = threadIdx.x;
    const int n_in = g * k;
    const int row = blockIdx.x;
    const bool owns = tid < n_in;
    float v = owns ? part_vals[(int64_t) row * n_in + tid] : -INFINITY;
    int32_t id = owns ? part_ids[(int64_t) row * n_in + tid] : INT32_MAX;
    const int lane = tid & 31, wid = tid >> 5;
    const int nw = (blockDim.x + 31) >> 5;
    for (int step = 0; step < k; ++step) {
        float bv = v;
        int32_t bid = id;
        warp_best_f(bv, bid);
        if (lane == 0) { bsh[wid] = bv; bsi[wid] = bid; }
        __syncthreads();
        if (wid == 0) {
            float wv = lane < nw ? bsh[lane] : -INFINITY;
            int32_t wi = lane < nw ? bsi[lane] : INT32_MAX;
            warp_best_f(wv, wi);
            if (lane == 0) { bsh[0] = wv; bsi[0] = wi; }
        }
        __syncthreads();
        if (tid == 0) {
            vals_out[(int64_t) row * k + step] = bsh[0];
            ids_out[(int64_t) row * k + step] = bsi[0];
        }
        if (owns && id == bsi[0]) {   // this entry has been emitted
            v = -INFINITY;
            id = INT32_MAX;
        }
        __syncthreads();
    }
}

/// `out[i] = table[ids[i]]`, `rank` halves per row (the selector's codebooks).
__global__ void gather_rows_kernel(const __half* __restrict__ table, const int32_t* __restrict__ ids,
                                   __half* __restrict__ out, int rank) {
    const int64_t row = blockIdx.x;
    const int64_t src = (int64_t) ids[row] * rank;
    for (int j = threadIdx.x; j < rank; j += blockDim.x) out[row * rank + j] = table[src + j];
}

/// The selector's predecessor ids: slot 0 is the anchor replicated over the candidates, slot e
/// takes slot e-1's candidates (`_score_edges`).
__global__ void pred_ids_kernel(const int32_t* __restrict__ cand, int32_t anchor, int slots, int k,
                                int32_t* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= slots * k) return;
    const int e = i / k, p = i - e * k;
    out[i] = e == 0 ? anchor : cand[(e - 1) * k + p];
}

/// The selector's walk ON THE DEVICE: one block, `slots` sequential steps.  Slot e scores the `k`
/// candidates of the REALIZED predecessor's lattice row, then takes the greedy argmax or the sampling
/// inverse-CDF draw, and writes the chosen position, the token id and the q row.
/// The score expression and its order, the temperature clamp (done by the caller), the NaN rules, the
/// tie rule and the CDF boundary rule (`the first c with u < cumsum`, else k-1) are the same as the
/// host walk this replaces; the products use explicit round-to-nearest so the compiler cannot contract
/// them into an FMA and change the last bits.
__global__ void selector_kernel(const int32_t* __restrict__ cand_ids, const float* __restrict__ unary,
                                const float* __restrict__ H, const __half* __restrict__ a_rows,
                                const __half* __restrict__ b_rows, const float* __restrict__ uniforms,
                                int slots, int k, int rank, float temperature, int greedy,
                                int32_t* __restrict__ sel_tokens,
                                float* __restrict__ sel_q) {
    __shared__ int s_prev;
    __shared__ float s_scores[kDFlash2MaxTopK];
    const int tid = threadIdx.x;
    if (tid == 0) s_prev = 0;
    __syncthreads();
    for (int e = 0; e < slots; ++e) {
        const int prev = s_prev;
        if (tid < k) {   // one candidate per thread: the row's scores in the host's summation order
            const int c = tid;
            const __half* A = a_rows + ((size_t) e * k + prev) * rank;
            const float* Hh = H + (size_t) e * rank;
            const __half* B = b_rows + ((size_t) e * k + c) * rank;
            float acc = unary[(size_t) e * k + c];
            for (int r = 0; r < rank; ++r)
                acc = __fadd_rn(acc, __fmul_rn(__fmul_rn(__half2float(A[r]), Hh[r]), __half2float(B[r])));
            s_scores[c] = acc;
        }
        __syncthreads();
        if (tid == 0) {
            int idx = 0;
            if (greedy) {
                for (int c = 0; c < k; ++c)
                    if (isnan(s_scores[c])) { idx = c; break; }
                for (int c = 1; c < k; ++c)
                    if (s_scores[c] > s_scores[idx]) idx = c;
                for (int c = 0; c < k; ++c) sel_q[(size_t) e * k + c] = c == idx ? 1.0f : 0.0f;
            } else {
                float mx = -INFINITY;   // fmaxf ignores a NaN operand exactly as std::max does
                for (int c = 0; c < k; ++c) mx = fmaxf(mx, s_scores[c] / temperature);
                float sum = 0.0f;
                for (int c = 0; c < k; ++c) {
                    const float pr = expf(s_scores[c] / temperature - mx);
                    sel_q[(size_t) e * k + c] = pr;
                    sum += pr;
                }
                const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
                for (int c = 0; c < k; ++c) sel_q[(size_t) e * k + c] *= inv;
                const float u = uniforms[e];
                float cum = 0.0f;
                idx = k - 1;
                for (int c = 0; c < k; ++c) {
                    cum += sel_q[(size_t) e * k + c];
                    if (u < cum) { idx = c; break; }
                }
            }
            sel_tokens[e] = cand_ids[(size_t) e * k + idx];
            s_prev = idx;
        }
        __syncthreads();
    }
}

/// Writes `count` rows into the ring at absolute rows `first..first+count-1` (modulo `cap`).
__global__ void ring_write_kernel(const float* __restrict__ src, float* __restrict__ ring, int row_len,
                                  int cap, int first, int count) {
    const int i = blockIdx.x;
    if (i >= count) return;
    const int dst = (first + i) % cap;
    for (int j = threadIdx.x; j < row_len; j += blockDim.x) ring[dst * row_len + j] = src[i * row_len + j];
}

/// Target tap rows -> one context row: the hyper-connection streams are averaged, the taps
/// concatenated in `tap_layers` order (the `fc` input).
/// `tap_stride` is the caller's row stride between taps (the capture buffer's capacity): row `r` of
/// tap `t` lives at `(t * tap_stride + r) * hc * hidden`.
__global__ void build_ctx_features_kernel(const float* __restrict__ in, float* __restrict__ out,
                                          int taps, int hc, int hidden, int tap_stride, int row0) {
    const int row = row0 + blockIdx.x;
    const float inv = 1.0f / (float) hc;
    float* o = out + (size_t) blockIdx.x * taps * hidden;
    for (int i = threadIdx.x; i < taps * hidden; i += blockDim.x) {
        const int t = i / hidden, ch = i - t * hidden;
        const float* p = in + ((size_t) t * tap_stride + row) * hc * hidden + ch;
        float s = 0.0f;
        for (int k = 0; k < hc; ++k) s += p[(size_t) k * hidden];
        o[i] = s * inv;
    }
}

}  // namespace

// --------------------------------------------------------------------------- the backend

struct DFlash2Backend::Impl {
    DFlash2Config cfg;
    bool loaded = false;
    int device = -1;
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;

    // geometry (validated against config.json)
    int B = 8, slots = 7, hidden = 2560, heads = 32, kv_heads = 8, hd = 128, inter = 7680;
    int taps = 5, ctaps = 2, hc = 4, group = 16, groups = 160, top_k = 16, rank = 256, n_layers = 5;
    int thid = 2560;                   ///< target hidden per hyper-connection stream (the tap width)
    int64_t vocab = 248320;
    float eps = 1e-6f, theta = 10000.0f, scale = 0.08838834764831845f;
    int window = 2048;
    int cap = 2047;                       // ring rows
    int32_t mask_token = 248077;
    int kv_dim = 1024;

    struct Layer {
        const float *input_ln = nullptr, *post_ln = nullptr, *q_norm = nullptr, *k_norm = nullptr;
        const float *attn_base = nullptr, *mlp_base = nullptr;
        const __half *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr;
        const __half *gate = nullptr, *up = nullptr, *down = nullptr;
        const __half *attn_kproj = nullptr, *mlp_kproj = nullptr;
    };
    std::vector<Layer> layers;
    const __half* fc = nullptr;
    const float* hidden_norm_w = nullptr;
    const float* final_norm_w = nullptr;
    const __half* pred_cb = nullptr;
    const __half* succ_cb = nullptr;
    const __half* hproj = nullptr;
    float* inv_freq = nullptr;

    // context ring: K/V are GEMM outputs and stay fp32 (the reference keeps them fp32)
    std::vector<float*> ring_k, ring_v;
    int ring_start = 0;
    int64_t ctx_len = 0;
    int64_t next_pos = 0;

    // scratch (device): every ACTIVATION and the residual stream are fp32 - the reference rounds only
    // the operand of a matmul to fp16, and the real checkpoint's residual reaches RMS ~4e4, which
    // overflows fp16 outright.  The weights stay fp16 (bf16 -> fp16 is exact for this checkpoint).
    std::vector<void*> allocs;
    float *emb = nullptr, *x_next = nullptr, *res = nullptr, *h = nullptr;
    float *conv_pre = nullptr, *attn_pre = nullptr;
    float *q = nullptr, *k = nullptr, *v = nullptr, *blk_k = nullptr, *blk_v = nullptr;
    float *attn_out = nullptr, *attn_proj = nullptr, *attn_fin = nullptr, *h2 = nullptr;
    float *gate = nullptr, *up = nullptr, *act = nullptr, *mlp_out = nullptr;
    float *final_h = nullptr;
    float *ctx_f = nullptr, *ctx_hidden = nullptr, *tmp_k = nullptr, *tmp_v = nullptr;
    __half* x16 = nullptr;                ///< the fp16 copy of a GEMM operand (the reference's cast)
    float *delta = nullptr, *delta2 = nullptr, *H = nullptr, *logits = nullptr;
    int32_t* cand_ids = nullptr;
    float* cand_vals = nullptr;
    int32_t* pred_ids = nullptr;
    __half* a_rows = nullptr;
    __half* b_rows = nullptr;
    // the sharded top-k's partial lists and the device selector's outputs (all preallocated)
    int32_t* part_ids = nullptr;
    float* part_vals = nullptr;
    int32_t* sel_tokens = nullptr;
    float* sel_q = nullptr;
    float* sel_uniforms = nullptr;

    // host
    std::vector<float> h_unary;
    std::vector<float> h_q, h_uniforms;
    std::vector<int32_t> h_cand, h_tokens;
    DFlash2HeadFn head_fn = nullptr;
    void* head_ctx = nullptr;
    uint64_t steps_ = 0;
    uint64_t bytes_alloc = 0;
    bool debug = false;

    ~Impl() {
        if (blas != nullptr) cublasDestroy(blas);
        if (stream != nullptr) cudaStreamDestroy(stream);
        for (void* p : allocs) cudaFree(p);
    }

    bool alloc(void** out, size_t bytes, const char* what, std::string& err) {
        void* p = nullptr;
        if (!cuda_ok(cudaMalloc(&p, bytes), what, err)) return false;
        bytes_alloc += bytes;
        allocs.push_back(p);
        *out = p;
        return true;
    }
    template <typename T>
    bool alloc_t(T** out, size_t count, const char* what, std::string& err) {
        void* p = nullptr;
        if (!alloc(&p, count * sizeof(T), what, err)) return false;
        *out = (T*) p;
        return true;
    }

    /// y[n_tok, n_out] = fp16(x) * W^T, fp32 accumulate and fp32 output - the reference's matmul
    /// (`x.astype(fp16) @ w.T`, weights already fp16).  The activation is rounded to fp16 on the way
    /// in; nothing else in the step is, so a large residual cannot overflow.
    bool gemm(const __half* W, int n_out, int n_in, const float* x, int n_tok, float* y,
              const char* what, std::string& err) {
        const int64_t n = (int64_t) n_tok * n_in;
        if (n > 0)
            f32_to_f16_kernel<<<(unsigned) ((n + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
                x, x16, n);
        const float one = 1.0f, zero = 0.0f;
        return blas_ok(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, n_out, n_tok, n_in, &one, W,
                                    CUDA_R_16F, n_in, x16, CUDA_R_16F, n_in, &zero, y, CUDA_R_32F, n_out,
                                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                       what, err);
    }

    bool load(const DFlash2Config& c, std::string& err);
    bool append(const float* dev_tap_rows, int64_t n_rows, int64_t tap_stride_rows, int64_t base_pos,
                std::string& err);
    bool propose(int32_t anchor_id, int64_t start_pos, const float* block_embeds, float temperature,
                 bool greedy, const float* uniforms, DFlash2Proposal& out, std::string& err);
};

// --------------------------------------------------------------------------- load

bool DFlash2Backend::Impl::load(const DFlash2Config& c, std::string& err) {
    cfg = c;
    debug = env_on("STRATA_DFLASH2_DEBUG");
    if (cfg.model_dir.empty()) { err = "dflash2: no model_dir"; return false; }
    if (!cuda_ok(cudaSetDevice(cfg.device), "cudaSetDevice", err)) return false;
    device = cfg.device;

    B = cfg.block_size;
    hidden = cfg.draft_hidden;
    thid = cfg.target_hidden;
    taps = cfg.n_taps;                 // context features (target layer taps)
    hc = cfg.hc_streams;
    vocab = cfg.vocab;
    top_k = cfg.top_k;
    cap = cfg.max_context;
    window = cfg.sliding_window;
    eps = cfg.rms_eps;
    theta = cfg.rope_theta;
    if (B < 2 || B > kDFlash2MaxSlots + 1) { err = "dflash2: block_size must be 2.." + std::to_string(kDFlash2MaxSlots + 1); return false; }
    if (top_k < 1 || top_k > kTopKMax) { err = "dflash2: top_k must be 1.." + std::to_string(kTopKMax); return false; }
    if (taps < 1 || taps > 16) { err = "dflash2: n_taps must be 1..16"; return false; }
    if (hc < 1) { err = "dflash2: hc_streams must be >= 1"; return false; }
    if (thid < 8 || thid % 8 != 0) { err = "dflash2: target_hidden must be a multiple of 8"; return false; }
    if (cap < 1 || cap > 8192) { err = "dflash2: max_context must be 1..8192"; return false; }
    if (cap < cfg.sliding_window - 1) {
        err = "dflash2: max_context (" + std::to_string(cap) + ") is below the attention window (" +
              std::to_string(cfg.sliding_window) + "): visible context keys would be dropped";
        return false;
    }
    if (cfg.draft_hidden % 16 != 0) { err = "dflash2: draft_hidden must be a multiple of 16"; return false; }
    slots = B - 1;

    Geom g;
    if (!read_geometry(cfg.model_dir, cfg, g, err)) return false;
    // the checkpoint is authoritative for the geometry; every tensor is checked against it below
    n_layers = g.layers;
    inter = g.inter;
    heads = g.heads;
    kv_heads = g.kv_heads;
    hd = g.hd;
    rank = g.rank;
    group = g.group;
    ctaps = g.taps;                    // the grouped conv's taps (conv_kernel_size)
    B = g.block;
    slots = B - 1;
    mask_token = g.mask;
    groups = hidden / group;
    kv_dim = kv_heads * hd;
    scale = 1.0f / std::sqrt((float) hd);

    if (!cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream", err)) return false;
    if (!blas_ok(cublasCreate(&blas), "cublasCreate", err)) return false;
    if (!blas_ok(cublasSetStream(blas, stream), "cublasSetStream", err)) return false;
    if (!blas_ok(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH), "cublasSetMathMode", err)) return false;

    SafeTensors st;
    if (!st.open(cfg.model_dir + "/model.safetensors", err)) return false;
    for (const std::string& n : st.names()) {
        if (n.rfind("prefix_gru.", 0) == 0 || n.rfind("embed_proj.", 0) == 0 ||
            n.find(".prefix_gru.") != std::string::npos || n.find(".embed_proj.") != std::string::npos) {
            err = "dflash2: the checkpoint carries Domino projector weights (" + n + "), unsupported";
            return false;
        }
    }

    // bf16 -> fp16/fp32, with the fp16 range check the conversion needs.
    auto convert = [&](const std::string& name, std::initializer_list<int64_t> dims, bool as_f32, void** dst,
                       std::string& e) -> bool {
        const StTensor* t = st.find(name);
        if (t == nullptr) { e = "dflash2: the checkpoint has no tensor '" + name + "'"; return false; }
        if (t->dtype != "BF16") { e = "dflash2: '" + name + "' is " + t->dtype + ", expected BF16"; return false; }
        int64_t want = 1;
        for (int64_t d : dims) want *= d;
        bool shape_ok = t->shape.size() == dims.size();
        if (shape_ok) {
            size_t i = 0;
            for (int64_t d : dims) shape_ok = shape_ok && t->shape[i++] == d;
        }
        if (!shape_ok || t->numel != want) {
            std::string got, want_s;
            for (size_t i = 0; i < t->shape.size(); ++i) got += (i ? "x" : "") + std::to_string(t->shape[i]);
            for (int64_t d : dims) want_s += (want_s.empty() ? "" : "x") + std::to_string(d);
            e = "dflash2: '" + name + "' has shape [" + got + "], expected [" + want_s + "]";
            return false;
        }
        void* dev = nullptr;
        if (!alloc(&dev, (size_t) want * (as_f32 ? 4 : 2), name.c_str(), e)) return false;
        std::vector<float> fbuf;
        std::vector<uint16_t> hbuf;
        if (as_f32) fbuf.resize((size_t) std::min<int64_t>(want, 1 << 21));
        else hbuf.resize((size_t) std::min<int64_t>(want, 1 << 21));
        int64_t done = 0;
        double max_abs = 0.0;
        int64_t bad = 0;
        bool ok = st.stream(*t, [&](const uint16_t* src, int64_t n) {
            for (int64_t i = 0; i < n; ++i) {
                const float f = [&] {
                    const uint32_t u = (uint32_t) src[i] << 16;
                    float x;
                    std::memcpy(&x, &u, sizeof x);
                    return x;
                }();
                if (!std::isfinite(f) || (!as_f32 && std::fabs(f) > 65504.0f)) ++bad;
                max_abs = std::max(max_abs, (double) std::fabs(f));
                if (as_f32) fbuf[(size_t) i] = f;
                else hbuf[(size_t) i] = bf16_to_f16(src[i]);
            }
            const void* host = as_f32 ? (const void*) fbuf.data() : (const void*) hbuf.data();
            if (cudaMemcpyAsync((char*) dev + done * (as_f32 ? 4 : 2), host,
                                (size_t) n * (as_f32 ? 4 : 2), cudaMemcpyHostToDevice, stream) != cudaSuccess) {
                e = "dflash2: uploading '" + name + "' failed";
                return false;
            }
            done += n;
            return true;
        }, e);
        if (!ok) return false;
        if (bad > 0) {
            e = "dflash2: '" + name + "': " + std::to_string(bad) +
                " values are outside fp16's range (max |v| = " + std::to_string(max_abs) +
                "); this checkpoint cannot run on the fp16 path";
            return false;
        }
        if (debug)
            std::fprintf(stderr, "dflash2: %-52s bf16->%s max|v|=%.4g\n", name.c_str(),
                         as_f32 ? "f32" : "f16", max_abs);
        *dst = dev;
        return true;
    };
    auto load_f16 = [&](const std::string& n, const __half** dst, std::initializer_list<int64_t> dims) {
        return convert(n, dims, false, (void**) dst, err);
    };
    auto load_f32 = [&](const std::string& n, const float** dst, std::initializer_list<int64_t> dims) {
        return convert(n, dims, true, (void**) dst, err);
    };

    // fc's width is the tap count's fail-fast check: taps x the target's per-stream hidden
    if (!load_f16("fc.weight", &fc, {hidden, (int64_t) taps * thid})) return false;
    if (!load_f32("hidden_norm.weight", &hidden_norm_w, {hidden})) return false;
    if (!load_f32("norm.weight", &final_norm_w, {hidden})) return false;
    if (!load_f16("candidate_selector.hidden_projection.weight", &hproj, {rank, hidden})) return false;
    // The two codebooks are stored WITHOUT `.weight`.
    if (!load_f16("candidate_selector.predecessor_codebook", &pred_cb, {vocab, rank})) return false;
    if (!load_f16("candidate_selector.successor_codebook", &succ_cb, {vocab, rank})) return false;

    layers.resize((size_t) n_layers);
    for (int li = 0; li < n_layers; ++li) {
        Layer& L = layers[(size_t) li];
        const std::string p = "layers." + std::to_string(li) + ".";
        if (!load_f32(p + "input_layernorm.weight", &L.input_ln, {hidden})) return false;
        if (!load_f32(p + "post_attention_layernorm.weight", &L.post_ln, {hidden})) return false;
        if (!load_f32(p + "self_attn.q_norm.weight", &L.q_norm, {hd})) return false;
        if (!load_f32(p + "self_attn.k_norm.weight", &L.k_norm, {hd})) return false;
        if (!load_f16(p + "self_attn.q_proj.weight", &L.q, {heads * hd, hidden})) return false;
        if (!load_f16(p + "self_attn.k_proj.weight", &L.k, {kv_dim, hidden})) return false;
        if (!load_f16(p + "self_attn.v_proj.weight", &L.v, {kv_dim, hidden})) return false;
        if (!load_f16(p + "self_attn.o_proj.weight", &L.o, {hidden, heads * hd})) return false;
        if (!load_f16(p + "mlp.gate_proj.weight", &L.gate, {inter, hidden})) return false;
        if (!load_f16(p + "mlp.up_proj.weight", &L.up, {inter, hidden})) return false;
        if (!load_f16(p + "mlp.down_proj.weight", &L.down, {hidden, inter})) return false;
        // the conv base kernels are [2 (pre/post side), conv taps, hidden]; the projections
        // [2 * conv taps * groups, hidden]
        if (!load_f32(p + "attention_conv.base_kernel", &L.attn_base, {2, ctaps, hidden})) return false;
        if (!load_f16(p + "attention_conv.kernel_projection.weight", &L.attn_kproj,
                      {2 * (int64_t) ctaps * groups, hidden})) return false;
        if (!load_f32(p + "mlp_conv.base_kernel", &L.mlp_base, {2, ctaps, hidden})) return false;
        if (!load_f16(p + "mlp_conv.kernel_projection.weight", &L.mlp_kproj,
                      {2 * (int64_t) ctaps * groups, hidden})) return false;
    }

    // rope frequencies
    std::vector<float> inv((size_t) (hd / 2));
    for (int i = 0; i < hd / 2; ++i) inv[(size_t) i] = std::pow(theta, -2.0f * (float) i / (float) hd);
    if (!alloc_t(&inv_freq, inv.size(), "inv_freq", err)) return false;
    if (!cuda_ok(cudaMemcpyAsync(inv_freq, inv.data(), inv.size() * sizeof(float), cudaMemcpyHostToDevice,
                                 stream), "inv_freq upload", err)) return false;

    // context ring: one K and one V ring per layer
    ring_k.resize((size_t) n_layers);
    ring_v.resize((size_t) n_layers);
    for (int li = 0; li < n_layers; ++li) {
        if (!alloc_t(&ring_k[(size_t) li], (size_t) cap * kv_dim, "ring k", err)) return false;
        if (!alloc_t(&ring_v[(size_t) li], (size_t) cap * kv_dim, "ring v", err)) return false;
    }

    // activations
    const size_t bb = (size_t) B;
    if (!alloc_t(&emb, bb * hidden, "emb", err)) return false;
    if (!alloc_t(&x_next, bb * hidden, "x2", err)) return false;
    if (!alloc_t(&res, bb * hidden, "res", err)) return false;
    if (!alloc_t(&h, bb * hidden, "h", err)) return false;
    if (!alloc_t(&conv_pre, bb * hidden, "conv", err)) return false;
    if (!alloc_t(&attn_pre, bb * hidden, "attn_pre", err)) return false;
    if (!alloc_t(&q, bb * (size_t) heads * hd, "q", err)) return false;
    if (!alloc_t(&k, bb * kv_dim, "k", err)) return false;
    if (!alloc_t(&v, bb * kv_dim, "v", err)) return false;
    if (!alloc_t(&blk_k, bb * kv_dim, "blk k", err)) return false;
    if (!alloc_t(&blk_v, bb * kv_dim, "blk v", err)) return false;
    if (!alloc_t(&attn_out, bb * (size_t) heads * hd, "attn out", err)) return false;
    if (!alloc_t(&attn_proj, bb * hidden, "attn proj", err)) return false;
    if (!alloc_t(&attn_fin, bb * hidden, "attn fin", err)) return false;
    if (!alloc_t(&h2, bb * hidden, "h2", err)) return false;
    if (!alloc_t(&gate, bb * inter, "gate", err)) return false;
    if (!alloc_t(&up, bb * inter, "up", err)) return false;
    if (!alloc_t(&act, bb * inter, "act", err)) return false;
    if (!alloc_t(&mlp_out, bb * hidden, "mlp", err)) return false;
    if (!alloc_t(&final_h, bb * hidden, "final", err)) return false;
    if (!alloc_t(&delta, bb * (size_t) 2 * ctaps * groups, "delta", err)) return false;
    if (!alloc_t(&delta2, bb * (size_t) 2 * ctaps * groups, "delta2", err)) return false;
    if (!alloc_t(&ctx_f, (size_t) kChunkRows * taps * thid, "ctx features", err)) return false;
    if (!alloc_t(&ctx_hidden, (size_t) kChunkRows * hidden, "ctx hidden", err)) return false;
    if (!alloc_t(&tmp_k, (size_t) kChunkRows * kv_dim, "ctx k", err)) return false;
    if (!alloc_t(&tmp_v, (size_t) kChunkRows * kv_dim, "ctx v", err)) return false;
    // the largest GEMM operand is the context's fc input (kChunkRows x taps x target_hidden)
    if (!alloc_t(&x16, (size_t) kChunkRows * taps * thid, "gemm operand", err)) return false;
    if (!alloc_t(&H, (size_t) slots * rank, "H", err)) return false;
    if (!alloc_t(&logits, (size_t) slots * (size_t) vocab, "logits", err)) return false;
    if (!alloc_t(&cand_ids, (size_t) slots * top_k, "cand ids", err)) return false;
    if (!alloc_t(&cand_vals, (size_t) slots * top_k, "cand vals", err)) return false;
    if (!alloc_t(&pred_ids, (size_t) slots * top_k, "pred ids", err)) return false;
    if (!alloc_t(&a_rows, (size_t) slots * top_k * rank, "A rows", err)) return false;
    if (!alloc_t(&b_rows, (size_t) slots * top_k * rank, "B rows", err)) return false;
    const size_t part_n = (size_t) slots * kSelShards * kTopKMax;
    if (!alloc_t(&part_ids, part_n, "topk parts", err)) return false;
    if (!alloc_t(&part_vals, part_n, "topk parts", err)) return false;
    if (!alloc_t(&sel_tokens, (size_t) slots, "sel tokens", err)) return false;
    if (!alloc_t(&sel_q, (size_t) slots * top_k, "sel q", err)) return false;
    if (!alloc_t(&sel_uniforms, (size_t) slots, "sel uniforms", err)) return false;

    h_unary.assign((size_t) slots * top_k, 0.0f);
    h_q.assign((size_t) slots * top_k, 0.0f);
    h_cand.assign((size_t) slots * top_k, 0);
    h_tokens.assign((size_t) slots, 0);
    h_uniforms.assign((size_t) slots, 0.0f);

    if (!cuda_ok(cudaStreamSynchronize(stream), "load sync", err)) return false;
    loaded = true;
    if (debug)
        std::fprintf(stderr, "dflash2: loaded %s: %d layers, hidden %d, block %d (%d slots), taps %d, hc %d, vocab %lld, ring %d rows\n",
                     cfg.model_dir.c_str(), n_layers, hidden, B, slots, taps, hc, (long long) vocab, cap);
    return true;
}

// --------------------------------------------------------------------------- context

bool DFlash2Backend::Impl::append(const float* dev_tap_rows, int64_t n_rows, int64_t tap_stride_rows,
                                  int64_t base_pos, std::string& err) {
    if (n_rows <= 0) { err = "dflash2: append_features with no rows"; return false; }
    // A fresh ring may start at any position (a batched prefill or a cache resume hands the first
    // window position); once it holds rows, the appends have to be consecutive.
    if (ctx_len == 0) next_pos = base_pos;
    else if (base_pos != next_pos) {
        err = "dflash2: append_features at position " + std::to_string(base_pos) + " but the ring expects " +
              std::to_string(next_pos);
        return false;
    }
    if (dev_tap_rows == nullptr) { err = "dflash2: append_features with no rows"; return false; }
    // The caller's per-tap row stride is its capture buffer's capacity; 0 means the rows are packed.
    const int64_t stride = tap_stride_rows > 0 ? tap_stride_rows : n_rows;
    if (stride < n_rows || stride > (int64_t) INT32_MAX) {
        err = "dflash2: append_features: the tap stride (" + std::to_string(stride) +
              ") is below the row count (" + std::to_string(n_rows) + ")";
        return false;
    }

    // Only the newest `cap` rows can ever be attended to (the window is narrower than the ring), so a
    // whole-prompt append keeps its tail and skips the rest.
    const int64_t first_row = n_rows > cap ? n_rows - cap : 0;
    const int64_t kept = n_rows - first_row;
    const int64_t total = ctx_len + kept;
    if (total > cap) {
        const int64_t drop = total - cap;
        ring_start = (int) ((ring_start + drop) % cap);
        ctx_len -= drop;
    }
    const int64_t keep = ctx_len;
    ctx_len += kept;
    next_pos += n_rows;

    const int kv_dim_ = kv_dim;
    for (int64_t off = first_row; off < n_rows; off += kChunkRows) {
        const int m = (int) std::min<int64_t>(kChunkRows, n_rows - off);
        build_ctx_features_kernel<<<m, kThreads, 0, stream>>>(dev_tap_rows, ctx_f, taps, hc, thid,
                                                              (int) stride, (int) off);
        if (!gemm(fc, hidden, taps * thid, ctx_f, m, ctx_hidden, "fc", err)) return false;
        rmsnorm_res_kernel<<<m, kThreads, 0, stream>>>(ctx_hidden, nullptr, hidden_norm_w, ctx_hidden,
                                                       hidden, eps, 0);
        const int64_t pos0 = base_pos + off;
        for (int li = 0; li < n_layers; ++li) {
            const Layer& L = layers[(size_t) li];
            if (!gemm(L.k, kv_dim_, hidden, ctx_hidden, m, tmp_k, "ctx k_proj", err)) return false;
            if (!gemm(L.v, kv_dim_, hidden, ctx_hidden, m, tmp_v, "ctx v_proj", err)) return false;
            head_rmsnorm_kernel<<<dim3((unsigned) m, (unsigned) kv_heads), 128, 0, stream>>>(
                tmp_k, L.k_norm, kv_heads, hd, eps);
            rope_kernel<<<dim3((unsigned) m, (unsigned) kv_heads), (unsigned) (hd / 2), 0, stream>>>(
                tmp_k, kv_heads, hd, pos0, inv_freq);
            const int first = (int) ((ring_start + keep + (off - first_row)) % cap);
            ring_write_kernel<<<(unsigned) m, kThreads, 0, stream>>>(tmp_k, ring_k[(size_t) li], kv_dim_,
                                                                     cap, first, m);
            ring_write_kernel<<<(unsigned) m, kThreads, 0, stream>>>(tmp_v, ring_v[(size_t) li], kv_dim_,
                                                                     cap, first, m);
        }
    }
    if (!cuda_ok(cudaGetLastError(), "append kernels", err)) return false;
    if (!cuda_ok(cudaStreamSynchronize(stream), "append sync", err)) return false;
    return true;
}

// --------------------------------------------------------------------------- one step

bool DFlash2Backend::Impl::propose(int32_t anchor_id, int64_t start_pos, const float* block_embeds,
                                   float temperature, bool greedy, const float* uniforms,
                                   DFlash2Proposal& out, std::string& err) {
    if (head_fn == nullptr) { err = "dflash2: propose() before set_target_head()"; return false; }
    if (block_embeds == nullptr) { err = "dflash2: propose() without block embeddings"; return false; }
    if (ctx_len > 0 && start_pos != next_pos) {
        err = "dflash2: propose() at position " + std::to_string(start_pos) + " but the context ends at " +
              std::to_string(next_pos);
        return false;
    }
    if (!(temperature > 0.0f)) temperature = 1e-5f;
    if (temperature < 1e-5f) temperature = 1e-5f;

    const int bb = B, hh = hidden, hd_ = hd, nq = heads, nkv = kv_heads, kv_dim_ = kv_dim;
    const size_t emb_n = (size_t) bb * hh;
    if (!cuda_ok(cudaMemcpyAsync(emb, block_embeds, emb_n * sizeof(float), cudaMemcpyDeviceToDevice,
                                 stream), "block embeddings", err)) return false;
    // the layers ping-pong between two row buffers: no copy of a layer's output per layer
    float* lin = emb;
    float* lout = x_next;

    for (int li = 0; li < n_layers; ++li) {
        const Layer& L = layers[(size_t) li];
        rmsnorm_res_kernel<<<bb, kThreads, 0, stream>>>(lin, res, L.input_ln, h, hh, eps, li == 0 ? 0 : 1);
        if (!gemm(L.attn_kproj, 2 * ctaps * groups, hh, h, bb, delta, "attn conv proj", err)) return false;
        grouped_conv_kernel<<<bb, kThreads, 0, stream>>>(h, delta, L.attn_base, conv_pre, hh, ctaps, group,
                                                         B, 0);
        if (!gemm(L.q, nq * hd_, hh, conv_pre, bb, q, "q_proj", err)) return false;
        if (!gemm(L.k, kv_dim_, hh, conv_pre, bb, blk_k, "k_proj", err)) return false;
        if (!gemm(L.v, kv_dim_, hh, conv_pre, bb, blk_v, "v_proj", err)) return false;
        head_rmsnorm_kernel<<<dim3((unsigned) bb, (unsigned) nq), 128, 0, stream>>>(q, L.q_norm, nq, hd_, eps);
        head_rmsnorm_kernel<<<dim3((unsigned) bb, (unsigned) nkv), 128, 0, stream>>>(blk_k, L.k_norm, nkv, hd_, eps);
        rope_kernel<<<dim3((unsigned) bb, (unsigned) nq), (unsigned) (hd_ / 2), 0, stream>>>(q, nq, hd_,
                                                                                            start_pos, inv_freq);
        rope_kernel<<<dim3((unsigned) bb, (unsigned) nkv), (unsigned) (hd_ / 2), 0, stream>>>(blk_k, nkv, hd_,
                                                                                             start_pos, inv_freq);

        const int n_keys = (int) ctx_len + bb;
        const size_t shmem = ((size_t) n_keys + 32 + (size_t) (kThreads / hd_) * hd_) * sizeof(float);
        attn_kernel<<<dim3((unsigned) bb, (unsigned) nq), kThreads, shmem, stream>>>(
            q, ring_k[(size_t) li], ring_v[(size_t) li], blk_k, blk_v, attn_out, bb, nq, nkv, hd_, ring_start,
            cap, (int) ctx_len, next_pos - ctx_len, start_pos, window, scale);
        if (!gemm(L.o, hh, nq * hd_, attn_out, bb, attn_proj, "o_proj", err)) return false;
        grouped_conv_kernel<<<bb, kThreads, 0, stream>>>(attn_proj, delta, L.attn_base, attn_fin, hh, ctaps,
                                                         group, B, 1);
        rmsnorm_res_kernel<<<bb, kThreads, 0, stream>>>(attn_fin, res, L.post_ln, h2, hh, eps, 1);

        if (!gemm(L.mlp_kproj, 2 * ctaps * groups, hh, h2, bb, delta2, "mlp conv proj", err)) return false;
        grouped_conv_kernel<<<bb, kThreads, 0, stream>>>(h2, delta2, L.mlp_base, attn_pre, hh, ctaps, group,
                                                         B, 0);
        if (!gemm(L.gate, inter, hh, attn_pre, bb, gate, "gate_proj", err)) return false;
        if (!gemm(L.up, inter, hh, attn_pre, bb, up, "up_proj", err)) return false;
        silu_mul_kernel<<<(unsigned) ((bb * (int64_t) inter + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
            gate, up, act, (int64_t) bb * inter);
        if (!gemm(L.down, hh, inter, act, bb, mlp_out, "down_proj", err)) return false;
        grouped_conv_kernel<<<bb, kThreads, 0, stream>>>(mlp_out, delta2, L.mlp_base, lout, hh, ctaps, group,
                                                         B, 1);
        std::swap(lin, lout);
    }

    rmsnorm_res_kernel<<<bb, kThreads, 0, stream>>>(lin, res, final_norm_w, final_h, hh, eps, 1);
    if (!cuda_ok(cudaGetLastError(), "forward kernels", err)) return false;

    // the TARGET's head over the draft rows (row 0, the anchor, is discarded), then the candidates
    if (!head_fn(head_ctx, final_h + hh, slots, logits, (void*) stream, err)) return false;
    if (!cuda_ok(cudaGetLastError(), "head", err)) return false;
    // the candidates: kSelShards contiguous shards per row, then the merge (identical to one block
    // scanning the whole row - see topk_shard_kernel)
    {
        const int64_t shard_len = (vocab + kSelShards - 1) / kSelShards;
        topk_shard_kernel<<<dim3((unsigned) slots, (unsigned) kSelShards), kTopKThreads, 0, stream>>>(
            logits, vocab, vocab, shard_len, top_k, part_ids, part_vals);
        topk_merge_kernel<<<(unsigned) slots, kMergeThreads, 0, stream>>>(part_ids, part_vals, kSelShards,
                                                                          top_k, cand_ids, cand_vals);
    }
    if (!gemm(hproj, rank, hh, final_h + hh, slots, H, "hidden_projection", err)) return false;
    pred_ids_kernel<<<(unsigned) ((slots * top_k + 127) / 128), 128, 0, stream>>>(cand_ids, anchor_id, slots,
                                                                                  top_k, pred_ids);
    gather_rows_kernel<<<(unsigned) (slots * top_k), kThreads, 0, stream>>>(pred_cb, pred_ids, a_rows, rank);
    gather_rows_kernel<<<(unsigned) (slots * top_k), kThreads, 0, stream>>>(succ_cb, cand_ids, b_rows, rank);
    if (!cuda_ok(cudaGetLastError(), "selector kernels", err)) return false;

    // the selector's uniforms (host -> device, `slots` floats) and the walk itself, on the device
    for (int e = 0; e < slots; ++e)
        h_uniforms[(size_t) e] = (!greedy && uniforms != nullptr) ? uniforms[e] : 0.0f;
    if (!cuda_ok(cudaMemcpyAsync(sel_uniforms, h_uniforms.data(), (size_t) slots * sizeof(float),
                                 cudaMemcpyHostToDevice, stream), "selector uniforms", err)) return false;
    selector_kernel<<<1, kThreads, 0, stream>>>(cand_ids, cand_vals, H, a_rows, b_rows, sel_uniforms, slots,
                                                top_k, rank, temperature, greedy ? 1 : 0, sel_tokens,
                                                sel_q);
    if (!cuda_ok(cudaGetLastError(), "selector", err)) return false;

    // host copies: the candidate list, the unary logits and the selector's own outputs - no H/A/B rows
    if (!cuda_ok(cudaMemcpyAsync(h_cand.data(), cand_ids, (size_t) slots * top_k * sizeof(int32_t),
                                 cudaMemcpyDeviceToHost, stream), "cand ids", err)) return false;
    if (!cuda_ok(cudaMemcpyAsync(h_unary.data(), cand_vals, (size_t) slots * top_k * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream), "unary", err)) return false;
    if (!cuda_ok(cudaMemcpyAsync(h_q.data(), sel_q, (size_t) slots * top_k * sizeof(float),
                                 cudaMemcpyDeviceToHost, stream), "q rows", err)) return false;
    if (!cuda_ok(cudaMemcpyAsync(h_tokens.data(), sel_tokens, (size_t) slots * sizeof(int32_t),
                                 cudaMemcpyDeviceToHost, stream), "tokens", err)) return false;
    if (!cuda_ok(cudaStreamSynchronize(stream), "step sync", err)) return false;

    // ---- the proposal: the device selector's tokens and q rows plus the candidate list it chose from
    const int k = top_k;
    out.n_slots = slots;
    out.n_cand = k;
    for (int e = 0; e < slots; ++e) {
        out.tokens[e] = h_tokens[(size_t) e];
        for (int c = 0; c < k; ++c) {
            out.cand[(size_t) e * k + c] = h_cand[(size_t) e * k + c];
            out.unary[(size_t) e * k + c] = h_unary[(size_t) e * k + c];
            out.q[(size_t) e * k + c] = h_q[(size_t) e * k + c];
        }
    }
    ++steps_;
    return true;
}

// --------------------------------------------------------------------------- public API

DFlash2Backend::DFlash2Backend() : impl_(new Impl) {}
DFlash2Backend::~DFlash2Backend() = default;

bool DFlash2Backend::load(const DFlash2Config& cfg, std::string& err) { return impl_->load(cfg, err); }
bool DFlash2Backend::loaded() const { return impl_->loaded; }

void DFlash2Backend::reset() {
    impl_->ring_start = 0;
    impl_->ctx_len = 0;
    impl_->next_pos = 0;
    impl_->steps_ = 0;
}

int DFlash2Backend::slots() const { return impl_->slots; }
int DFlash2Backend::block_size() const { return impl_->B; }
int DFlash2Backend::top_k() const { return impl_->top_k; }
int DFlash2Backend::n_draft_layers() const { return impl_->n_layers; }
uint64_t DFlash2Backend::device_bytes() const { return impl_->bytes_alloc; }
uint64_t DFlash2Backend::steps() const { return impl_->steps_; }
int32_t DFlash2Backend::mask_token_id() const { return impl_->mask_token; }

void DFlash2Backend::set_target_head(DFlash2HeadFn fn, void* ctx) {
    impl_->head_fn = fn;
    impl_->head_ctx = ctx;
}

bool DFlash2Backend::append_features(const float* dev_tap_rows, int64_t n_rows, int64_t tap_stride_rows,
                                     int64_t base_pos, std::string& err) {
    if (!impl_->loaded) { err = "dflash2: append_features before load"; return false; }
    return impl_->append(dev_tap_rows, n_rows, tap_stride_rows, base_pos, err);
}

int64_t DFlash2Backend::next_pos() const { return impl_->next_pos; }
int64_t DFlash2Backend::ctx_len() const { return impl_->ctx_len; }

bool DFlash2Backend::propose(int32_t anchor_id, int64_t start_pos, const float* block_embeds,
                             float temperature, bool greedy, const float* uniforms, DFlash2Proposal& out,
                             std::string& err) {
    if (!impl_->loaded) { err = "dflash2: propose before load"; return false; }
    return impl_->propose(anchor_id, start_pos, block_embeds, temperature, greedy, uniforms, out, err);
}

bool DFlash2Backend::copy_pred_hidden(float* out, std::string& err) const {
    const Impl& d = *impl_;
    if (!d.loaded) { err = "dflash2: copy_pred_hidden before load"; return false; }
    if (out == nullptr) { err = "dflash2: copy_pred_hidden with no destination"; return false; }
    if (d.steps_ == 0) { err = "dflash2: copy_pred_hidden before the first propose()"; return false; }
    return cuda_ok(cudaMemcpy(out, d.final_h + d.hidden, (size_t) d.slots * d.hidden * sizeof(float),
                              cudaMemcpyDeviceToHost), "pred hidden", err);
}
}  // namespace strata::core
