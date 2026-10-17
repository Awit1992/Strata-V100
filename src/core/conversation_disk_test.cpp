// src/core/conversation_disk_test.cpp - the L3 disk tier (append-only chains), CPU only. No CUDA.
//
// The test builds SavedConversation deltas by hand - the same shape the engine captures: sliced
// ids/imgs, the current running state, only the NEW checkpoints, and K/V TAILS from the first new
// page - writes them through the store, and checks the merged read-back, the prefix lookup, the
// base ('p' root) resolution, the torn-tail recovery, the rewind, the eviction ordering, the
// startup re-index, and every corruption path.
#include "strata/core/conversation_disk.hpp"
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using namespace strata::core;

namespace {

int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) {
        std::fprintf(stderr, "FAIL: %s\n", description);
        std::exit(1);
    }
}

ConversationBuffer make_buffer(size_t bytes, uint8_t value) {
    ConversationBuffer buffer;
    buffer.resize(bytes);
    buffer.visit(0, bytes, [&](uint8_t* data, size_t chunk, size_t) {
        std::memset(data, value, chunk);
        return true;
    });
    return buffer;
}

int64_t aligned(int64_t cells, int64_t page) { return (cells + page - 1) / page * page; }

// A stage skeleton with one K/V layer; the caller sets the ids and the page-rounded cells.
SavedConversation make_stage(int64_t rate, uint8_t fill, bool cvec = true) {
    SavedConversation stage;
    stage.geometry[0] = 8;
    stage.geometry[1] = 1;
    stage.geometry[2] = 64;
    stage.layer_lo = 0;
    stage.layer_hi = 4;
    stage.cvec = cvec;
    stage.live.gdn.assign(8, fill);
    stage.live.ple.assign(5, static_cast<uint8_t>(fill + 1));
    stage.live.tails.assign(3, static_cast<uint8_t>(fill + 2));
    stage.live.dead.assign(2, static_cast<uint8_t>(fill + 3));
    stage.live.block_pos.assign(4, static_cast<uint8_t>(fill + 4));
    stage.kv.resize(1);
    ConversationKv& kv = stage.kv.front();
    kv.format = 0;
    kv.heads = 1;
    kv.head_dim = 64;
    kv.page_size = 8;
    kv.pooled_rows = 2;
    kv.idx_dim = 8;
    return stage;
}

// The full image of a conversation at `upto` tokens (the reference the merges must equal).

const std::vector<int32_t>& all_ids() {
    static const std::vector<int32_t> ids = [] {
        std::vector<int32_t> v;
        for (int32_t i = 11; v.size() < 40; i += 11) v.push_back(i);
        return v;
    }();
    return ids;
}
// The ids-based form serves the longer chains the retention tests build (all_ids() stops at 40).
SavedConversation full_image(const std::vector<int32_t>& ids, size_t upto, uint8_t fill = 77,
                             int64_t rate = 4) {
    SavedConversation stage = make_stage(rate, fill);
    stage.live.ids.assign(ids.begin(), ids.begin() + (int64_t) upto);
    stage.kv.front().cells = aligned((int64_t) upto, 8);
    const int64_t cells = stage.kv.front().cells;
    stage.kv.front().k = make_buffer((size_t) (cells * rate), fill);
    stage.kv.front().v = make_buffer((size_t) (cells * rate), static_cast<uint8_t>(fill + 1));
    stage.kv.front().k_scale = make_buffer((size_t) (cells * 2), static_cast<uint8_t>(fill + 2));
    stage.kv.front().v_scale = make_buffer((size_t) (cells * 2), static_cast<uint8_t>(fill + 3));
    stage.kv.front().pooled = make_buffer((size_t) ((upto / 4 + 1) * 32), static_cast<uint8_t>(fill + 4));
    return stage;
}
SavedConversation full_image(size_t upto, uint8_t fill = 77, int64_t rate = 4) {
    return full_image(all_ids(), upto, fill, rate);
}

// The engine-captured DELTA for the same stage: ids [first..end), the CURRENT run-state, only
// checkpoints newer than `first`, and K/V TAILS beginning at the page containing `first`.

// A byte-exact tail slice of a full buffer, starting at byte `offset`.
ConversationBuffer copy_slice(const ConversationBuffer& src, size_t offset, size_t count) {
    ConversationBuffer out;
    out.resize(count);
    if (count != 0) {
        std::vector<uint8_t> tmp(count);
        check(src.read(tmp.data(), offset, count), "a tail slice reads its source");
        size_t done = 0;
        out.visit(0, count, [&](uint8_t* p, size_t chunk, size_t) {
            std::memcpy(p, tmp.data() + done, chunk);
            done += chunk;
            return true;
        });
    }
    return out;
}
SavedConversation make_delta_stage(const SavedConversation& full, int64_t first, int64_t rate) {
    SavedConversation delta;
    delta.geometry = full.geometry;
    delta.layer_lo = full.layer_lo;
    delta.layer_hi = full.layer_hi;
    delta.cvec = full.cvec;
    delta.live.ids.assign(full.live.ids.begin() + first, full.live.ids.end());
    delta.live.gdn = full.live.gdn;
    delta.live.ple = full.live.ple;
    delta.live.tails = full.live.tails;
    delta.live.dead = full.live.dead;
    delta.live.block_pos = full.live.block_pos;
    delta.live.used = full.live.used;
    delta.kv.resize(1);
    const ConversationKv& kv = full.kv.front();
    const int64_t page_pad = (first / kv.page_size) * kv.page_size;
    ConversationKv& out = delta.kv.front();
    out.format = kv.format;
    out.cells = kv.cells;
    out.heads = kv.heads;
    out.head_dim = kv.head_dim;
    out.page_size = kv.page_size;
    out.pooled_rows = kv.pooled_rows;
    out.idx_dim = kv.idx_dim;
    out.k = copy_slice(kv.k, (size_t) page_pad * rate, kv.k.size() - (size_t) page_pad * rate);
    out.v = copy_slice(kv.v, (size_t) page_pad * rate, kv.v.size() - (size_t) page_pad * rate);
    out.k_scale = copy_slice(kv.k_scale, (size_t) page_pad * 2, kv.k_scale.size() - (size_t) page_pad * 2);
    out.v_scale = copy_slice(kv.v_scale, (size_t) page_pad * 2, kv.v_scale.size() - (size_t) page_pad * 2);
    out.pooled = copy_slice(kv.pooled, (size_t) ((first / 4) * 32), kv.pooled.size() - (size_t) ((first / 4) * 32));
    out.first_units = {page_pad, page_pad, page_pad, page_pad, first / 4};
    return delta;
}

bool same_checkpoint(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    return a.gdn == b.gdn && a.ple == b.ple && a.tails == b.tails && a.dead == b.dead &&
           a.block_pos == b.block_pos && a.used == b.used && a.stage_parts.size() == b.stage_parts.size();
}

bool same_kv(const ConversationKv& a, const ConversationKv& b) {
    return a.format == b.format && a.cells == b.cells && a.heads == b.heads && a.head_dim == b.head_dim &&
           a.page_size == b.page_size && a.pooled_rows == b.pooled_rows && a.idx_dim == b.idx_dim && a.k == b.k &&
           a.v == b.v && a.k_scale == b.k_scale && a.v_scale == b.v_scale && a.pooled == b.pooled;
}

bool same_stage(const SavedConversation& a, const SavedConversation& b) {
    if (a.geometry != b.geometry || a.layer_lo != b.layer_lo || a.layer_hi != b.layer_hi || a.cvec != b.cvec)
        return false;
    if (!same_checkpoint(a.live, b.live) || !same_kv(a.kv.front(), b.kv.front())) return false;
    return true;
}

bool same_record(const ConversationDiskRecord& a, const ConversationDiskRecord& b) {
    if (a.stages.size() != b.stages.size()) return false;
    for (size_t i = 0; i < a.stages.size(); ++i)
        if (!same_stage(a.stages[i], b.stages[i])) return false;
    return true;
}

ConversationDiskOptions options_for(const std::filesystem::path& directory,
                                    const ConversationDiskIdentity& identity, uint64_t budget) {
    ConversationDiskOptions options;
    options.directory = directory;
    options.identity = identity;
    options.budget_bytes = budget;
    return options;
}

void patch_byte(const std::filesystem::path& path, uint64_t offset, uint8_t value) {
    std::FILE* file = std::fopen(path.string().c_str(), "r+b");
    check(file != nullptr, "a patched record opens");
    check(std::fseek(file, static_cast<long>(offset), SEEK_SET) == 0, "a patched record seeks");
    check(std::fwrite(&value, 1, 1, file) == 1, "a patched record writes");
    std::fclose(file);
}

std::filesystem::path scratch_dir(const char* tag) {
    std::filesystem::path dir = std::filesystem::temp_directory_path() /
                                ("strata-disk-" + std::string(tag) + "-" +
                                 std::to_string((unsigned long) getpid()));
    std::filesystem::remove_all(dir);
    return dir;
}

// What the store charges for one chain: its own file plus the anchor sidecar it keeps beside it.
uint64_t stored_bytes(const std::filesystem::path& directory, const std::string& chain) {
    uint64_t bytes =
        std::filesystem::file_size(directory / (chain + ConversationDiskStore::file_suffix));
    const std::filesystem::path states = directory / (chain + ".states");
    std::error_code error;
    if (std::filesystem::is_regular_file(states, error))
        bytes += std::filesystem::file_size(states, error);
    return bytes;
}

void test_l2_first_spill() {
    const auto dir = scratch_dir("l2-spill");
    ConversationDiskStore store;
    std::string error;
    check(store.open(options_for(dir, ConversationDiskIdentity::from_string("l2-spill"), 1ull << 30), error) ==
              ConversationDiskStatus::ok, "spill store opens");
    ConversationCache cache(1 << 20, 2);
    cache.on_eviction([](SavedConversation&& image, bool root, void* context) noexcept {
        std::string error;
        check(static_cast<ConversationDiskStore*>(context)->spill(std::move(image), root, error) ==
                  ConversationDiskStatus::ok, "capacity victim spills to disk");
    }, &store);
    auto first = full_image(8, 41);
    first.stage_images.push_back(full_image(8, 42));
    const auto root_name = conversation_disk_prefix_name(first.live.ids, {}, true);
    auto second = full_image(8, 51);
    second.live.ids[0] = 901;
    check(cache.put(std::move(first), 0, true) && cache.put(std::move(second), 0, true),
          "roots fill L2 before any disk spill");
    check(store.records() == 0 && std::filesystem::is_empty(dir),
          "resident L2 entries create no disk records");
    auto active = full_image(8, 61);
    active.live.ids[0] = 902;
    const auto active_ids = active.live.ids;
    check(cache.put(std::move(active)), "latest conversation enters a root-filled L2");
    check(store.records() == 1 && store.has(root_name),
          "only the evicted oldest root enters L3");
    ConversationDiskRecord back;
    check(store.get(root_name, back, error) == ConversationDiskStatus::ok &&
              back.stages.size() == 2 &&
              back.stages[0].live.ids == full_image(8, 41).live.ids &&
              same_stage(back.stages[0], full_image(8, 41)) &&
              same_stage(back.stages[1], full_image(8, 42)),
          "spilled root restores both stages and exact payloads");
    auto prompt = active_ids;
    prompt.push_back(999);
    auto a = cache.take(cache.best(prompt, {}, true).index);
    auto other = full_image(8, 71);
    other.live.ids[0] = 903;
    auto other_prompt = other.live.ids;
    other_prompt.push_back(999);
    check(cache.put(std::move(other), a.bytes()), "second conversation parks during a RAM exchange");
    auto b = cache.take(cache.best(other_prompt, {}, true).index);
    check(cache.put(std::move(a), b.bytes()) && cache.best(prompt, {}, true).tokens == 8,
          "alternating conversations continue to restore from L2");
    check(store.records() == 1, "RAM exchanges do not write through to disk");
    check(cache.evict_oldest(), "physical RAM pressure spills an ordinary victim");
    const auto conversation_name = conversation_disk_name(active_ids, {}, true);
    check(store.has(conversation_name) && store.get(conversation_name, back, error) ==
              ConversationDiskStatus::ok && back.stages[0].live.ids == active_ids &&
              same_stage(back.stages[0], full_image(8, 61)),
          "ordinary spill preserves the full conversation payload");
    std::filesystem::remove_all(dir);
}

void test_basic_chain() {
    std::filesystem::path dir = scratch_dir("chain");
    ConversationDiskStore store;
    std::string error;
    check(store.open(options_for(dir, ConversationDiskIdentity::from_string("test"), 1ull << 30), error) ==
              ConversationDiskStatus::ok,
          "a fresh store opens");

    // Seed [0..12), append [12..24), append [24..32).  The merged read-back must equal the full
    // image at 32 tokens.
    const std::string name = conversation_disk_name(std::vector<int32_t>(all_ids().begin(), all_ids().begin() + 32), {}, true);
    ConversationDiskRecord seed;
    seed.stages.push_back(make_delta_stage(full_image(12), 0, 4));
    check(store.seed(name, std::move(seed), "", 0, error) == ConversationDiskStatus::ok, "a seed parks");
    uint64_t cover = 0;
    check(store.cover(name, cover) && cover == 12, "the seed covers 12 tokens");
    check(store.records() == 1, "one chain after the seed");

    ConversationDiskRecord merged;
    check(store.get(name, merged, error) == ConversationDiskStatus::ok, "the seed reads back");
    check(merged.stages[0].live.ids.size() == 12, "the seed merges to 12 ids");

    ConversationDiskRecord d12;
    d12.stages.push_back(make_delta_stage(full_image(24), 12, 4));
    check(store.append(name, std::move(d12), error) == ConversationDiskStatus::ok, "an append parks");
    ConversationDiskRecord d24;
    d24.stages.push_back(make_delta_stage(full_image(32), 24, 4));
    check(store.append(name, std::move(d24), error) == ConversationDiskStatus::ok, "the second append parks");
    check(store.cover(name, cover) && cover == 32, "the chain covers 32 tokens");
    check(store.records() == 1, "still one chain after appends");

    check(store.get(name, merged, error) == ConversationDiskStatus::ok, "the chain reads back");
    check(merged.stages[0].live.ids ==
              std::vector<int32_t>(all_ids().begin(), all_ids().begin() + 32),
          "the merged ids are the full chain");
    check(same_record(merged, [&] {
        ConversationDiskRecord full;
        full.stages.push_back(full_image(32));
        return full;
    }()), "the merged image equals the full capture");

    // best() returns the longest stored prefix.
    ConversationDiskRecord short_seed;
    short_seed.stages.push_back(make_delta_stage(full_image(12), 0, 4));
    const std::string short_name = conversation_disk_name(std::vector<int32_t>(all_ids().begin(), all_ids().begin() + 12), {}, true);
    check(store.seed(short_name, std::move(short_seed), "", 0, error) == ConversationDiskStatus::ok,
          "a second seed");
    ConversationDiskMatch m;
    check(store.best(std::vector<int32_t>(all_ids().begin(), all_ids().begin() + 20), {}, true, m), "best finds a prefix");
    check(m.tokens == 12 && (m.name == short_name || m.name == name), "best returns a 12-token chain");

    check(store.remove(name, error) == ConversationDiskStatus::ok, "the chain removes");
    check(store.remove(short_name, error) == ConversationDiskStatus::ok, "the second chain removes");
    check(store.records() == 0, "the store is empty after removal");
    std::filesystem::remove_all(dir);
}

void test_root_base() {
    std::filesystem::path dir = scratch_dir("root");
    ConversationDiskStore store;
    std::string error;
    check(store.open(options_for(dir, ConversationDiskIdentity::from_string("test"), 1ull << 30), error) ==
              ConversationDiskStatus::ok,
          "a fresh store opens");

    // The shared system prompt: a self-contained 'p' chain at 8 tokens.
    const std::string pname = conversation_disk_prefix_name(std::vector<int32_t>(all_ids().begin(), all_ids().begin() + 8), {}, true);
    ConversationDiskRecord root_rec;
    root_rec.stages.push_back(make_delta_stage(full_image(8), 0, 4));
    check(store.seed(pname, std::move(root_rec), "", 0, error) == ConversationDiskStatus::ok, "the root parks");

    // Two chats sharing the root: each seeds from the root's coverage and appends its own tail.
    const std::string cname1 = conversation_disk_name(std::vector<int32_t>(all_ids().begin(), all_ids().begin() + 16), {}, true);
    const std::string cname2 = conversation_disk_name(std::vector<int32_t>(all_ids().begin(), all_ids().begin() + 12), {}, true);
    ConversationDiskRecord seed1;
    seed1.stages.push_back(make_delta_stage(full_image(16), 8, 4));
    ConversationDiskRecord seed2;
    seed2.stages.push_back(make_delta_stage(full_image(12), 8, 4));
    check(store.seed(cname1, std::move(seed1), pname, 8, error) == ConversationDiskStatus::ok,
          "a chat seeds on the root base");
    check(store.seed(cname2, std::move(seed2), pname, 8, error) == ConversationDiskStatus::ok,
          "the second chat seeds");

    // (At these small scales the per-chat delta and the root's snapshot are comparable in bytes -
    // the real dedup signal is the content below: a root-backed chat must MERGE to its full image
    // without ever holding the root's K/V, and the eviction test pins the referenced root.)

    ConversationDiskRecord back;
    check(store.get(cname1, back, error) == ConversationDiskStatus::ok, "a root-backed chat reads back");
    check(back.stages[0].live.ids.size() == 16, "the merged chat has the full ids");
    check(same_record(back, [&] {
        ConversationDiskRecord full;
        full.stages.push_back(full_image(16));
        return full;
    }()), "the merged chat equals its full capture");

    // A 9-token request matches the root at 8 (a record may never cover the whole prompt).
    ConversationDiskMatch m;
    check(store.best(std::vector<int32_t>(all_ids().begin(), all_ids().begin() + 9), {}, true, m),
          "a 9-token request matches a stored prefix");
    check(m.tokens == 8 && m.name == pname, "the root wins the equal-length tie");
    check(store.records() == 3, "root + two chats");
    std::filesystem::remove_all(dir);
}

void test_torn_tail() {
    std::filesystem::path dir = scratch_dir("torn");
    ConversationDiskStore store;
    std::string error;
    check(store.open(options_for(dir, ConversationDiskIdentity::from_string("test"), 1ull << 30), error) ==
              ConversationDiskStatus::ok,
          "a fresh store opens");
    const std::string name = conversation_disk_name(std::vector<int32_t>(all_ids().begin(), all_ids().begin() + 32), {}, true);
    ConversationDiskRecord seed;
    seed.stages.push_back(make_delta_stage(full_image(12), 0, 4));
    check(store.seed(name, std::move(seed), "", 0, error) == ConversationDiskStatus::ok, "the seed parks");
    ConversationDiskRecord d12;
    d12.stages.push_back(make_delta_stage(full_image(24), 12, 4));
    check(store.append(name, std::move(d12), error) == ConversationDiskStatus::ok, "the first append");
    ConversationDiskRecord d24;
    d24.stages.push_back(make_delta_stage(full_image(32), 24, 4));
    check(store.append(name, std::move(d24), error) == ConversationDiskStatus::ok, "the second append");

    // Tear the LAST panel's payload (one byte in its middle) and re-read: the store truncates
    // back to the last complete panel and serves the chain at 24 tokens.
    const std::filesystem::path file = dir / (name + ".conversation");
    std::FILE* f = std::fopen(file.string().c_str(), "rb");
    check(f != nullptr, "the chain file opens");
    uint64_t pos = 128;
    uint64_t payload_at = 0;
    uint64_t meta = 0, payload = 0;
    for (int k = 0; k < 3; ++k) {
        uint64_t magic, base, end, mck, pck;
        check(std::fseek(f, (long) pos, SEEK_SET) == 0, "the scan seeks to a panel");
        check(std::fread(&magic, 8, 1, f) == 1 && std::fread(&base, 8, 1, f) == 1 &&
                  std::fread(&end, 8, 1, f) == 1 && std::fread(&meta, 8, 1, f) == 1 &&
                  std::fread(&mck, 8, 1, f) == 1 && std::fread(&payload, 8, 1, f) == 1 &&
                  std::fread(&pck, 8, 1, f) == 1,
              "panel headers read");
        pos += 56 + meta + payload;
        if (k == 2) payload_at = pos - payload;   // the last panel's payload start
    }
    std::fclose(f);
    patch_byte(file, payload_at + 100, 0x5A);
    ConversationDiskRecord back;
    check(store.get(name, back, error) == ConversationDiskStatus::ok, "a torn tail recovers on read");
    check(back.stages[0].live.ids ==
              std::vector<int32_t>(all_ids().begin(), all_ids().begin() + 24),
          "the recovered ids are the pre-tear chain");
    uint64_t cover = 0;
    check(store.cover(name, cover) && cover == 24, "the coverage reverts to the last good panel");
    store.remove(name, error);
    std::filesystem::remove_all(dir);
}

void test_rewind() {
    std::filesystem::path dir = scratch_dir("rewind");
    ConversationDiskStore store;
    std::string error;
    check(store.open(options_for(dir, ConversationDiskIdentity::from_string("test"), 1ull << 30), error) ==
              ConversationDiskStatus::ok,
          "a fresh store opens");
    const std::string name = conversation_disk_name(std::vector<int32_t>(all_ids().begin(), all_ids().begin() + 32), {}, true);
    ConversationDiskRecord seed;
    seed.stages.push_back(make_delta_stage(full_image(12), 0, 4));
    check(store.seed(name, std::move(seed), "", 0, error) == ConversationDiskStatus::ok, "the seed parks");
    ConversationDiskRecord d12;
    d12.stages.push_back(make_delta_stage(full_image(24), 12, 4));
    check(store.append(name, std::move(d12), error) == ConversationDiskStatus::ok, "the first append");
    ConversationDiskRecord d24;
    d24.stages.push_back(make_delta_stage(full_image(32), 24, 4));
    check(store.append(name, std::move(d24), error) == ConversationDiskStatus::ok, "the second append");

    uint64_t cover = 0;
    const uint64_t original_bytes = store.bytes();
    check(store.rewind_chain(name, 8, cover, error) == ConversationDiskStatus::miss && cover == 0,
          "a rewind below the seed requests a new record");
    check(store.cover(name, cover) && cover == 32 && store.bytes() == original_bytes,
          "a rewind below the seed preserves the original chain");
    ConversationDiskRecord preserved;
    check(store.get(name, preserved, error) == ConversationDiskStatus::ok,
          "the original conversation remains loadable after a below-seed rewind");
    if (!preserved.stages.empty()) {
        check(preserved.stages[0].live.ids ==
                  std::vector<int32_t>(all_ids().begin(), all_ids().begin() + 32),
              "the preserved conversation retains its full token history");
        std::vector<uint8_t> key_bytes(preserved.stages[0].kv.front().k.size());
        check(preserved.stages[0].kv.front().k.read(key_bytes.data(), 0, key_bytes.size()) &&
                  key_bytes == std::vector<uint8_t>((size_t) (aligned(32, 8) * 4), 77),
              "the preserved conversation retains its original K/V payload");
    }
    check(store.rewind_chain(name, 15, cover, error) == ConversationDiskStatus::ok && cover == 12,
          "a rewind to 15 truncates at the last panel end <= 15 (12)");
    check(store.cover(name, cover) && cover == 12, "the chain covers 12 after the rewind");

    // The editor rewrote the tail: a fresh lineage from 12 onward (a different run-state).
    SavedConversation redo = full_image(20, 90);
    redo.live.used = 7;
    ConversationDiskRecord redo_rec;
    redo_rec.stages.push_back(make_delta_stage(redo, 12, 4));
    check(store.append(name, std::move(redo_rec), error) == ConversationDiskStatus::ok, "the rewrite appends");
    ConversationDiskRecord back;
    check(store.get(name, back, error) == ConversationDiskStatus::ok, "the rewound chain reads back");
    check(back.stages[0].live.ids ==
              std::vector<int32_t>(all_ids().begin(), all_ids().begin() + 20),
          "the rewound chain holds [0..20)");
    check(back.stages[0].live.used == 7, "the merged live state is the rewrite's");
    check(back.stages[0].kv.front().k.size() == (size_t) (aligned(20, 8) * 4),
          "the merged K/V is back to the full extent");
    store.remove(name, error);
    std::filesystem::remove_all(dir);
}

void test_eviction_order() {
    std::filesystem::path dir = scratch_dir("evict");
    std::string error;
    // Phase 1: root + two root-backed chats, in a generous store.
    uint64_t budget = 1ull << 30;
    ConversationDiskStore store;
    check(store.open(options_for(dir, ConversationDiskIdentity::from_string("test"), budget), error) ==
              ConversationDiskStatus::ok,
          "a fresh store opens");
    const std::string pname = conversation_disk_prefix_name(std::vector<int32_t>(all_ids().begin(), all_ids().begin() + 8), {}, true);
    ConversationDiskRecord root_rec;
    root_rec.stages.push_back(make_delta_stage(full_image(8), 0, 4));
    check(store.seed(pname, std::move(root_rec), "", 0, error) == ConversationDiskStatus::ok, "the root parks");
    const uint64_t root_bytes = stored_bytes(dir, pname);

    auto chat_seed = [&](size_t upto, const char* suffix, std::string& cname, uint8_t fill) {
        SavedConversation full = full_image(upto, fill);
        ConversationDiskRecord rec;
        rec.stages.push_back(make_delta_stage(full, 8, 4));
        cname = conversation_disk_name(std::vector<int32_t>(all_ids().begin(), all_ids().begin() + (int64_t) upto),
                                       {}, true) + std::string(suffix);
        check(store.seed(cname, std::move(rec), pname, 8, error) == ConversationDiskStatus::ok,
              "a root-backed chat seeds");
    };
    std::string c1, c2;
    chat_seed(12, "-a", c1, 80);
    chat_seed(12, "-b", c2, 81);
    const uint64_t chat_bytes = stored_bytes(dir, c1);
    check(store.records() == 3, "root + two chats fit");

    // Phase 2: reopen the same directory with a budget that fits the root + ONE chat + the
    // forcing self-contained chat.  Seeding the force then evicts the OLDEST unpinned chat;
    // the referenced root stays.
    ConversationDiskStore tight;
    const uint64_t tight_budget = root_bytes + chat_bytes * 2 + 1;
    check(tight.open(options_for(dir, ConversationDiskIdentity::from_string("test"), tight_budget), error) ==
              ConversationDiskStatus::ok,
          "a snug store reopens");
    check(tight.records() == 3, "the snug store re-indexed everything");
    SavedConversation other = full_image(6, 82);
    ConversationDiskRecord other_rec;
    other_rec.stages.push_back(make_delta_stage(other, 0, 4));
    std::string oname = "c-other-" + std::to_string((unsigned long) getpid());
    check(tight.seed(oname, std::move(other_rec), "", 0, error) == ConversationDiskStatus::ok,
          "the forcing chat seeds");
    check(tight.has(oname), "the forcing chat is in");
    check(!tight.has(c1), "the oldest unpinned chat was evicted to make room");
    check(tight.has(pname), "the referenced root is still pinned");
    std::filesystem::remove_all(dir);
}

void test_reopen_and_purge() {
    std::filesystem::path dir = scratch_dir("reopen");
    std::string error;
    {
        ConversationDiskStore store;
        check(store.open(options_for(dir, ConversationDiskIdentity::from_string("test"), 1ull << 30), error) ==
                  ConversationDiskStatus::ok,
              "a fresh store opens");
        const std::string name = conversation_disk_name(std::vector<int32_t>(all_ids().begin(), all_ids().begin() + 32), {}, true);
        ConversationDiskRecord seed;
        seed.stages.push_back(make_delta_stage(full_image(12), 0, 4));
        check(store.seed(name, std::move(seed), "", 0, error) == ConversationDiskStatus::ok, "the seed parks");
        ConversationDiskRecord d12;
        d12.stages.push_back(make_delta_stage(full_image(24), 12, 4));
        check(store.append(name, std::move(d12), error) == ConversationDiskStatus::ok, "an append parks");
    }
    // An invalid old-format file is purged at open.
    std::filesystem::path garbage = dir / ("oldbad.conversation");
    {
        std::FILE* f = std::fopen(garbage.string().c_str(), "wb");
        check(f != nullptr, "the garbage file opens");
        for (int i = 0; i < 64; ++i) std::fwrite("garbage!", 1, 8, f);
        std::fclose(f);
    }
    ConversationDiskStore store;
    check(store.open(options_for(dir, ConversationDiskIdentity::from_string("test"), 1ull << 30), error) ==
              ConversationDiskStatus::ok,
          "the store reopens the directory");
    check(store.records() == 1, "one valid chain after reopen");
    check(!std::filesystem::exists(garbage), "the garbage file was purged");
    for (auto& p : std::filesystem::directory_iterator(dir)) {
        const std::string filename = p.path().filename().string();
        const size_t suffix = std::strlen(ConversationDiskStore::file_suffix);
        if (filename.size() <= suffix ||
            filename.compare(filename.size() - suffix, suffix, ConversationDiskStore::file_suffix) != 0)
            continue;   // the chain's sidecar is not a chain
        const std::string name = filename.substr(0, filename.size() - suffix);
        ConversationDiskRecord back;
        check(store.get(name, back, error) == ConversationDiskStatus::ok, "the reopened chain reads back");
        check(back.stages[0].live.ids.size() == 24, "the reopened chain keeps both panels");
    }
    std::filesystem::remove_all(dir);
}

void test_two_stages() {
    std::filesystem::path dir = scratch_dir("stages");
    ConversationDiskStore store;
    std::string error;
    // The running states live in the chain's sidecar, whose slots are bounded (the earliest anchor,
    // the newest, and a few log-spaced ones).  This chain's anchors are its three panel ends plus
    // two captured positions on the first panel, so the policy retains every one of them.
    ConversationDiskOptions options =
        options_for(dir, ConversationDiskIdentity::from_string("test"), 1ull << 30);
    options.max_checkpoint_states = 16;
    check(store.open(options, error) == ConversationDiskStatus::ok, "a fresh store opens");
    // A layer split: two stages with separate carves and run-states.
    auto make2 = [&](size_t upto, int64_t first, uint8_t fill) {
        ConversationDiskRecord rec;
        SavedConversation full0 = full_image(upto, fill);
        full0.layer_lo = 0;
        full0.layer_hi = 20;
        SavedConversation full1 = full_image(upto, static_cast<uint8_t>(fill + 10));
        full1.layer_lo = 20;
        full1.layer_hi = 40;
        rec.stages.push_back(make_delta_stage(full0, first, 4));
        rec.stages.push_back(make_delta_stage(full1, first, 4));
        // Only the first panel carries captured positions: the sidecar's slots are bounded, and this
        // chain must keep every anchor it publishes so each one can be checked exactly.
        for (size_t si = 0; si < rec.stages.size() && first == 0; ++si)
            for (size_t position : {(size_t) first + 4, upto - 2}) {
                ConversationCheckpoint checkpoint =
                    full_image(position, static_cast<uint8_t>(fill + si * 10 + position)).live;
                checkpoint.used = position;
                rec.stages[si].checkpoints.push_back(std::move(checkpoint));
            }
        return rec;
    };
    const std::string name = conversation_disk_name(std::vector<int32_t>(all_ids().begin(), all_ids().begin() + 24), {}, true);
    check(store.seed(name, make2(12, 0, 60), "", 0, error) == ConversationDiskStatus::ok,
          "a two-stage seed parks");
    check(store.append(name, make2(24, 12, 61), error) == ConversationDiskStatus::ok, "a two-stage append parks");
    check(store.append(name, make2(32, 24, 62), error) == ConversationDiskStatus::ok,
          "a third two-stage panel parks");
    ConversationDiskRecord back;
    check(store.get(name, back, error) == ConversationDiskStatus::ok, "a two-stage chain reads back");
    check(back.stages.size() == 2, "both stages merge");
    check(back.stages[0].live.ids == back.stages[1].live.ids, "the stages share the token chain");
    check(back.stages[0].live.ids.size() == 32, "each stage has the full ids");
    // A panel stores neither a running state nor a repeated position: the anchors are the sidecar's.
    // Every one of them must come back with its own token prefix and its own exact state.
    auto anchor = [](size_t tokens, uint8_t fill, uint64_t used) {
        ConversationCheckpoint checkpoint = full_image(tokens, fill).live;
        checkpoint.used = used;
        return checkpoint;
    };
    for (size_t si = 0; si < back.stages.size(); ++si) {
        const uint8_t fill = static_cast<uint8_t>(60 + si * 10);
        const std::vector<ConversationCheckpoint> wanted = {
            anchor(12, fill, 0), anchor(24, static_cast<uint8_t>(fill + 1), 0),
            anchor(32, static_cast<uint8_t>(fill + 2), 0),
            anchor(4, static_cast<uint8_t>(fill + 4), 4),
            anchor(10, static_cast<uint8_t>(fill + 10), 10)};
        for (const auto& want : wanted) {
            const ConversationCheckpoint* got = nullptr;
            for (const auto& checkpoint : back.stages[si].checkpoints)
                if (checkpoint.ids == want.ids) {
                    got = &checkpoint;
                    break;
                }
            check(got != nullptr, "every panel anchor survives in the sidecar");
            if (got != nullptr)
                check(same_checkpoint(*got, want),
                      "each anchor keeps its own token prefix and running-state payload");
        }
        check(same_checkpoint(back.stages[si].live, wanted[2]),
              "the merged running state is the last panel's own end state");
    }
    // Reuse a checkpoint inside the first panel, including a shorter replacement
    // whose buffers must not receive the discarded suffix of the older panels.
    const uint64_t before = store.bytes();
    uint64_t retained = 0;
    check(store.rewind_chain(name, 10, retained, error) == ConversationDiskStatus::ok && retained == 10,
          "a panel-interior checkpoint retains the exact unchanged prefix");
    check(store.bytes() == before && store.records() == 1,
          "preparing a tail replacement does not copy or remove the stored prefix");
    ConversationDiskRecord replacement = make2(20, 10, 90);
    for (auto& stage : replacement.stages) {
        for (auto& id : stage.live.ids) id += 1000;
        for (auto& checkpoint : stage.checkpoints)
            for (size_t i = 10; i < checkpoint.ids.size(); ++i) checkpoint.ids[i] += 1000;
    }
    check(store.append(name, std::move(replacement), error) == ConversationDiskStatus::ok,
          "a shorter changed tail appends to the original chain");
    check(store.records() == 1 && store.bytes() > before,
          "a replacement retains one chain rather than a full independent seed");
    ConversationDiskStore reopened;
    check(reopened.open(options, error) == ConversationDiskStatus::ok,
          "overlapping tail panels survive a restart");
    check(reopened.get(name, back, error) == ConversationDiskStatus::ok && back.stages.size() == 2,
          "a shorter replacement restores both stages");
    for (size_t si = 0; si < back.stages.size(); ++si) {
        std::vector<int32_t> ids(all_ids().begin(), all_ids().begin() + 20);
        for (size_t i = 10; i < ids.size(); ++i) ids[i] += 1000;
        check(back.stages[si].live.ids == ids, "the restored history contains only the replacement tail");
        std::vector<uint8_t> actual(back.stages[si].kv[0].k.size());
        check(back.stages[si].kv[0].k.read(actual.data(), 0, actual.size()),
              "the replaced key buffer can be read");
        std::vector<uint8_t> wanted(actual.size(), (uint8_t) (90 + si * 10));
        std::fill(wanted.begin(), wanted.begin() + 8 * 4, (uint8_t) (60 + si * 10));
        check(actual == wanted, "the original key prefix and replacement overlap page remain byte exact");
        check(same_checkpoint(back.stages[si].live, full_image(20, (uint8_t) (90 + si * 10)).live),
              "the merged running state comes from the replacement panel");
        const auto& kv = back.stages[si].kv[0];
        const ConversationBuffer* buffers[] = {&kv.v, &kv.k_scale, &kv.v_scale, &kv.pooled};
        const size_t prefix_bytes[] = {32, 16, 16, 64};
        for (size_t e = 0; e < 4; ++e) {
            std::vector<uint8_t> values(buffers[e]->size());
            std::vector<uint8_t> expected_values(values.size(), (uint8_t) (91 + si * 10 + e));
            std::fill(expected_values.begin(), expected_values.begin() + prefix_bytes[e],
                      (uint8_t) (61 + si * 10 + e));
            check(buffers[e]->read(values.data(), 0, values.size()) && values == expected_values,
                  "all value, scale and indexer buffers retain the prefix and replace the tail");
        }
        for (const auto& checkpoint : back.stages[si].checkpoints)
            check(checkpoint.ids.size() <= ids.size() &&
                      std::equal(checkpoint.ids.begin(), checkpoint.ids.end(), ids.begin()),
                  "discarded branch checkpoints do not survive the replacement");
    }
    const uint64_t delta_bytes = store.bytes() - before;
    const uint64_t stored_bytes = reopened.bytes();
    const std::string full_name = conversation_disk_name(
        std::vector<int32_t>(all_ids().begin(), all_ids().begin() + 20), {}, true);
    check(reopened.seed(full_name, make2(20, 0, 90), "", 0, error) == ConversationDiskStatus::ok,
          "a full seed provides the offload byte-size reference");
    const uint64_t full_bytes = reopened.bytes() - stored_bytes;
    check(delta_bytes < full_bytes, "a panel-interior replacement writes fewer bytes than a full seed");
    std::printf("panel-interior replacement: %llu bytes appended versus %llu bytes for a full seed\n",
                (unsigned long long) delta_bytes, (unsigned long long) full_bytes);
    store.remove(name, error);
    std::filesystem::remove_all(dir);
}

void test_delta_checkpoint_below_base() {
    // The store rejects any delta whose checkpoint list includes a position at or under the park's
    // base: such a checkpoint is already inside the chain (or the base 'p' chain).  generate.cpp
    // used to send the whole session checkpoint list, so a root-backed chat that kept an early
    // checkpoint under the root base failed every park - and re-captured the whole K/V slice on
    // every request.  park_disk now filters these out; this test pins the store side of the
    // contract so a regression there fails fast.
    std::filesystem::path dir = scratch_dir("ckptbase");
    ConversationDiskStore store;
    std::string error;
    check(store.open(options_for(dir, ConversationDiskIdentity::from_string("test"), 1ull << 30), error) ==
              ConversationDiskStatus::ok,
          "a fresh store opens");

    // The shared system prompt: a self-contained 'p' chain at 8 tokens.
    const std::string pname = conversation_disk_prefix_name(
        std::vector<int32_t>(all_ids().begin(), all_ids().begin() + 8), {}, true);
    ConversationDiskRecord root_rec;
    root_rec.stages.push_back(make_delta_stage(full_image(8), 0, 4));
    check(store.seed(pname, std::move(root_rec), "", 0, error) == ConversationDiskStatus::ok, "the root parks");

    // A chat that seeds on the root base at 8 but carries a checkpoint at 4 (inside the base).
    const std::string cname = conversation_disk_name(
        std::vector<int32_t>(all_ids().begin(), all_ids().begin() + 20), {}, true);
    ConversationDiskRecord bad_seed;
    bad_seed.stages.push_back(make_delta_stage(full_image(20), 8, 4));
    ConversationCheckpoint ck0;
    ck0.ids.assign(all_ids().begin(), all_ids().begin() + 4);
    bad_seed.stages[0].checkpoints.push_back(std::move(ck0));
    check(store.seed(cname, std::move(bad_seed), pname, 8, error) == ConversationDiskStatus::invalid &&
              error.find("an invalid checkpoint in the delta") != std::string::npos,
          "a seed with a checkpoint under the base is rejected");

    // An append whose delta carries the chain's own coverage checkpoint (at 20) is rejected too.
    const std::string cname2 = conversation_disk_name(
        std::vector<int32_t>(all_ids().begin(), all_ids().begin() + 28), {}, true);
    ConversationDiskRecord good_seed;
    good_seed.stages.push_back(make_delta_stage(full_image(20), 8, 4));
    check(store.seed(cname2, std::move(good_seed), pname, 8, error) == ConversationDiskStatus::ok,
          "a clean chat seeds on the root base");
    ConversationDiskRecord bad_append;
    bad_append.stages.push_back(make_delta_stage(full_image(28), 20, 4));
    ConversationCheckpoint ck1;
    ck1.ids.assign(all_ids().begin(), all_ids().begin() + 20);
    bad_append.stages[0].checkpoints.push_back(std::move(ck1));
    check(store.append(cname2, std::move(bad_append), error) == ConversationDiskStatus::invalid &&
              error.find("an invalid checkpoint in the delta") != std::string::npos,
          "an append with an already-merged checkpoint is rejected");

    // The rejected deltas never touched the stored chain.
    ConversationDiskRecord back;
    check(store.get(cname2, back, error) == ConversationDiskStatus::ok, "the clean chain still reads back");
    check(back.stages[0].live.ids.size() == 20, "the clean chain has the full ids");

    // The same checkpoint correctly placed ABOVE the base parks.
    ConversationDiskRecord good_append;
    good_append.stages.push_back(make_delta_stage(full_image(28), 20, 4));
    ConversationCheckpoint ck2;
    ck2.ids.assign(all_ids().begin(), all_ids().begin() + 24);
    good_append.stages[0].checkpoints.push_back(std::move(ck2));
    check(store.append(cname2, std::move(good_append), error) == ConversationDiskStatus::ok,
          "an append with a checkpoint above the base parks");
    std::filesystem::remove_all(dir);
}

// ---- Branch chains: one stored prefix, one divergent suffix per branch. ----

// The 128-byte chain header's fields, mirroring conversation_disk.cpp.
constexpr uint64_t kHeaderBaseTokens = 72;
constexpr uint64_t kHeaderBaseName = 80;
constexpr uint64_t kHeaderBasePrefixChecksum = 120;

// The prefix a branch borrows from `full` at `tokens` tokens.
ConversationDiskBase borrow_prefix(const SavedConversation& full, uint64_t tokens) {
    ConversationDiskBase base;
    base.tokens = tokens;
    base.ids.assign(full.live.ids.begin(), full.live.ids.begin() + (std::ptrdiff_t) tokens);
    for (const auto& image : full.live.imgs) {
        if ((uint64_t) image.start >= tokens) continue;
        base.imgs.push_back(image);
    }
    return base;
}

ConversationCheckpoint checkpoint_at(const std::vector<int32_t>& ids, size_t upto, uint8_t fill) {
    ConversationCheckpoint checkpoint = full_image(ids, upto, fill).live;
    checkpoint.used = upto;
    return checkpoint;
}
ConversationCheckpoint checkpoint_at(size_t upto, uint8_t fill) {
    return checkpoint_at(all_ids(), upto, fill);
}

ConversationDiskRecord record_of(size_t upto, uint8_t fill = 77) {
    ConversationDiskRecord record;
    record.stages.push_back(full_image(upto, fill));
    return record;
}

// The record name the engine gives a chain whose whole content is `upto` tokens of all_ids().
std::string ids_name(size_t upto, const char* suffix = "") {
    return conversation_disk_name(
        std::vector<int32_t>(all_ids().begin(), all_ids().begin() + (std::ptrdiff_t) upto), {}, true) +
        suffix;
}

std::vector<uint8_t> read_file(const std::filesystem::path& path) {
    std::FILE* file = std::fopen(path.string().c_str(), "rb");
    check(file != nullptr, "a chain file opens for a byte read");
    std::vector<uint8_t> bytes;
    uint8_t chunk[8192];
    size_t got = 0;
    while ((got = std::fread(chunk, 1, sizeof(chunk), file)) != 0)
        bytes.insert(bytes.end(), chunk, chunk + got);
    std::fclose(file);
    return bytes;
}

void patch_u64(const std::filesystem::path& path, uint64_t offset, uint64_t value) {
    std::FILE* file = std::fopen(path.string().c_str(), "r+b");
    check(file != nullptr, "a patched record opens");
    check(std::fseek(file, static_cast<long>(offset), SEEK_SET) == 0 &&
              std::fwrite(&value, 8, 1, file) == 1,
          "a patched record writes");
    std::fclose(file);
}

// The source chain of the branch tests: three panels to 32 tokens, with a running state at 20.
std::string seed_source_chain(ConversationDiskStore& store, std::string& error) {
    const std::string source = ids_name(32);
    ConversationDiskRecord seed;
    seed.stages.push_back(make_delta_stage(full_image(12), 0, 4));
    check(store.seed(source, std::move(seed), "", 0, error) == ConversationDiskStatus::ok,
          "the source chain seeds");
    ConversationDiskRecord d12;
    d12.stages.push_back(make_delta_stage(full_image(24), 12, 4));
    d12.stages[0].checkpoints.push_back(checkpoint_at(20, 60));
    check(store.append(source, std::move(d12), error) == ConversationDiskStatus::ok,
          "the source chain appends to 24");
    ConversationDiskRecord d24;
    d24.stages.push_back(make_delta_stage(full_image(32), 24, 4));
    check(store.append(source, std::move(d24), error) == ConversationDiskStatus::ok,
          "the source chain appends to 32");
    return source;
}

void test_branch_prefix_reuse() {
    const std::filesystem::path dir = scratch_dir("branch");
    ConversationDiskStore store;
    std::string error;
    check(store.open(options_for(dir, ConversationDiskIdentity::from_string("branch"), 1ull << 30), error) ==
              ConversationDiskStatus::ok,
          "the branch store opens");
    const std::string source = seed_source_chain(store, error);
    const std::vector<uint8_t> source_bytes = read_file(dir / (source + ".conversation"));

    // The branch borrows the source's first 12 tokens and stores [12, 20) only.
    const std::string branch = ids_name(20);
    ConversationDiskBase base = borrow_prefix(full_image(20), 12);
    base.name = source;
    ConversationDiskRecord branch_seed;
    branch_seed.stages.push_back(make_delta_stage(full_image(20), 12, 4));
    check(store.seed(branch, std::move(branch_seed), base, error) == ConversationDiskStatus::ok,
          "the branch seeds on the source's prefix");
    check(store.records() == 2, "the branch is a second chain");
    ConversationDiskRecord back;
    check(store.get(branch, back, error) == ConversationDiskStatus::ok && same_record(back, record_of(20)),
          "the merged branch equals the full capture");
    // The branch stores only its divergent suffix.
    const uint64_t branch_bytes = std::filesystem::file_size(dir / (branch + ".conversation"));
    const std::string reference = ids_name(20, "-ref");
    check(store.seed(reference, record_of(20), "", 0, error) == ConversationDiskStatus::ok,
          "the reference seed parks");
    const uint64_t standalone_bytes = std::filesystem::file_size(dir / (reference + ".conversation"));
    check(store.remove(reference, error) == ConversationDiskStatus::ok, "the reference seed removes");
    check(branch_bytes < standalone_bytes, "a branch stores only its divergent suffix");
    // The source keeps its tail, byte for byte, and still reads back whole.
    check(read_file(dir / (source + ".conversation")) == source_bytes,
          "the source chain is unchanged by the branch");
    uint64_t cover = 0;
    check(store.cover(source, cover) && cover == 32, "the source still covers 32 tokens");
    check(store.get(source, back, error) == ConversationDiskStatus::ok && same_record(back, record_of(32)),
          "the source still reads back whole");
    // Appending to the branch rewrites nothing: the file's prefix bytes stay identical.
    const std::vector<uint8_t> branch_before = read_file(dir / (branch + ".conversation"));
    ConversationDiskRecord branch_append;
    branch_append.stages.push_back(make_delta_stage(full_image(24), 20, 4));
    check(store.append(branch, std::move(branch_append), error) == ConversationDiskStatus::ok,
          "the branch appends to 24");
    const std::vector<uint8_t> branch_after = read_file(dir / (branch + ".conversation"));
    check(branch_after.size() > branch_before.size() &&
              std::equal(branch_before.begin() + 128, branch_before.end(), branch_after.begin() + 128),
          "an append keeps every stored panel byte and adds only the new panel");
    check(store.get(branch, back, error) == ConversationDiskStatus::ok && same_record(back, record_of(24)),
          "the grown branch reads back whole");

    // A restart reads both chains.
    ConversationDiskStore reopened;
    check(reopened.open(options_for(dir, ConversationDiskIdentity::from_string("branch"), 1ull << 30), error) ==
              ConversationDiskStatus::ok,
          "the branch store reopens");
    check(reopened.get(branch, back, error) == ConversationDiskStatus::ok && same_record(back, record_of(24)),
          "the branch survives a restart");
    check(reopened.get(source, back, error) == ConversationDiskStatus::ok && same_record(back, record_of(32)),
          "the source survives a restart");

    // A prefix that does not match the base is rejected before any write.
    const uint64_t records = reopened.records();
    const uint64_t bytes = reopened.bytes();
    ConversationDiskBase wrong = base;
    wrong.ids[3] ^= 0x7f;
    ConversationDiskRecord bad;
    bad.stages.push_back(make_delta_stage(full_image(20), 12, 4));
    const std::string bad_name = ids_name(20, "-bad");
    check(reopened.seed(bad_name, std::move(bad), wrong, error) == ConversationDiskStatus::invalid,
          "a prefix that does not match the base is rejected");
    check(reopened.records() == records && reopened.bytes() == bytes &&
              !std::filesystem::exists(dir / (bad_name + ".conversation")),
          "a rejected prefix writes nothing");
    // A cvec mismatch, a stage-layout mismatch, and an unknown base are rejected too.
    SavedConversation other = full_image(20);
    other.cvec = false;
    ConversationDiskRecord cvec_rec;
    cvec_rec.stages.push_back(make_delta_stage(other, 12, 4));
    check(reopened.seed(ids_name(20, "-cvec"), std::move(cvec_rec), base, error) ==
              ConversationDiskStatus::invalid,
          "a cvec mismatch is rejected");
    SavedConversation reshaped = full_image(20);
    reshaped.layer_hi = 99;
    ConversationDiskRecord shape_rec;
    shape_rec.stages.push_back(make_delta_stage(reshaped, 12, 4));
    check(reopened.seed(ids_name(20, "-shape"), std::move(shape_rec), base, error) ==
              ConversationDiskStatus::invalid,
          "a stage-layout mismatch is rejected");
    ConversationDiskBase missing = base;
    missing.name = "c0-nosuch";
    ConversationDiskRecord missing_rec;
    missing_rec.stages.push_back(make_delta_stage(full_image(20), 12, 4));
    check(reopened.seed(ids_name(20, "-missing"), std::move(missing_rec), missing, error) ==
              ConversationDiskStatus::invalid,
          "an unknown base is rejected");
    check(reopened.records() == records && reopened.bytes() == bytes,
          "no rejected branch entered the store");
    std::filesystem::remove_all(dir);
}

void test_get_prefix_extent() {
    const std::filesystem::path dir = scratch_dir("prefix");
    ConversationDiskStore store;
    std::string error;
    check(store.open(options_for(dir, ConversationDiskIdentity::from_string("prefix"), 1ull << 30), error) ==
              ConversationDiskStatus::ok,
          "the prefix store opens");
    const std::string source = seed_source_chain(store, error);
    ConversationDiskRecord back;
    const std::vector<int32_t> twelve(all_ids().begin(), all_ids().begin() + 12);
    const std::vector<int32_t> twenty(all_ids().begin(), all_ids().begin() + 20);

    // A cut on a panel end: the panel's own state is the live state.
    check(store.get_prefix(source, back, 12, error) == ConversationDiskStatus::ok,
          "a panel-end prefix read succeeds");
    check(back.stages[0].live.ids == twelve, "the panel-end read trims the ids");
    check(back.stages[0].live.has_state() &&
              same_checkpoint(back.stages[0].live, full_image(12).live),
          "the panel-end read carries the panel's own state");

    // A cut inside a panel: the state at the cut comes from the checkpoint there.
    check(store.get_prefix(source, back, 20, error) == ConversationDiskStatus::ok,
          "an interior prefix read succeeds");
    check(back.stages[0].live.ids == twenty, "the interior read trims the ids");
    check(back.stages[0].live.has_state() &&
              same_checkpoint(back.stages[0].live, checkpoint_at(20, 60)),
          "the interior read promotes the checkpoint at the cut to the live state");
    check(back.stages[0].kv[0].cells == aligned(20, 8) &&
              back.stages[0].kv[0].k.size() == (size_t) (aligned(20, 8) * 4) &&
              back.stages[0].kv[0].pooled.size() == (size_t) ((20 / 4 + 1) * 32),
          "the interior read sizes every K/V buffer to the cut");
    std::vector<uint8_t> keys(back.stages[0].kv[0].k.size());
    check(back.stages[0].kv[0].k.read(keys.data(), 0, keys.size()) &&
              keys == std::vector<uint8_t>(keys.size(), 77),
          "the interior read merges the K/V below the cut byte-exactly");
    for (const auto& checkpoint : back.stages[0].checkpoints)
        check(checkpoint.ids.size() <= 20, "no checkpoint above the cut is returned");

    // A cut at the coverage equals a full read; a cut beyond it returns the chain's own extent.
    check(store.get_prefix(source, back, 32, error) == ConversationDiskStatus::ok &&
              same_record(back, record_of(32)),
          "a cut at the coverage equals a full read");
    check(store.get_prefix(source, back, 40, error) == ConversationDiskStatus::ok &&
              back.stages[0].live.ids.size() == 32,
          "a cut beyond the coverage returns the chain");
    check(store.get_prefix(source, back, 0, error) == ConversationDiskStatus::invalid,
          "a zero cut is rejected");
    check(store.get_prefix("c0-nosuch", back, 8, error) == ConversationDiskStatus::miss,
          "an unknown chain misses");
    std::filesystem::remove_all(dir);
}

// ---- The frozen v2 LEGACY chain file: a fixture the CURRENT writer can no longer produce ----
//
// A pre-sidecar build wrote every panel's running states and every checkpoint position INLINE, and
// kept no <name>.states file.  The store must keep reading those bytes forever, so the writer below
// is a FROZEN copy of the v2 panel layout: it must NEVER be updated when the store's own writer
// changes.  Its content comes from the test's capture helpers, so the fixture and the expected
// read-back have one source.
constexpr uint64_t kFixtureChainMagic = 0x4B53494443525453ull;    // "STRCDISK", little-endian
constexpr uint64_t kFixturePanelMagic = 0x53444B4C454E4150ull;    // "PANELKDS", little-endian

// One legacy panel: the v2 metadata (shape, ids, images, inline running states, section lengths)
// followed by its payload sections in the same order.
struct LegacyPanel {
    uint64_t base_tokens = 0;
    uint64_t end_tokens = 0;
    std::vector<uint8_t> metadata;
    std::vector<uint8_t> payload;
    bool stages_written = false;

    void raw(const void* source, size_t bytes) {
        const uint8_t* first = static_cast<const uint8_t*>(source);
        metadata.insert(metadata.end(), first, first + bytes);
    }
    void u8(uint8_t value) { metadata.push_back(value); }
    void u32(uint32_t value) { raw(&value, 4); }
    void u64(uint64_t value) { raw(&value, 8); }
    void i32(int32_t value) { raw(&value, 4); }
    void i64(int64_t value) { raw(&value, 8); }
    void section(const std::vector<uint8_t>& bytes) {
        u64(bytes.size());
        payload.insert(payload.end(), bytes.begin(), bytes.end());
    }
    void section(const ConversationBuffer& buffer) {
        std::vector<uint8_t> bytes(buffer.size());
        check(bytes.empty() || buffer.read(bytes.data(), 0, bytes.size()), "a fixture buffer reads");
        section(bytes);
    }
    void checkpoint(const ConversationCheckpoint& value) {
        u32(static_cast<uint32_t>(value.ids.size()));
        if (!value.ids.empty()) raw(value.ids.data(), value.ids.size() * sizeof(int32_t));
        u32(static_cast<uint32_t>(value.imgs.size()));
        for (const auto& key : value.imgs) {
            i64(key.start);
            u64(key.hash);
        }
        u64(value.used);
        section(value.gdn.vec());
        section(value.ple.vec());
        section(value.tails.vec());
        section(value.dead.vec());
        section(value.block_pos.vec());
        u32(static_cast<uint32_t>(value.stage_parts.size()));
        for (const auto& part : value.stage_parts) checkpoint(part);
    }
    // One stage block: the leading stage count, then the shape, the live state, the checkpoints and
    // the K/V tails, in the order the v2 metadata held them.
    void add(const SavedConversation& stage) {
        if (!stages_written) {
            u32(1);
            stages_written = true;
        }
        i64(stage.layer_lo);
        i64(stage.layer_hi);
        for (const int64_t value : stage.geometry) i64(value);
        u8(stage.cvec ? 1 : 0);
        checkpoint(stage.live);
        u32(static_cast<uint32_t>(stage.checkpoints.size()));
        for (const auto& value : stage.checkpoints) checkpoint(value);
        u32(static_cast<uint32_t>(stage.kv.size()));
        for (const auto& kv : stage.kv) {
            i32(kv.format);
            i64(kv.cells);
            i64(kv.heads);
            i64(kv.head_dim);
            i64(kv.page_size);
            i64(kv.pooled_rows);
            i64(kv.idx_dim);
            for (const int64_t first : kv.first_units) i64(first);
            section(kv.k);
            section(kv.v);
            section(kv.k_scale);
            section(kv.v_scale);
            section(kv.pooled);
        }
    }
};

// The frozen v2 chain file: a 128-byte chain header, then per panel a 56-byte panel header, the
// metadata and the payload.  No base reference and no sidecar.
void write_legacy_chain(const std::filesystem::path& path, const ConversationDiskIdentity& identity,
                        bool cvec, uint64_t total_tokens, const std::vector<LegacyPanel>& panels) {
    std::vector<uint8_t> bytes(128, 0);
    auto set_u32 = [&](size_t at, uint32_t value) { std::memcpy(bytes.data() + at, &value, 4); };
    auto set_u64 = [&](size_t at, uint64_t value) { std::memcpy(bytes.data() + at, &value, 8); };
    set_u64(0, kFixtureChainMagic);
    set_u32(8, ConversationDiskStore::format_version);
    set_u64(16, total_tokens);
    set_u64(24, panels.size());
    std::memcpy(bytes.data() + 32, identity.bytes.data(), ConversationDiskIdentity::size);
    bytes[64] = cvec ? 1 : 0;      // base_kind stays 0: a legacy fixture is self-contained
    for (const LegacyPanel& panel : panels) {
        const size_t at = bytes.size();
        bytes.resize(at + 56, 0);
        set_u64(at + 0, kFixturePanelMagic);
        set_u64(at + 8, panel.base_tokens);
        set_u64(at + 16, panel.end_tokens);
        set_u64(at + 24, panel.metadata.size());
        set_u64(at + 32, conversation_disk_checksum(panel.metadata.data(), panel.metadata.size()));
        set_u64(at + 40, panel.payload.size());
        set_u64(at + 48, conversation_disk_checksum(panel.payload.data(), panel.payload.size()));
        bytes.insert(bytes.end(), panel.metadata.begin(), panel.metadata.end());
        bytes.insert(bytes.end(), panel.payload.begin(), panel.payload.end());
    }
    std::FILE* file = std::fopen(path.string().c_str(), "wb");
    check(file != nullptr, "the legacy fixture opens for writing");
    check(std::fwrite(bytes.data(), 1, bytes.size(), file) == bytes.size(), "the legacy fixture writes");
    std::fclose(file);
}

// Two panels to 24 tokens, with a running state INLINE at every captured position (both panel ends
// and six checkpoints) and a real K/V payload per panel.
std::vector<LegacyPanel> legacy_fixture() {
    std::vector<LegacyPanel> panels(2);
    panels[0].base_tokens = 0;
    panels[0].end_tokens = 16;
    SavedConversation first = make_delta_stage(full_image(16), 0, 4);
    for (const size_t upto : {4u, 8u, 12u})
        first.checkpoints.push_back(checkpoint_at(upto, (uint8_t) (100 + upto)));
    panels[0].add(first);
    panels[1].base_tokens = 16;
    panels[1].end_tokens = 24;
    SavedConversation second = make_delta_stage(full_image(24), 16, 4);
    for (const size_t upto : {18u, 20u, 22u})
        second.checkpoints.push_back(checkpoint_at(upto, (uint8_t) (100 + upto)));
    panels[1].add(second);
    return panels;
}

// The running-state bytes a loaded record PHYSICALLY holds: what the bound's allocation gate rules.
size_t allocated_states(const ConversationDiskRecord& record) {
    size_t bytes = 0;
    for (const auto& stage : record.stages)
        for (const auto& checkpoint : stage.checkpoints) bytes += checkpoint.allocation_bytes();
    return bytes;
}

// A pre-existing v2 file with no sidecar: many inline anchors, real K/V payloads.  A load must keep
// every position, bound the states it ALLOCATES, return the exact state at any requested cut (even
// inside a panel), survive a restart, and detect a corrupt inline state it would not have kept.
void test_legacy_inline_chain() {
    const std::filesystem::path dir = scratch_dir("legacy");
    const ConversationDiskIdentity identity = ConversationDiskIdentity::from_string("legacy");
    const std::string name = ids_name(24);
    const std::filesystem::path chain_file = dir / (name + ConversationDiskStore::file_suffix);
    const std::filesystem::path state_file = dir / (name + ".states");
    std::filesystem::create_directories(dir);
    write_legacy_chain(chain_file, identity, true, 24, legacy_fixture());
    check(!std::filesystem::exists(state_file), "the legacy fixture has no sidecar");

    ConversationDiskOptions options = options_for(dir, identity, 1ull << 30);
    options.max_checkpoint_states = 2;
    ConversationDiskStore store;
    std::string error;
    check(store.open(options, error) == ConversationDiskStatus::ok, "the legacy store opens");
    check(store.records() == 1 && store.has(name), "the legacy chain indexes");
    check(!std::filesystem::exists(state_file), "indexing a legacy chain creates no sidecar");

    ConversationDiskRecord back;
    check(store.get(name, back, error) == ConversationDiskStatus::ok, "the legacy chain reads back");
    check(back.stages.size() == 1 &&
              back.stages[0].live.ids ==
                  std::vector<int32_t>(all_ids().begin(), all_ids().begin() + 24),
          "the legacy chain merges to its full ids");
    check(same_record(back, record_of(24)), "the legacy chain merges to its full capture");
    check(back.stages[0].checkpoints.size() == 6, "the bound keeps every captured position");
    size_t states = 0;
    for (const auto& checkpoint : back.stages[0].checkpoints) states += checkpoint.has_state() ? 1 : 0;
    check(states <= 3 && states < 6, "the bound keeps the earliest, the newest, and its own count");

    // The dropped states' bytes are never ALLOCATED - not merely released - so the read's physical
    // allocations cover the bound's states while the file carries all six.
    const size_t per_state = full_image(4).live.allocation_bytes();
    check(allocated_states(back) <= 4 * per_state,
          "the bounded read allocates only the states the bound keeps");
    ConversationDiskOptions wide = options;
    wide.max_checkpoint_states = 0;
    ConversationDiskStore unbounded;
    check(unbounded.open(wide, error) == ConversationDiskStatus::ok, "the unbounded store opens");
    ConversationDiskRecord all;
    check(unbounded.get(name, all, error) == ConversationDiskStatus::ok, "the unbounded read succeeds");
    size_t all_states = 0;
    for (const auto& checkpoint : all.stages[0].checkpoints) all_states += checkpoint.has_state() ? 1 : 0;
    check(all_states == 6, "0 keeps every inline state a legacy panel carries");
    check(allocated_states(all) >= 6 * per_state, "the unbounded read allocates every state");

    // A cut INSIDE the second panel: the checkpoint there is promoted to the live state, exactly.
    check(store.get_prefix(name, back, 20, error) == ConversationDiskStatus::ok,
          "an interior legacy prefix read succeeds");
    check(back.stages[0].live.ids == std::vector<int32_t>(all_ids().begin(), all_ids().begin() + 20),
          "the interior read trims the ids to the cut");
    check(back.stages[0].live.has_state() &&
              same_checkpoint(back.stages[0].live, checkpoint_at(20, 120)),
          "the interior read promotes the checkpoint at the cut, exactly");
    check(back.stages[0].kv[0].cells == aligned(20, 8) &&
              back.stages[0].kv[0].k.size() == (size_t) (aligned(20, 8) * 4),
          "the interior read sizes the K/V to the cut");
    std::vector<uint8_t> keys(back.stages[0].kv[0].k.size());
    check(back.stages[0].kv[0].k.read(keys.data(), 0, keys.size()) &&
              keys == std::vector<uint8_t>(keys.size(), 77),
          "the interior read merges the K/V below the cut byte-exactly");
    for (const auto& checkpoint : back.stages[0].checkpoints)
        check(checkpoint.ids.size() <= 20, "no checkpoint above the cut is returned");
    // A cut inside the FIRST panel, and one exactly on a panel end.
    check(store.get_prefix(name, back, 12, error) == ConversationDiskStatus::ok &&
              back.stages[0].live.ids.size() == 12 &&
              same_checkpoint(back.stages[0].live, checkpoint_at(12, 112)),
          "an interior cut in the first panel returns its exact state");
    check(store.get_prefix(name, back, 16, error) == ConversationDiskStatus::ok &&
              same_checkpoint(back.stages[0].live, full_image(16).live),
          "a cut on a panel end returns that panel's own state");

    // A restart re-indexes the same bytes: a legacy chain must never gain a sidecar.
    ConversationDiskStore reopened;
    check(reopened.open(options, error) == ConversationDiskStatus::ok, "the legacy store reopens");
    check(reopened.has(name) && !std::filesystem::exists(state_file),
          "a legacy chain reopens with no sidecar");
    check(reopened.get(name, back, error) == ConversationDiskStatus::ok && same_record(back, record_of(24)),
          "the legacy chain reads back after a restart");
    check(reopened.get_prefix(name, back, 20, error) == ConversationDiskStatus::ok &&
              same_checkpoint(back.stages[0].live, checkpoint_at(20, 120)),
          "the interior cut still returns its exact state after a restart");

    // A corrupt inline state is a corrupt record even when the bound would drop it: the panel's
    // payload checksum covers every byte, so no wrong state can reach a caller.
    uint64_t payload_at = 0;
    {
        std::FILE* file = std::fopen(chain_file.string().c_str(), "rb");
        check(file != nullptr, "the legacy fixture opens for the scan");
        uint64_t pos = 128;
        for (int panel = 0; panel < 2; ++panel) {
            uint64_t magic = 0, base = 0, end = 0, meta = 0, mck = 0, payload = 0, pck = 0;
            check(std::fseek(file, (long) pos, SEEK_SET) == 0 &&
                      std::fread(&magic, 8, 1, file) == 1 && std::fread(&base, 8, 1, file) == 1 &&
                      std::fread(&end, 8, 1, file) == 1 && std::fread(&meta, 8, 1, file) == 1 &&
                      std::fread(&mck, 8, 1, file) == 1 && std::fread(&payload, 8, 1, file) == 1 &&
                      std::fread(&pck, 8, 1, file) == 1,
                  "the fixture panel headers read");
            if (panel == 0) payload_at = pos + 56 + meta;
            pos += 56 + meta + payload;
        }
        std::fclose(file);
    }
    // The panel's payload opens with the live state then its three checkpoint states: this byte sits
    // in the SECOND checkpoint's state, which the bound above drops.
    patch_byte(chain_file, payload_at + 2 * per_state + 1, 0x5A);
    check(store.get(name, back, error) == ConversationDiskStatus::corrupt,
          "a corrupt inline state is detected");
    check(!store.has(name) && !std::filesystem::exists(chain_file),
          "the corrupt legacy chain is dropped");
    std::filesystem::remove_all(dir);
}

// ---- The bounded anchor sidecar: retention across many parks, pins, torn slots and restarts. ----

// A longer token chain for the retention tests (all_ids() stops at 40).
std::vector<int32_t> chain_ids(size_t count) {
    std::vector<int32_t> ids = all_ids();
    while (ids.size() < count) ids.push_back(static_cast<int32_t>(1000 + ids.size() * 7));
    return ids;
}

void test_sidecar_retention() {
    const std::filesystem::path dir = scratch_dir("states");
    const ConversationDiskIdentity identity = ConversationDiskIdentity::from_string("states");
    ConversationDiskOptions options = options_for(dir, identity, 1ull << 30);
    options.max_checkpoint_states = 2;
    ConversationDiskStore store;
    std::string error;
    check(store.open(options, error) == ConversationDiskStatus::ok, "the sidecar store opens");
    const std::vector<int32_t> ids = chain_ids(488);
    const std::string name =
        conversation_disk_name(std::vector<int32_t>(ids.begin(), ids.begin() + 8), {}, true);
    const std::filesystem::path chain_file = dir / (name + ConversationDiskStore::file_suffix);
    const std::filesystem::path state_file = dir / (name + ".states");
    const size_t per_state = full_image(4).live.allocation_bytes();
    // One header, one section-size row per stage, and the bounded slots.
    const uint64_t slot_bytes = 64 + per_state;
    ConversationDiskRecord seed;
    seed.stages.push_back(make_delta_stage(full_image(ids, 8), 0, 4));
    check(store.seed(name, std::move(seed), "", 0, error) == ConversationDiskStatus::ok,
          "the sidecar chain seeds");
    check(std::filesystem::exists(state_file), "the running states live in the chain's sidecar");
    check(std::filesystem::file_size(state_file) <= 128 + 5 * 8 + 3 * slot_bytes,
          "the sidecar's slots are bounded by the caller's state bound");

    // A hundred appends, each publishing its own end anchor and one captured position - one of them
    // PINNED.  The chain file grows by its K/V deltas; the state file must not grow per turn.
    const size_t pinned_at = 202;
    const uint8_t pinned_fill = static_cast<uint8_t>(pinned_at - 2);
    for (uint64_t base = 8; base < 408; base += 4) {
        ConversationDiskRecord panel;
        panel.stages.push_back(make_delta_stage(full_image(ids, (size_t) base + 4), (int64_t) base, 4));
        ConversationCheckpoint anchor = checkpoint_at(ids, (size_t) base + 2, static_cast<uint8_t>(base));
        anchor.pinned = (size_t) base + 2 == pinned_at;
        panel.stages[0].checkpoints.push_back(std::move(anchor));
        check(store.append(name, std::move(panel), error) == ConversationDiskStatus::ok,
              "a sidecar chain appends a panel");
    }
    uint64_t cover = 0;
    check(store.cover(name, cover) && cover == 408, "the chain covers every appended token");
    const uint64_t state_bytes = std::filesystem::file_size(state_file);
    check(state_bytes <= 128 + 5 * 8 + 4 * slot_bytes,
          "a hundred parks keep the state file bounded (the pin costs one slot)");
    for (uint64_t base = 408; base < 488; base += 4) {
        ConversationDiskRecord panel;
        panel.stages.push_back(make_delta_stage(full_image(ids, (size_t) base + 4), (int64_t) base, 4));
        panel.stages[0].checkpoints.push_back(checkpoint_at(ids, (size_t) base + 2, static_cast<uint8_t>(base)));
        check(store.append(name, std::move(panel), error) == ConversationDiskStatus::ok,
              "the chain keeps appending");
    }
    check(store.cover(name, cover) && cover == 488, "the chain reaches its last panel");
    check(std::filesystem::file_size(state_file) == state_bytes,
          "the state file never grows one state per turn");
    check(std::filesystem::file_size(chain_file) > state_bytes, "the chain grew by its K/V deltas");

    // The merged chain, the pinned anchor, and the earliest anchor: each keeps its own state.
    ConversationDiskRecord back;
    check(store.get(name, back, error) == ConversationDiskStatus::ok, "the retained chain reads back");
    check(back.stages[0].live.ids == ids, "the chain merges to its full ids");
    check(same_checkpoint(back.stages[0].live, full_image(ids, 488).live),
          "the merged running state is the last panel's own end state");
    check(store.get_prefix(name, back, pinned_at, error) == ConversationDiskStatus::ok &&
              back.stages[0].live.ids.size() == pinned_at &&
              same_checkpoint(back.stages[0].live, checkpoint_at(ids, pinned_at, pinned_fill)),
          "a pinned anchor keeps its state through every ordinary rotation");
    check(store.get_prefix(name, back, 8, error) == ConversationDiskStatus::ok &&
              same_checkpoint(back.stages[0].live, full_image(ids, 8).live),
          "the earliest anchor survives every rotation");
    // A cut the sidecar holds no anchor for returns no state - never a wrong one.
    check(store.get_prefix(name, back, 350, error) == ConversationDiskStatus::ok &&
              back.stages[0].live.ids.size() == 350 && !back.stages[0].live.has_state(),
          "a cut with no stored anchor returns no state");

    // A restart re-indexes the anchors from the sidecar: the panels carry none of them.
    ConversationDiskStore reopened;
    check(reopened.open(options, error) == ConversationDiskStatus::ok, "the sidecar store reopens");
    check(reopened.has(name), "the chain re-indexes after a restart");
    check(reopened.get(name, back, error) == ConversationDiskStatus::ok && back.stages[0].live.ids == ids,
          "the chain reads back whole after a restart");
    check(reopened.get_prefix(name, back, pinned_at, error) == ConversationDiskStatus::ok &&
              same_checkpoint(back.stages[0].live, checkpoint_at(ids, pinned_at, pinned_fill)),
          "the pinned anchor survives a restart");

    // A torn slot is never restored: the position keeps no state instead of a wrong one, and the
    // other anchors are untouched.  Slot zero holds the earliest anchor, whose payload starts after
    // the 128-byte state header and the five section sizes.
    patch_byte(state_file, 128 + 5 * 8 + 64 + per_state - 1, 0x5A);
    check(reopened.get_prefix(name, back, 8, error) == ConversationDiskStatus::ok &&
              back.stages[0].live.ids.size() == 8 && !back.stages[0].live.has_state(),
          "a torn slot leaves its position without a state");
    check(reopened.get_prefix(name, back, 488, error) == ConversationDiskStatus::ok &&
              same_checkpoint(back.stages[0].live, full_image(ids, 488).live),
          "the other anchors are untouched by a torn slot");
    std::filesystem::remove_all(dir);
}

// ---- shared_prefix: the index-only K/V borrow lookup a park uses for a branch base. ----

// A chain whose sidecar is gone still holds valid K/V, but no position it can resume from: the
// restore lookup must not offer it (a caller that picked it would discard the whole chain), while
// the borrow lookup and the append path still find it.
void test_lost_sidecar_is_not_offered() {
    const std::filesystem::path dir = scratch_dir("lost-states");
    const ConversationDiskIdentity identity = ConversationDiskIdentity::from_string("lost");
    ConversationDiskOptions options = options_for(dir, identity, 1ull << 30);
    options.max_checkpoint_states = 2;
    const std::string name = ids_name(24);
    {
        ConversationDiskStore store;
        std::string error;
        check(store.open(options, error) == ConversationDiskStatus::ok, "the lost-sidecar store opens");
        ConversationDiskRecord seed;
        seed.stages.push_back(make_delta_stage(full_image(24), 0, 4));
        seed.stages[0].checkpoints.push_back(checkpoint_at(8, 108));
        auto metadata_only = checkpoint_at(16, 116);
        metadata_only.drop_state();
        seed.stages[0].checkpoints.push_back(std::move(metadata_only));
        check(store.seed(name, std::move(seed), "", 0, error) == ConversationDiskStatus::ok,
              "the chain seeds with its anchors");
        check(std::filesystem::exists(dir / (name + ".states")), "the chain has its sidecar");
        // While the anchors exist the chain's end is a restorable point for a longer prompt.
        const std::vector<int32_t> prompt = chain_ids(32);
        ConversationDiskMatch match;
        check(store.best(prompt, {}, true, match) && match.name == name && match.tokens == 24,
              "the chain's end is offered while its sidecar holds the state");
    }
    // The sidecar is lost (a torn header, a deleted file): the panels keep their K/V but no state.
    check(std::filesystem::remove(dir / (name + ".states")), "the sidecar is removed");
    ConversationDiskStore store;
    std::string error;
    check(store.open(options, error) == ConversationDiskStatus::ok, "the store reopens without it");
    const std::vector<int32_t> prompt = chain_ids(32);
    ConversationDiskMatch match;
    check(!store.best(prompt, {}, true, match),
          "no position of a state-less chain is offered for a restore");
    check(store.shared_prefix(prompt, {}, true, match) && match.name == name &&
              match.prefix_tokens == 24,
          "the borrow lookup still finds the chain's K/V extent");
    const std::vector<int32_t> ids(chain_ids(24));
    check(store.can_append(name, ids, {}, true, 24),
          "the chain can still be appended to, which republishes its anchors");
    ConversationDiskRecord back;
    check(store.get(name, back, error) == ConversationDiskStatus::ok &&
              back.stages[0].live.ids.size() == 24 && !back.stages[0].live.has_state(),
          "the read returns the K/V with no running state at all");
    std::filesystem::remove_all(dir);
}

void test_sidecar_write_failure() {
    const auto dir = scratch_dir("state-write-failure");
    ConversationDiskStore store;
    std::string error;
    const auto options = options_for(dir, ConversationDiskIdentity::from_string("state-write-failure"), 1ull << 30);
    check(store.open(options, error) == ConversationDiskStatus::ok, "the state-write failure store opens");
    const std::string name = ids_name(24);
    const auto state_path = dir / (name + ".states");
    std::filesystem::create_directory(state_path);
    ConversationDiskRecord seed;
    seed.stages.push_back(make_delta_stage(full_image(24), 0, 4));
    check(store.seed(name, std::move(seed), "", 0, error) == ConversationDiskStatus::failed &&
              store.records() == 0 && !std::filesystem::exists(dir / (name + ".conversation")),
          "a failed sidecar publication cannot fall back to an unbounded inline-state chain");
    std::filesystem::remove_all(state_path);
    ConversationDiskRecord available;
    available.stages.push_back(make_delta_stage(full_image(24), 0, 4));
    check(store.seed(name, std::move(available), "", 0, error) == ConversationDiskStatus::ok,
          "a chain can seed after its state storage is available");
    const auto before = std::filesystem::file_size(dir / (name + ".conversation"));
    std::filesystem::remove(state_path);
    std::filesystem::create_directory(state_path);
    ConversationDiskRecord next;
    next.stages.push_back(make_delta_stage(full_image(32), 24, 4));
    check(store.append(name, std::move(next), error) == ConversationDiskStatus::failed &&
              std::filesystem::file_size(dir / (name + ".conversation")) == before,
          "a failed sidecar update cannot append an inline-state fallback or change the original K/V chain");
    ConversationDiskRecord back;
    check(store.get(name, back, error) == ConversationDiskStatus::ok &&
              back.stages[0].live.ids == full_image(24).live.ids &&
              back.stages[0].kv[0].k == full_image(24).kv[0].k,
          "the original prefix remains readable after the failed state update");
    std::filesystem::remove_all(dir);
}


void test_shared_prefix_lookup() {
    const std::filesystem::path dir = scratch_dir("shared");
    ConversationDiskStore store;
    std::string error;
    check(store.open(options_for(dir, ConversationDiskIdentity::from_string("shared"), 1ull << 30), error) ==
              ConversationDiskStatus::ok,
          "the shared-prefix store opens");
    // A long chain whose ONLY point is its own end: it holds no anchor at or below the short prompt.
    const std::vector<int32_t> long_ids = chain_ids(1000);
    const std::string source = conversation_disk_name(long_ids, {}, true);
    ConversationDiskRecord seed;
    seed.stages.push_back(make_delta_stage(full_image(long_ids, 1000), 0, 4));
    check(store.seed(source, std::move(seed), "", 0, error) == ConversationDiskStatus::ok,
          "the long chain seeds");
    const std::vector<uint8_t> source_bytes = read_file(dir / (source + ConversationDiskStore::file_suffix));
    const std::vector<int32_t> prompt(long_ids.begin(), long_ids.begin() + 100);

    // The restore lookup cannot see it: its only point is longer than the prompt.
    ConversationDiskMatch match;
    check(!store.best(prompt, {}, true, match), "best() selects no restore point for the short prompt");
    // The borrow lookup finds the shared extent regardless of the missing anchor.
    check(store.shared_prefix(prompt, {}, true, match), "shared_prefix finds the chain");
    check(match.name == source && match.prefix_tokens == 100 && match.tokens == 0,
          "shared_prefix returns the common K/V extent, not a restore point");
    std::vector<int32_t> other = prompt;
    other[0] ^= 0x7f;
    check(!store.shared_prefix(other, {}, true, match), "a diverging prompt shares no prefix");
    check(!store.shared_prefix(prompt, {}, false, match), "a cvec mismatch is not offered");

    // The borrow itself: the caller proves the bytes, then bases a branch on the extent.
    ConversationDiskRecord base;
    check(store.get_prefix(match.name, base, (uint64_t) match.prefix_tokens, error) ==
              ConversationDiskStatus::ok &&
              base.stages[0].live.ids == prompt,
          "the borrowed extent reads back");
    ConversationDiskBase borrowed = borrow_prefix(base.stages[0], 100);
    borrowed.name = match.name;
    const std::vector<int32_t> branch_ids = chain_ids(140);
    const std::string branch = conversation_disk_name(branch_ids, {}, true);
    ConversationDiskRecord branch_rec;
    branch_rec.stages.push_back(make_delta_stage(full_image(branch_ids, 140), 100, 4));
    check(store.seed(branch, std::move(branch_rec), borrowed, error) == ConversationDiskStatus::ok,
          "a branch seeds on the borrowed extent");
    ConversationDiskRecord back;
    check(store.get(branch, back, error) == ConversationDiskStatus::ok &&
              back.stages[0].live.ids == branch_ids,
          "the branch merges to its own content");
    check(store.get(source, back, error) == ConversationDiskStatus::ok &&
              back.stages[0].live.ids == long_ids &&
              read_file(dir / (source + ConversationDiskStore::file_suffix)) == source_bytes,
          "the shared source keeps every byte it had");
    std::filesystem::remove_all(dir);
}

// ---- A branch bases on the extent an ancestor really LENDS, not on one it merely covers. ----

void test_branch_collapse_extent() {
    const std::filesystem::path dir = scratch_dir("collapse");
    ConversationDiskStore store;
    std::string error;
    check(store.open(options_for(dir, ConversationDiskIdentity::from_string("collapse"), 1ull << 30), error) ==
              ConversationDiskStatus::ok,
          "the collapse store opens");
    // A root covering 16 tokens.
    const std::string root = ids_name(16);
    ConversationDiskRecord root_rec;
    root_rec.stages.push_back(make_delta_stage(full_image(16), 0, 4));
    check(store.seed(root, std::move(root_rec), "", 0, error) == ConversationDiskStatus::ok,
          "the root seeds");
    // A child that borrows only the root's first 8 tokens and then extends DIVERGENTLY to 16: the
    // root covers the child's whole extent but lends only half of it.
    std::vector<int32_t> child_ids(all_ids().begin(), all_ids().begin() + 16);
    for (size_t i = 8; i < child_ids.size(); ++i) child_ids[i] += 500;
    const std::string child = conversation_disk_name(child_ids, {}, true);
    ConversationDiskBase child_base;
    child_base.name = root;
    child_base.tokens = 8;
    child_base.ids.assign(all_ids().begin(), all_ids().begin() + 8);
    ConversationDiskRecord child_rec;
    child_rec.stages.push_back(make_delta_stage(full_image(child_ids, 16), 8, 4));
    check(store.seed(child, std::move(child_rec), child_base, error) == ConversationDiskStatus::ok,
          "the child seeds on the root's first half");

    // A new branch borrows 12 tokens.  The root covers 16 but lends only 8, so the child must be the
    // stored base: the root's tokens at [8, 12) are not the child's.
    std::vector<int32_t> content = child_ids;
    const std::vector<int32_t> tail = chain_ids(20);
    content.insert(content.end(), tail.begin() + 16, tail.begin() + 20);
    const std::string branch = conversation_disk_name(content, {}, true);
    ConversationDiskBase borrowed = borrow_prefix(full_image(child_ids, 16), 12);
    borrowed.name = child;
    ConversationDiskRecord branch_rec;
    branch_rec.stages.push_back(make_delta_stage(full_image(content, 20), 12, 4));
    check(store.seed(branch, std::move(branch_rec), borrowed, error) == ConversationDiskStatus::ok,
          "the branch seeds on the child");
    // The chain header names the child, not the wider root.
    const std::vector<uint8_t> bytes = read_file(dir / (branch + ConversationDiskStore::file_suffix));
    check(bytes.size() > kHeaderBaseName + child.size() &&
              std::memcmp(bytes.data() + kHeaderBaseName, child.data(), child.size()) == 0,
          "the branch stores the child as its base, not the wider root");
    ConversationDiskRecord back;
    check(store.get(branch, back, error) == ConversationDiskStatus::ok &&
              back.stages[0].live.ids == content,
          "the branch merges the child's divergent tokens, not the root's");
    check(store.get(child, back, error) == ConversationDiskStatus::ok &&
              back.stages[0].live.ids == child_ids,
          "the child keeps its own content");
    check(store.get(root, back, error) == ConversationDiskStatus::ok &&
              back.stages[0].live.ids == std::vector<int32_t>(all_ids().begin(), all_ids().begin() + 16),
          "the root keeps its own content");
    std::filesystem::remove_all(dir);
}

void test_incremental_spill() {
    const std::filesystem::path dir = scratch_dir("spill-append");
    ConversationDiskStore store;
    std::string error;
    check(store.open(options_for(dir, ConversationDiskIdentity::from_string("spill"), 1ull << 30), error) ==
              ConversationDiskStatus::ok,
          "the spill store opens");
    const std::string name = ids_name(12);
    ConversationDiskRecord seed;
    seed.stages.push_back(make_delta_stage(full_image(12), 0, 4));
    check(store.seed(name, std::move(seed), "", 0, error) == ConversationDiskStatus::ok,
          "the conversation chain seeds");
    const uint64_t records = store.records();
    const uint64_t bytes = store.bytes();

    // The same conversation, grown to 24 tokens: the spill appends the delta only.
    uint64_t written = 0;
    check(store.spill(full_image(24), false, error, &written) == ConversationDiskStatus::ok,
          "the grown image spills");
    check(written == 12, "the spill writes only the new tokens");
    check(store.records() == records, "the spill appends to the existing chain");
    uint64_t cover = 0;
    check(store.cover(name, cover) && cover == 24, "the chain covers the spilled image");
    ConversationDiskRecord back;
    check(store.get(name, back, error) == ConversationDiskStatus::ok && same_record(back, record_of(24)),
          "the appended chain reads back whole");
    const uint64_t appended_bytes = store.bytes() - bytes;
    check(appended_bytes < std::filesystem::file_size(dir / (name + ".conversation")),
          "the append wrote less than the chain now holds");

    // A repeated spill of the same content writes nothing at all.
    written = 1;
    check(store.spill(full_image(24), false, error, &written) == ConversationDiskStatus::ok && written == 0,
          "a repeated spill writes no token");
    check(store.records() == records, "a repeated spill adds no chain");

    // The same token ids with different K/V bytes must NOT append to the stored chain.
    check(store.spill(full_image(24, 55), false, error, &written) == ConversationDiskStatus::ok,
          "a different prefill spills");
    check(store.records() == records + 1, "a different prefill becomes its own chain");
    check(store.get(name, back, error) == ConversationDiskStatus::ok && same_record(back, record_of(24)),
          "the original chain keeps its own bytes");
    check(store.spill(full_image(24, 66), false, error, &written) == ConversationDiskStatus::ok,
          "another arithmetic variant is stored despite the same content-derived name");
    bool variant_restored = false;
    for (const auto& file : std::filesystem::directory_iterator(dir)) {
        if (file.path().extension() != ".conversation") continue;
        check(store.get(file.path().stem().string(), back, error) == ConversationDiskStatus::ok,
              "an arithmetic variant remains readable");
        if (back.stages[0].live.gdn.vec()[0] == 66)
            variant_restored = same_stage(back.stages[0], full_image(24, 66));
    }
    check(variant_restored, "a same-name arithmetic variant preserves its own state and K/V");
    std::filesystem::remove_all(dir);
}

void test_terminal_rewrite_retention() {
    const auto dir = scratch_dir("terminal-rewrite");
    auto options = options_for(dir, ConversationDiskIdentity::from_string("terminal-rewrite"), 1ull << 30);
    options.max_checkpoint_states = 2;
    ConversationDiskStore store;
    std::string error;
    check(store.open(options, error) == ConversationDiskStatus::ok, "terminal rewrite store opens");
    std::vector<int32_t> ids(96);
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = (int32_t) i + 1;
    const std::string name = "c-terminal-rewrite";
    ConversationDiskRecord seed;
    seed.stages.push_back(full_image(ids, 32));
    seed.stages[0].terminal_from = 24;
    seed.stages[0].checkpoints.push_back(checkpoint_at(ids, 24, 77));
    check(store.seed(name, std::move(seed), "", 0, error) == ConversationDiskStatus::ok,
          "a generated tail and its input anchor are stored");
    SavedConversation expected;
    for (size_t turn = 0; turn < 20; ++turn) {
        const size_t from = 24 + turn, prompt = from + 1;
        auto content = ids;
        for (size_t i = prompt; i < prompt + 4; ++i) content[i] = (int32_t) (9000 + turn * 4 + i);
        expected = full_image(content, prompt + 4);
        ConversationDiskRecord delta;
        delta.stages.push_back(make_delta_stage(expected, (int64_t) from, 4));
        delta.stages[0].terminal_from = (int64_t) prompt;
        check(store.replace_terminal_tail(name, from, std::move(delta), error) == ConversationDiskStatus::ok,
              "a growing input replaces only the previous generated tail");
        check(store.records() == 1, "reply rewrites do not consume additional conversation slots");
    }
    ConversationDiskStore reopened;
    check(reopened.open(options, error) == ConversationDiskStatus::ok, "rewritten chain reopens");
    ConversationDiskRecord back;
    check(reopened.get(name, back, error) == ConversationDiskStatus::ok &&
              same_stage(back.stages[0], expected),
          "restart restores the latest rewritten tokens, K/V, and running state");
    check(reopened.get_prefix(name, back, 24, error) == ConversationDiskStatus::ok &&
              same_checkpoint(back.stages[0].live, checkpoint_at(ids, 24, 77)),
          "the original input anchor survives every rewrite and restart");
    const auto state_bytes = std::filesystem::file_size(dir / (name + ".states"));
    check(state_bytes < 1024 + 4 * full_image(ids, 24).live.allocation_bytes(),
          "twenty reply rewrites retain bounded recurrent-state storage");
    ConversationDiskRecord branch;
    branch.stages.push_back(make_delta_stage(full_image(ids, 20), 12, 4));
    branch.stages[0].terminal_from = 16;
    const auto slice = branch.stages[0].live.ids;
    check(reopened.replace_terminal_tail(name, 12, std::move(branch), error) == ConversationDiskStatus::invalid &&
              branch.stages[0].live.ids == slice,
          "a historical branch refuses destructive replacement and keeps its delta available");
    check(reopened.get(name, back, error) == ConversationDiskStatus::ok && same_stage(back.stages[0], expected),
          "refusing the historical branch preserves the long continuation");
    std::filesystem::remove_all(dir);
}

void test_consumed_origin_retention() {
    const auto dir = scratch_dir("consumed-origin");
    auto options = options_for(dir, ConversationDiskIdentity::from_string("consumed-origin"), 1ull << 30);
    options.max_checkpoint_states = 2;
    ConversationDiskStore store;
    std::string error;
    check(store.open(options, error) == ConversationDiskStatus::ok, "the consumed-origin store opens");
    const auto ids = chain_ids(128);
    const std::string name = "c-consumed-origin";
    ConversationDiskRecord seed;
    seed.stages.push_back(full_image(ids, 32));
    seed.stages[0].terminal_from = 24;
    seed.stages[0].checkpoints.push_back(checkpoint_at(ids, 8, 55));
    seed.stages[0].checkpoints.push_back(checkpoint_at(ids, 24, 66));
    check(store.seed(name, std::move(seed), "", 0, error) == ConversationDiskStatus::ok,
          "the original generated end and older input anchors park");
    const auto state_bytes = std::filesystem::file_size(dir / (name + ".states"));
    for (size_t base = 32; base < 112; base += 4) {
        ConversationDiskRecord next;
        next.stages.push_back(make_delta_stage(full_image(ids, base + 4), (int64_t) base, 4));
        next.stages[0].live.gdn.vec()[0] = (uint8_t) base;
        check(store.append(name, std::move(next), error) == ConversationDiskStatus::ok,
              "a consumed original prefix extends through dense input checkpoint rotation");
    }
    ConversationDiskRecord back;
    check(store.get_prefix(name, back, 32, error) == ConversationDiskStatus::ok &&
              same_checkpoint(back.stages[0].live, full_image(ids, 32).live),
          "the original consumed continuation keeps its exact recurrent state");
    check(std::filesystem::file_size(dir / (name + ".states")) == state_bytes,
          "retaining the consumed origin uses the existing ordinary state slots");
    ConversationDiskStore reopened;
    check(reopened.open(options, error) == ConversationDiskStatus::ok &&
              reopened.get_prefix(name, back, 32, error) == ConversationDiskStatus::ok &&
              same_checkpoint(back.stages[0].live, full_image(ids, 32).live),
          "the original continuation survives dense rotation and restart");
    std::filesystem::remove_all(dir);
}

void test_spill_replayed_prefix() {
    const auto dir = scratch_dir("replayed-prefix");
    ConversationDiskStore store;
    std::string error;
    check(store.open(options_for(dir, ConversationDiskIdentity::from_string("replayed-prefix"), 1ull << 30),
                     error) == ConversationDiskStatus::ok, "the replay-prefix store opens");
    const auto ids = chain_ids(64);
    const std::string source = "c-replayed-prefix";
    auto original = full_image(ids, 32);
    original.terminal_from = 24;
    original.checkpoints = {checkpoint_at(ids, 8, 77), checkpoint_at(ids, 23, 77),
                            checkpoint_at(ids, 30, 77)};
    ConversationDiskRecord seed;
    seed.stages.push_back(original);
    check(store.seed(source, std::move(seed), "", 0, error) == ConversationDiskStatus::ok,
          "the input and original generated tail park");
    auto replay = original;
    check(replay.kv[0].k.visit(23 * 4, 9 * 4, [](uint8_t* p, size_t n, size_t) {
              std::memset(p, 99, n); return true;
          }),
          "the replay changes the last input cell and generated K/V cells");
    replay.live.gdn.vec()[0] = 99;
    const auto expected = replay;
    uint64_t written = 0;
    check(store.spill(std::move(replay), false, error, &written) == ConversationDiskStatus::ok &&
              written == 9 && store.records() == 1,
          "an unchanged input permits replacement from its earlier restore anchor");
    ConversationDiskRecord back;
    check(store.get(source, back, error) == ConversationDiskStatus::ok && same_stage(back.stages[0], expected),
          "the rewritten reply keeps its exact input, new keys, and running state");
    auto edited = expected;
    edited.live.ids[23] += 9000;
    edited.checkpoints.erase(std::remove_if(edited.checkpoints.begin(), edited.checkpoints.end(),
        [](const auto& cp) { return cp.ids.size() > 23; }), edited.checkpoints.end());
    const auto expected_edit = edited;
    const auto edited_name = conversation_disk_name(edited.live.ids, {}, true);
    check(store.spill(std::move(edited), false, error, &written) == ConversationDiskStatus::ok &&
              written == 9 && store.records() == 2,
          "an edit before the original input boundary creates a separate prefix branch");
    check(store.get(edited_name, back, error) == ConversationDiskStatus::ok &&
              same_stage(back.stages[0], expected_edit),
          "the edited input restores its exact branch state and cells");
    check(store.get(source, back, error) == ConversationDiskStatus::ok &&
              same_stage(back.stages[0], expected),
          "the input edit cannot replace the original continuation");
    auto branch = full_image(ids, 20);
    branch.terminal_from = 18;
    branch.checkpoints.push_back(checkpoint_at(ids, 16, 77));
    check(branch.kv[0].k.visit(16 * 4, 4 * 4, [](uint8_t* p, size_t n, size_t) {
              std::memset(p, 88, n); return true;
          }),
          "the shorter replay changes cells after its retained anchor");
    const auto expected_branch = branch;
    const auto branch_name = conversation_disk_name(branch.live.ids, {}, true);
    check(store.spill(std::move(branch), false, error, &written) == ConversationDiskStatus::ok && written == 4,
          "a failed maximum-prefix proof still shares the earlier byte-identical prefix");
    const auto bytes = read_file(dir / (branch_name + ConversationDiskStore::file_suffix));
    uint64_t borrowed = 0;
    std::memcpy(&borrowed, bytes.data() + kHeaderBaseTokens, sizeof(borrowed));
    check(borrowed == 16 && store.get(branch_name, back, error) == ConversationDiskStatus::ok &&
              same_stage(back.stages[0], expected_branch),
          "the replayed branch references its original prefix instead of storing a second copy");
    check(store.get(source, back, error) == ConversationDiskStatus::ok && same_stage(back.stages[0], expected),
          "sharing a shorter replay prefix leaves the original continuation unchanged");
    std::filesystem::remove_all(dir);
}

void test_nested_branches() {
    const std::filesystem::path dir = scratch_dir("nested");
    ConversationDiskStore store;
    std::string error;
    check(store.open(options_for(dir, ConversationDiskIdentity::from_string("nested"), 1ull << 30), error) ==
              ConversationDiskStatus::ok,
          "the nested store opens");
    // A root chain at 4 tokens, then twelve branches that each borrow the whole prefix of the one
    // before: more levels than a one-level base bound allowed.
    const std::string root = ids_name(4);
    ConversationDiskRecord root_rec;
    root_rec.stages.push_back(make_delta_stage(full_image(4), 0, 4));
    check(store.seed(root, std::move(root_rec), "", 0, error) == ConversationDiskStatus::ok,
          "the root chain seeds");
    std::string previous = root;
    uint64_t tokens = 4;
    for (int level = 1; level <= 12; ++level) {
        const uint64_t upto = tokens + 3;
        ConversationDiskBase base = borrow_prefix(full_image((size_t) upto), tokens);
        base.name = previous;
        ConversationDiskRecord rec;
        rec.stages.push_back(make_delta_stage(full_image((size_t) upto), (int64_t) tokens, 4));
        const std::string name = ids_name((size_t) upto);
        check(store.seed(name, std::move(rec), base, error) == ConversationDiskStatus::ok,
              "a nested branch seeds");
        previous = name;
        tokens = upto;
    }
    check(store.records() == 13, "every nested branch is a chain");
    ConversationDiskRecord back;
    check(store.get(previous, back, error) == ConversationDiskStatus::ok &&
              same_record(back, record_of((size_t) tokens)),
          "the deepest branch merges to its full capture");
    ConversationDiskStore reopened;
    check(reopened.open(options_for(dir, ConversationDiskIdentity::from_string("nested"), 1ull << 30), error) ==
              ConversationDiskStatus::ok,
          "the nested store reopens");
    check(reopened.get(previous, back, error) == ConversationDiskStatus::ok &&
              same_record(back, record_of((size_t) tokens)),
          "the deepest branch survives a restart");
    // A branch of the deepest chain at a point the root already covers collapses to the root, so
    // the chain it named is no longer needed.
    ConversationDiskBase collapsed = borrow_prefix(full_image(7), 4);
    collapsed.name = previous;
    ConversationDiskRecord rec;
    rec.stages.push_back(make_delta_stage(full_image(7), 4, 4));
    const std::string short_branch = ids_name(7, "-short");
    check(store.seed(short_branch, std::move(rec), collapsed, error) == ConversationDiskStatus::ok,
          "a branch below the deepest chain's own base seeds");
    check(store.remove(previous, error) == ConversationDiskStatus::ok, "the intermediate chain removes");
    check(store.get(short_branch, back, error) == ConversationDiskStatus::ok &&
              same_record(back, record_of(7)),
          "the collapsed branch reads back without the chain it named");
    std::filesystem::remove_all(dir);
}

void test_dependency_lifecycle() {
    const std::filesystem::path dir = scratch_dir("depend");
    ConversationDiskStore store;
    std::string error;
    check(store.open(options_for(dir, ConversationDiskIdentity::from_string("depend"), 1ull << 30), error) ==
              ConversationDiskStatus::ok,
          "the dependency store opens");
    const std::string source = ids_name(24);
    ConversationDiskRecord seed;
    seed.stages.push_back(make_delta_stage(full_image(12), 0, 4));
    check(store.seed(source, std::move(seed), "", 0, error) == ConversationDiskStatus::ok,
          "the source chain seeds");
    ConversationDiskRecord d12;
    d12.stages.push_back(make_delta_stage(full_image(24), 12, 4));
    check(store.append(source, std::move(d12), error) == ConversationDiskStatus::ok,
          "the source chain appends to 24");
    ConversationDiskBase base = borrow_prefix(full_image(20), 16);
    base.name = source;
    ConversationDiskRecord branch_rec;
    branch_rec.stages.push_back(make_delta_stage(full_image(20), 16, 4));
    const std::string branch = ids_name(20, "-branch");
    check(store.seed(branch, std::move(branch_rec), base, error) == ConversationDiskStatus::ok,
          "the branch seeds on the source");

    // A rewind below the borrowed prefix would strand the branch: it is refused, and the source
    // keeps its tail byte for byte.
    const std::vector<uint8_t> before = read_file(dir / (source + ".conversation"));
    uint64_t cover = 0;
    check(store.rewind_chain(source, 12, cover, error) == ConversationDiskStatus::invalid,
          "a rewind below a borrowed prefix is refused");
    check(read_file(dir / (source + ".conversation")) == before && store.cover(source, cover) &&
              cover == 24,
          "the refused rewind leaves the source intact");
    check(store.rewind_chain(source, 24, cover, error) == ConversationDiskStatus::ok && cover == 24,
          "a rewind at the borrowed prefix is a no-op");

    // Removing the source removes the branch with it: neither can load alone.
    check(store.remove(source, error) == ConversationDiskStatus::ok, "the source removes");
    check(!store.has(source) && !store.has(branch), "the branch goes with the chain it borrows from");
    check(!std::filesystem::exists(dir / (branch + ".conversation")), "the branch file is gone");

    // Eviction: the chains that borrow go first, so a referenced chain keeps its dependents.
    const std::filesystem::path tight_dir = scratch_dir("depend-tight");
    ConversationDiskStore tight;
    check(tight.open(options_for(tight_dir, ConversationDiskIdentity::from_string("tight"), 1ull << 30),
                      error) == ConversationDiskStatus::ok,
          "the tight store opens");
    const std::string tight_source = ids_name(12);
    ConversationDiskRecord tight_seed;
    tight_seed.stages.push_back(make_delta_stage(full_image(12), 0, 4));
    check(tight.seed(tight_source, std::move(tight_seed), "", 0, error) == ConversationDiskStatus::ok,
          "the tight source seeds");
    ConversationDiskBase tight_base = borrow_prefix(full_image(20), 12);
    tight_base.name = tight_source;
    ConversationDiskRecord tight_branch;
    tight_branch.stages.push_back(make_delta_stage(full_image(20), 12, 4));
    check(tight.seed(ids_name(20, "-tight"), std::move(tight_branch), tight_base, error) ==
              ConversationDiskStatus::ok,
          "the tight branch seeds");
    // A budget that holds both chains exactly: the forcing seed must evict the chain that borrows
    // (the only unreferenced one) and leave the referenced source in place.
    const uint64_t source_bytes = stored_bytes(tight_dir, tight_source);
    const uint64_t branch_bytes = stored_bytes(tight_dir, ids_name(20, "-tight"));
    ConversationDiskStore snug;
    check(snug.open(options_for(tight_dir, ConversationDiskIdentity::from_string("tight"),
                               source_bytes + branch_bytes + 1),
                    error) == ConversationDiskStatus::ok,
          "the snug store reopens");
    check(snug.records() == 2, "the snug store re-indexed both chains");
    ConversationDiskRecord forcing;
    forcing.stages.push_back(make_delta_stage(full_image(4, 90), 0, 4));
    const std::string forcing_name = ids_name(4, "-forcing");
    check(snug.seed(forcing_name, std::move(forcing), "", 0, error) == ConversationDiskStatus::ok,
          "the forcing seed parks within the budget");
    check(snug.has(forcing_name) && snug.has(tight_source),
          "the referenced chain survives the eviction");
    check(!snug.has(ids_name(20, "-tight")), "the unreferenced branch is evicted first");
    std::filesystem::remove_all(dir);
    std::filesystem::remove_all(tight_dir);
}

void test_bounded_state_storage() {
    const std::filesystem::path dir = scratch_dir("bounded");
    ConversationDiskOptions options = options_for(dir, ConversationDiskIdentity::from_string("bounded"), 1ull << 30);
    options.max_checkpoint_states = 2;
    ConversationDiskStore store;
    std::string error;
    check(store.open(options, error) == ConversationDiskStatus::ok, "the bounded store opens");
    const std::string name = ids_name(32);
    ConversationDiskRecord seed;
    seed.stages.push_back(make_delta_stage(full_image(16), 0, 4));
    for (size_t upto = 4; upto <= 16; upto += 4)
        seed.stages[0].checkpoints.push_back(checkpoint_at(upto, (uint8_t) (100 + upto)));
    check(store.seed(name, std::move(seed), "", 0, error) == ConversationDiskStatus::ok, "the bounded chain seeds");
    const std::filesystem::path chain_file = dir / (name + ".conversation");
    const std::filesystem::path state_file = dir / (name + ".states");
    check(std::filesystem::exists(state_file), "the running states live in the chain's sidecar");
    const uint64_t chain_bytes = std::filesystem::file_size(chain_file);
    const uint64_t state_bytes = std::filesystem::file_size(state_file);
    // Four more panels, each with its own checkpoint: the K/V delta grows, the state store does not.
    for (uint64_t base = 16; base < 32; base += 4) {
        ConversationDiskRecord panel;
        panel.stages.push_back(make_delta_stage(full_image((size_t) base + 4), (int64_t) base, 4));
        panel.stages[0].checkpoints.push_back(checkpoint_at((size_t) base + 2, (uint8_t) (60 + base)));
        check(store.append(name, std::move(panel), error) == ConversationDiskStatus::ok,
              "a bounded chain appends a panel");
    }
    check(std::filesystem::file_size(state_file) == state_bytes,
          "the state store keeps its slot count: it never grows with the number of anchors");
    check(std::filesystem::file_size(chain_file) > chain_bytes, "the chain grew by its K/V deltas");
    ConversationDiskStore reopened;
    check(reopened.open(options, error) == ConversationDiskStatus::ok, "the bounded store reopens");
    ConversationDiskRecord back;
    check(reopened.get(name, back, error) == ConversationDiskStatus::ok && same_record(back, record_of(32)),
          "the bounded chain reads back whole");
    check(!back.stages[0].checkpoints.empty(), "the sidecar restores the chain's anchors");
    check(reopened.get_prefix(name, back, 20, error) == ConversationDiskStatus::ok &&
              back.stages[0].live.ids.size() == 20,
          "a bounded prefix read trims to the cut");
    std::filesystem::remove_all(dir);
}

void test_branch_corruption() {
    const std::filesystem::path dir = scratch_dir("branch-corrupt");
    ConversationDiskStore store;
    std::string error;
    check(store.open(options_for(dir, ConversationDiskIdentity::from_string("corrupt"), 1ull << 30), error) ==
              ConversationDiskStatus::ok,
          "the corruption store opens");
    const std::string source = ids_name(16);
    ConversationDiskRecord seed;
    seed.stages.push_back(make_delta_stage(full_image(16), 0, 4));
    check(store.seed(source, std::move(seed), "", 0, error) == ConversationDiskStatus::ok,
          "the corruption source seeds");
    ConversationDiskBase base = borrow_prefix(full_image(24), 8);
    base.name = source;
    ConversationDiskRecord branch_rec;
    branch_rec.stages.push_back(make_delta_stage(full_image(24), 8, 4));
    const std::string branch = ids_name(24, "-corrupt");
    check(store.seed(branch, std::move(branch_rec), base, error) == ConversationDiskStatus::ok,
          "the corruption branch seeds");
    const std::filesystem::path branch_file = dir / (branch + ".conversation");
    const std::filesystem::path source_file = dir / (source + ".conversation");
    const std::vector<uint8_t> branch_bytes = read_file(branch_file);
    const std::vector<uint8_t> source_bytes = read_file(source_file);
    const uint64_t corruptions = store.corruptions();

    // A borrowed length beyond the base's coverage, and a prefix fingerprint that does not match
    // the base, both make the branch unloadable: open() drops it and keeps the source.
    ConversationDiskStore beyond;
    patch_u64(branch_file, kHeaderBaseTokens, 4096);
    check(beyond.open(options_for(dir, ConversationDiskIdentity::from_string("corrupt"), 1ull << 30), error) ==
              ConversationDiskStatus::ok,
          "the store opens after a bad borrowed length");
    check(!beyond.has(branch) && !std::filesystem::exists(branch_file),
          "a branch whose base is too short is dropped");
    check(beyond.has(source) && read_file(source_file) == source_bytes,
          "the source survives the dropped branch");
    check(beyond.corruptions() > corruptions, "the drop counts as a corruption");

    std::FILE* restore = std::fopen(branch_file.string().c_str(), "wb");
    check(restore != nullptr, "the branch file is rewritten");
    check(std::fwrite(branch_bytes.data(), 1, branch_bytes.size(), restore) == branch_bytes.size(),
          "the branch bytes are restored");
    std::fclose(restore);
    ConversationDiskStore fingerprint;
    patch_u64(branch_file, kHeaderBasePrefixChecksum, 0x5eed);
    check(fingerprint.open(options_for(dir, ConversationDiskIdentity::from_string("corrupt"), 1ull << 30),
                           error) == ConversationDiskStatus::ok,
          "the store opens after a bad fingerprint");
    check(!fingerprint.has(branch) && fingerprint.has(source),
          "a branch whose prefix fingerprint does not match the base is dropped");

    restore = std::fopen(branch_file.string().c_str(), "wb");
    check(restore != nullptr, "the branch file is rewritten again");
    check(std::fwrite(branch_bytes.data(), 1, branch_bytes.size(), restore) == branch_bytes.size(),
          "the branch bytes are restored again");
    std::fclose(restore);
    ConversationDiskStore missing;
    std::FILE* rename_file = std::fopen(branch_file.string().c_str(), "r+b");
    check(rename_file != nullptr, "the branch header opens");
    const char gone[40] = "c0-nosuch";
    check(std::fseek(rename_file, static_cast<long>(kHeaderBaseName), SEEK_SET) == 0 &&
              std::fwrite(gone, 1, sizeof(gone), rename_file) == sizeof(gone),
          "the branch names a missing base");
    std::fclose(rename_file);
    check(missing.open(options_for(dir, ConversationDiskIdentity::from_string("corrupt"), 1ull << 30), error) ==
              ConversationDiskStatus::ok,
          "the store opens after a missing base");
    check(!missing.has(branch) && missing.has(source), "a branch with a missing base is dropped");
    std::filesystem::remove_all(dir);
}
} // namespace

int main() {
    test_l2_first_spill();
    test_basic_chain();
    test_root_base();
    test_torn_tail();
    test_rewind();
    test_eviction_order();
    test_reopen_and_purge();
    test_two_stages();
    test_delta_checkpoint_below_base();
    test_branch_prefix_reuse();
    test_get_prefix_extent();
    test_legacy_inline_chain();
    test_sidecar_retention();
    test_lost_sidecar_is_not_offered();
    test_sidecar_write_failure();
    test_shared_prefix_lookup();
    test_branch_collapse_extent();
    test_incremental_spill();
    test_terminal_rewrite_retention();
    test_consumed_origin_retention();
    test_spill_replayed_prefix();
    test_nested_branches();
    test_dependency_lifecycle();
    test_branch_corruption();
    test_bounded_state_storage();
    std::fprintf(stderr, "conversation_disk_test: %d checks passed\n", checks);
    return 0;
}
