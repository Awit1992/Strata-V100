// include/strata/core/conversation_disk.hpp - the L3 disk tier for parked conversations.
//
// The RAM cache (conversation_cache.hpp) holds whole conversations in host memory. It forgets
// them when the process exits. This store keeps the same images in files. A restarted server
// can then resume a conversation without a full prompt read.
//
// The store is CPU only. It makes no CUDA call. The caller captures and restores the images
// with conversation_snapshot.hpp.
//
// Each conversation is an append-only CHAIN: one file, a fixed master header followed by one
// PANEL per park.  A park writes only the delta since the chain's last panel - the ids/imgs it
// gained, new checkpoints, and per-layer K/V tails - so a growing conversation never rewrites
// its history.  A self-contained chain covers tokens [0 .. N).  A chain may instead BORROW a
// prefix from another chain:
//   * base kind 1 - the named 'p' chain (a shared system prompt) lends [0 .. base_tokens);
//   * base kind 2 - the named 'c' chain lends its first base_tokens tokens (it may hold more), so a
//     short branch stores only its divergent suffix and the source keeps its own tail.
// A base chain may itself have a base (a branch of a branch).  The graph is acyclic and its depth
// is bounded by max_base_depth; the store resolves it at open() and drops a chain whose base is
// gone, too short, or built on a different stage layout.
// Reading merges the base and the panels back into full stage images, then hands them to the
// caller exactly like a one-record store.  get_prefix() does the same for the first `upto` tokens
// only, so a short branch never reads a long chain's whole K/V.
//
// A torn append (crash mid-panel) is recovered on the next open: the panel checksum fails and
// the file is truncated to the last complete panel, so the conversation survives to its previous
// park - strictly better than a full-snapshot store, where a crash mid-write loses the record.
//
// Ownership and lifetime:
//  * The store owns its index and its files.
//  * seed() and append() consume the record. The store writes it and the caller must not reuse it.
//  * get() fills a record the caller owns. The record owns its buffers.
//  * A ConversationDiskMatch holds a record name. That name stays valid until the next mutation
//    or remove(). The caller must call get() before then.
//  * One thread uses one store. The store is synchronous and makes no lock.
#pragma once

#include "strata/core/conversation_cache.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace strata::core {

// The FNV-1a 64 checksum seed and mix. The store uses one checksum for the metadata block and
// one for the payload block.
inline constexpr uint64_t conversation_disk_checksum_seed = 14695981039346656037ull;

inline uint64_t conversation_disk_checksum(const uint8_t* data, size_t size,
                                           uint64_t seed = conversation_disk_checksum_seed) {
    uint64_t hash = seed;
    for (size_t i = 0; i < size; ++i) {
        hash ^= data[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

// A caller-supplied compatibility key. Two runs share a store only if the keys match.
// The key must cover the model weights and every runtime flag that changes the payload layout
// (quantization, rope scaling, layer split, cvec mode). The store compares the 32 bytes exactly.
struct ConversationDiskIdentity {
    static constexpr size_t size = 32;
    std::array<uint8_t, size> bytes{};
    bool operator==(const ConversationDiskIdentity&) const = default;
    // A stable label from a text key. This is a compatibility label, not a secure hash.
    static ConversationDiskIdentity from_string(std::string_view text);
};

// One conversation as a set of layer-split stage images.
struct ConversationDiskRecord {
    // One entry per stage, ordered by layer_lo. A single-GPU run uses one entry.
    std::vector<SavedConversation> stages;
    // The RAM size of the images. The caller uses it for admission.
    size_t bytes() const {
        size_t n = stages.capacity() * sizeof(SavedConversation);
        for (const auto& stage : stages) n += stage.bytes();
        return n;
    }
};

// A lookup result. name is the record to load. `tokens` is the matched prefix length whose running
// state a restore can resume from; `prefix_tokens` is the K/V extent the chain shares with the
// prompt (the common prefix of the chain's stored content), which can be LONGER than `tokens` when
// the prompt diverges inside the chain.  Both are byte-verified by the caller before reuse.
struct ConversationDiskMatch {
    std::string name;
    int64_t tokens = 0;
    int64_t prefix_tokens = 0;
};

enum class ConversationDiskStatus {
    ok,        // the call succeeded
    miss,      // no record has that name or prefix
    corrupt,   // the file failed validation and the store removed it
    invalid,   // the caller's record, name, or options are not usable
    disabled,  // the store has no byte budget
    failed,    // an I/O or allocation error; the caller may retry
};

struct ConversationDiskOptions {
    std::filesystem::path directory;   // the store root; open() creates it
    ConversationDiskIdentity identity; // the current run's compatibility key
    uint64_t budget_bytes = 0;         // the file budget; 0 disables the store
    size_t max_records = 0;            // the record cap; 0 means no cap
    uint64_t min_free_bytes = 0;       // the free-space floor the store keeps
    // The checkpoint running states a chain keeps per stage.  The store always keeps every POSITION
    // it indexed (ids/imgs); only the state payloads outside the bound go.
    //   * a chain with a sidecar: 0 asks for the MINIMUM - the earliest anchor and the newest - so
    //     the slots stay bounded (it never keeps "every state");
    //   * a chain with no usable sidecar (an older record, or one whose sidecar cannot be created):
    //     the states its panels carry inline are kept, and this bound drops the ones outside it at
    //     load, so 0 keeps every state such a panel carries.
    // The default is conservative; the caller sets this from --prompt-cache.
    size_t max_checkpoint_states = 6;
};

// A base reference for seed(): the named chain contributes its first `tokens` tokens and their
// images to the new chain, so a branch stores only its divergent suffix. `ids`/`imgs` are the
// caller's own copy of that prefix. seed() validates them byte-for-byte against the base chain's
// stored content, and its cvec flag and stage layout, and fails before it writes anything.
struct ConversationDiskBase {
    std::string name;                       // the base chain: a 'p' root or a 'c' conversation
    uint64_t tokens = 0;                    // how many of its tokens the new chain borrows
    std::vector<int32_t> ids;               // the caller's expected prefix (validated, not stored)
    std::vector<ConversationImageKey> imgs; // the caller's expected prefix images (same)
};

// A safe record name from a token prefix. The store uses the same name in put().
std::string conversation_disk_name(const std::vector<int32_t>& ids,
                                   const std::vector<ConversationImageKey>& images, bool cvec);
// The name of a shared system-prompt prefix record: the same ids for every chat of a client, and the
// record class ('p') the LRU evicts only after every conversation record ('c').
std::string conversation_disk_prefix_name(const std::vector<int32_t>& ids,
                                          const std::vector<ConversationImageKey>& images, bool cvec);

class ConversationDiskStore {
public:
    static constexpr uint32_t format_version = 2;
    static constexpr uint64_t max_metadata_bytes = 64ull << 20;
    // Allocation-free sizes in the disk format. Archive growth uses the same
    // limit as the writer, before it moves or discards any checkpoint state.
    static size_t metadata_bytes(const SavedConversation& image);
    static size_t metadata_bytes(const ConversationCheckpoint& checkpoint);
    // The metadata bytes of a whole record: its stage count and every stage's shape.
    static size_t metadata_bytes(const ConversationDiskRecord& record);
    static constexpr const char* file_suffix = ".conversation";

    // The fixed layout of one stage: the layer carve, the runtime geometry, and the K/V geometry
    // that every panel of a chain shares. The index holds one per stage, so a branch seed is
    // validated against its base without reading the base's payload.
    struct StageShape {
        int64_t layer_lo = 0, layer_hi = 0;
        std::array<int64_t, 18> geometry{};
        bool cvec = true;
        int64_t format = 0, heads = 0, head_dim = 0, page_size = 0, idx_dim = 0;
        bool operator==(const StageShape&) const = default;
    };
    // The deepest chain of bases the store resolves. A branch chain may use another branch chain
    // as its base, so a run of branches on one conversation nests. The graph is acyclic (a seed
    // refuses a base that reaches the chain's own name) and this is its bound.
    static constexpr size_t max_base_depth = 1024;

    ConversationDiskStore() = default;
    ConversationDiskStore(const ConversationDiskStore&) = delete;
    ConversationDiskStore& operator=(const ConversationDiskStore&) = delete;
    ConversationDiskStore(ConversationDiskStore&&) = default;
    ConversationDiskStore& operator=(ConversationDiskStore&&) = default;

    // Opens the directory and rebuilds the index. Removes every file that fails validation,
    // including files from an older format and chains whose base record is gone, too short, or
    // built on a different stage layout.  A chain whose last panel is torn is truncated to its
    // last complete panel instead (the crash recovery).
    ConversationDiskStatus open(const ConversationDiskOptions& options, std::string& error);

    bool enabled() const { return options_.budget_bytes != 0; }

    // Finds the longest stored token prefix whose running state a restore can resume from.  It
    // considers only chains with the same cvec flag.  A tie prefers the most recently used chain.
    // This call does no I/O and no allocation.
    bool best(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
              bool cvec, ConversationDiskMatch& match) const;

    // An index-only lookup for a K/V prefix BORROW (a park that wants a branch base rather than a
    // restore point): the chain whose stored CONTENT shares the longest prefix with `ids`/`images`,
    // whether or not it holds a restorable running state at that position.  `best()` cannot serve
    // this: it selects a restore point, so a chain whose only stored point is longer than the prompt
    // is invisible to it, and it prefers the longest point rather than the longest common K/V
    // extent.  `match.name` and `match.prefix_tokens` (the common extent) are set; `match.tokens` is
    // 0 because no restore point is selected.  The caller MUST still byte-verify the prefix before it
    // borrows it.  This call does no I/O and no allocation.
    bool shared_prefix(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                       bool cvec, ConversationDiskMatch& match) const;

    // Publishes a NEW chain: one self-contained panel covering [0 .. end) OR, when `base_name`
    // is non-empty, a panel covering [base_tokens .. end) whose earlier tokens the named chain
    // contributes (it is pinned while this chain lives).  `name` must be the caller's stable chain
    // key (the record name of the seed content).  Replaces a chain with the same name.  Consumes
    // the record in all cases.
    ConversationDiskStatus seed(const std::string& name, ConversationDiskRecord&& record,
                                std::string base_name, uint64_t base_tokens, std::string& error);

    // The same, with the borrowed prefix validated against the caller's own copy of it.  This is
    // the branch form: `base.name` may cover MORE than `base.tokens` (base kind 2), so a short
    // branch of a long conversation stores only its divergent suffix and never copies the prefix.
    // The base chain must be indexed, must cover `base.tokens`, and its content, images, cvec flag
    // and stage layout must equal the caller's.  An empty `base.name` is the self-contained form.
    ConversationDiskStatus seed(const std::string& name, ConversationDiskRecord&& record,
                                const ConversationDiskBase& base, std::string& error);

    // Moves a complete L2 victim to one L3 record.  When the store already holds a chain whose
    // stored content is a prefix of this image, the call APPENDS only the new tokens to it instead
    // of writing a second copy of the prefix; a chain that already holds the whole image writes
    // nothing.  Otherwise the image is published self-contained.  The stored prefix BYTES are
    // compared before an append: the same ids from a different arithmetic path must not mix into
    // the stored chain's K/V.
    // `written_tokens` (optional) receives the tokens THIS call wrote (the delta on an append, the
    // whole image on a seed, 0 when the chain already held them).  `written_bytes` (optional)
    // receives the bytes it serialized: the chain header, the panel header, the metadata, and the
    // payload - the honest L3 write size, never the logical size of the shared image.
    ConversationDiskStatus spill(SavedConversation&& image, bool root, std::string& error,
                                 uint64_t* written_tokens = nullptr, size_t* written_bytes = nullptr);

    // Appends one panel to an existing chain: the delta record covers [the chain's current
    // coverage .. its ids length later).  Returns miss when the chain is gone (the caller then
    // seeds it again).  Consumes the record in all cases except `invalid`.
    ConversationDiskStatus append(const std::string& name, ConversationDiskRecord&& record,
                                  std::string& error);

    // True when `name` is an indexed chain whose coverage is exactly `tokens`, whose stored content
    // equals the first `tokens` tokens of `ids`/`images` (cvec must match), and whose tail is not
    // already scheduled for replacement.  append() to that chain then adds exactly the new tokens.
    // This call does no I/O and no allocation.
    bool can_append(const std::string& name, const std::vector<int32_t>& ids,
                    const std::vector<ConversationImageKey>& images, bool cvec, uint64_t tokens) const;

    // The token coverage of a chain (its latest panel's end), or false when it is not indexed.
    bool cover(const std::string& name, uint64_t& tokens) const;

    // Reuses a saved checkpoint at `at` as the next append's base without rewriting
    // the existing file. Other positions retain the deepest complete panel.
    // The pending replacement becomes persistent only after a successful append.
    // A chain that another chain borrows a prefix from is never cut below that prefix:
    // the call returns invalid and writes nothing.
    ConversationDiskStatus rewind_chain(const std::string& name, uint64_t at, uint64_t& new_cover,
                                        std::string& error);

    // Replaces a chain's TERMINAL tail with `delta`, which covers [from, from + its slice), without
    // writing a second chain: the file keeps its earlier panels and the next panel starts at `from`,
    // so every later panel is superseded.  The caller has already proven that the chain's stored
    // content [0, from) equals the incoming image's - token ids, images and K/V bytes - and that the
    // divergence is the client's rewritten terminal reply, not a genuine branch.
    // The store refuses (invalid, and the delta is left untouched so the caller may seed a genuine
    // branch from it) unless every safety condition holds:
    //   * the chain carries a known input-prompt boundary. If replay changes cells before that
    //     boundary, the delta must contain the same remaining input tokens and image keys.
    //     An input edit or unknown provenance (0, a legacy record) always branches instead
    //     of losing the original tail;
    //   * no PINNED anchor lies above `from`, including a pinned live tail: a pin is the caller's
    //     explicit retention contract, so a replacement never drops one;
    //   * no branch chain borrows a prefix past `from`;
    //   * the chain keeps at least one restore anchor with a running state at or below `from`.
    // Ordinary generated anchors above the terminal boundary are ignored: they describe the reply
    // being replaced.  `delta`'s own terminal boundary is persisted as the chain's new one (0 keeps
    // it unknown, which only makes a LATER replacement branch), and it may lie past `from`: a
    // rewritten reply usually grows its input.  On failure the file stays at its previous coverage
    // and no rewrite stays scheduled; the chain is never left partially replaced.
    ConversationDiskStatus replace_terminal_tail(const std::string& name, uint64_t from,
                                                 ConversationDiskRecord&& delta, std::string& error);

    // Loads a chain into RAM: resolves the base, merges every panel, validates each panel's
    // checksums while it streams, and truncates a torn tail.  Removes the file on corruption.
    ConversationDiskStatus get(const std::string& name, ConversationDiskRecord& record, std::string& error);

    // Loads the first `upto` tokens of a chain into RAM.  `upto` must be non-zero.  The returned
    // image covers [0 .. min(upto, the chain's coverage)): the ids and images are trimmed, every
    // K/V buffer is sized to the page-aligned cut, and a checkpoint above the cut is not returned.
    // The live running state is returned when the chain holds a state at the cut - a stored panel
    // ends exactly there, or a retained anchor does (its own sidecar slot, or a panel's inline
    // state), which is promoted into the live state.  Otherwise the live state is empty and the
    // caller MUST restore a checkpoint at or below the cut and replay forward.  The panels past the
    // cut are not read at all; the panel that holds the cut is read in full for its checksum.
    ConversationDiskStatus get_prefix(const std::string& name, ConversationDiskRecord& record,
                                      uint64_t upto, std::string& error);
    bool has(const std::string& name) const { return find(name) != nullptr; }

    // Removes a chain and every chain that borrows a prefix from it.
    ConversationDiskStatus remove(const std::string& name, std::string& error);

    size_t records() const { return entries_.size(); }
    uint64_t bytes() const { return bytes_; }
    uint64_t evictions() const { return evictions_; }
    uint64_t corruptions() const { return corruptions_; }
    uint64_t recoveries() const { return recoveries_; }
    const std::filesystem::path& directory() const { return options_.directory; }

private:
    // The in-RAM index of one chain. The points hold ids and imgs only, never payload bytes.
    struct Entry {
        std::string name;
        uint64_t bytes = 0;
        uint64_t stamp = 0;
        uint64_t cover = 0;            ///< tokens the chain covers at its last write
        uint64_t rewrite_from = UINT64_MAX; ///< pending tail replacement; file remains intact until append
        bool cvec = true;
        bool root = false;             ///< a system-prompt ('p') chain: evicted after conversations
        size_t refs = 0;               ///< chains whose base this chain is
        std::string base;              ///< the referenced chain's name, or empty
        uint64_t base_tokens = 0;      ///< how many of the base's tokens this chain borrows
        size_t depth = 0;              ///< the base chain depth (0 = self-contained)
        std::vector<StageShape> shape; ///< one per stage, the layout every panel shares
        std::vector<ConversationCheckpoint> points;
        std::vector<bool> point_states; ///< per point: a running state is reachable there (see best())
        std::vector<bool> point_inline_states; ///< per point: a panel holds the state inline (not the sidecar)
        uint64_t terminal_from = 0;    ///< the input-prompt boundary of the chain's last write (0 unknown)
    };
    // Promotes the checkpoint at exactly `upto` into the live state when no panel ends there, so
    // a caller that asked for a stored restore point always receives a live state at it.
    static void promote_anchor_state(ConversationDiskRecord& record, uint64_t upto);
    // ---- The bounded anchor-state sidecar (<name>.states) ----
    std::filesystem::path state_path_for(const std::string& name) const;
    uint64_t state_file_bytes(const std::string& name) const;
    // The ordinary anchors the retention policy keeps (the caller's bound, or the minimum).
    size_t ordinary_slots() const;
    // The ordinary anchor slots: the caller's state bound plus the protected earliest anchor.
    size_t anchor_slots() const;
    // Publishes a record's running states into the chain's sidecar and strips them from the record,
    // so the panel stores neither a state nor a repeated checkpoint position, and persists the
    // record's terminal boundary in the sidecar header.  False: the sidecar cannot be opened or
    // written, so the caller fails the whole write instead of publishing a panel whose states the
    // sidecar could not serve.
    bool state_store(const std::string& name, ConversationDiskRecord& record, uint64_t end,
                     const std::vector<int32_t>& content_ids,
                     const std::vector<ConversationImageKey>& content_imgs, size_t* written_bytes,
                     std::string& error) const;
    // Blanks every anchor above `tokens` (a rewind rewrote the tail they describe) and, with
    // `remove_file`, unlinks the sidecar.
    void state_discard(const std::string& name, uint64_t tokens, bool remove_file) const;
    // Installs an entry's exact restore anchors: every point carries a running state when a panel
    // holds it inline or the sidecar currently retains a slot at its position, and a retained slot
    // that is not a point yet becomes one.  The sidecar's slots are the truth - the retention
    // rotates them - so the in-RAM index and a reopen offer exactly the same restore points.
    void settle_entry_states(Entry& entry, std::vector<bool> inline_states) const;

    std::filesystem::path path_for(const std::string& name) const;
    static bool valid_name(const std::string& name);
    const Entry* find(const std::string& name) const;
    Entry* find(const std::string& name);
    size_t lru_victim(const std::string& skip, const std::string& keep) const;
    void evict(size_t index);
    ConversationDiskStatus make_room(uint64_t incoming, const std::string& skip,
                                     const std::string& keep, bool adds_record, std::string& error);
    ConversationDiskStatus index_file(const std::filesystem::path& path, const std::string& name,
                                      Entry& entry, std::string& error) const;
    // The longest prefix any branch chain borrows from `name` (0 when none does).
    uint64_t borrowed_prefix(const std::string& name) const;
    // Truncates a torn last panel (a crash mid-append) back to the chain's previous coverage and
    // re-indexes the entry.  Returns false when the tail is intact or the file does not validate.
    bool truncate_torn_tail(const std::string& name);
    // Loads one chain file into `out`, merging at most the first `upto` tokens.  A referenced base
    // is resolved through the same path (depth is limited to max_base_depth).  Panels are merged in
    // order; a torn tail is detected by its checksum and dropped.
    ConversationDiskStatus read_chain(const std::filesystem::path& path, int depth,
                                      ConversationDiskRecord& out, std::string& error,
                                      uint64_t upto) const;
    static ConversationDiskStatus read_chain_impl(const std::filesystem::path& path,
                                                  const ConversationDiskStore& self, int depth,
                                                  ConversationDiskRecord& out, std::string& error,
                                                  uint64_t upto);
    // Bounds the checkpoint states of a freshly merged record to options_.max_checkpoint_states:
    // the positions stay, the states outside the bound go (SavedConversation::drop_checkpoint_states).
    void bound_checkpoint_states(ConversationDiskRecord& record) const;

    ConversationDiskOptions options_{};
    std::vector<Entry> entries_;
    uint64_t bytes_ = 0;
    uint64_t stamp_ = 0;
    uint64_t evictions_ = 0;
    uint64_t corruptions_ = 0;
    uint64_t recoveries_ = 0;
};

} // namespace strata::core
