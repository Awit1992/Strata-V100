#pragma once

#include "strata/core/progress.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace strata::core {

// Snapshot-owned host storage.  The payload is a list of pieces, each a shared
// 16 MiB allocation plus the number of bytes of it this buffer uses.  A copy, a
// fork, or a prefix trim adopts the pieces without copying a byte; a write copies
// only the piece bytes it actually touches (a shared piece is split first, so a
// dirty tail copies that tail alone), and appending to a shared piece allocates a
// new one instead of copying the prefix.  The piece directory stays private.
class ConversationBuffer {
public:
    static constexpr size_t segment_bytes = 16 * 1024 * 1024;

    ConversationBuffer() = default;
    ConversationBuffer(std::initializer_list<uint8_t> values) {
        resize(values.size());
        visit(0, size(), [&](uint8_t* p, size_t n, size_t at) {
            std::memcpy(p, values.begin() + at, n);
            return true;
        });
    }
    // A copy adopts the source's pieces: neither side copies a byte until it writes.
    ConversationBuffer(const ConversationBuffer&) = default;
    ConversationBuffer& operator=(const ConversationBuffer&) = default;
    ConversationBuffer(ConversationBuffer&& other) noexcept
        : pieces_(std::move(other.pieces_)), size_(std::exchange(other.size_, 0)) {
        other.pieces_.clear();
    }
    ConversationBuffer& operator=(ConversationBuffer&& other) noexcept {
        if (this == &other) return *this;
        pieces_ = std::move(other.pieces_);
        size_ = std::exchange(other.size_, 0);
        other.pieces_.clear();
        return *this;
    }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    // Every allocation this buffer reaches, counted once here however many buffers
    // share it; the cache's ledger deduplicates across its entries.
    size_t bytes() const {
        return pieces_.capacity() * sizeof(Piece) + allocation_bytes();
    }
    // The physical allocations alone, without this buffer's directory: the bytes
    // the ledger charges and releases.
    size_t allocation_bytes() const {
        size_t n = 0;
        visit_allocations([&](const void*, size_t bytes) { n += bytes; });
        return n;
    }
    // fn(const void* allocation, size_t bytes): every physical allocation, once.
    template<class Fn> void visit_allocations(Fn&& fn) const {
        for (size_t i = 0; i < pieces_.size(); ++i) {
            const auto allocation = pieces_[i].seg.get();
            bool first = true;
            for (size_t j = 0; j < i; ++j)
                if (pieces_[j].seg.get() == allocation) { first = false; break; }
            if (first) fn((const void*) allocation, allocation->capacity());
        }
    }
    // Bytes a writable visit of [offset, offset+count) must copy because they are
    // still shared with another owner.  Zero for a private range: a shared piece is
    // split, so only these bytes are copied, never the prefix or the spare tail.
    size_t detach_peak(size_t offset, size_t count) const {
        if (!count || offset > size_ || count > size_ - offset) return 0;
        const size_t stop = offset + count;
        size_t begin = 0, total = 0;
        for (const auto& piece : pieces_) {
            const size_t end = begin + piece.len;
            if (begin >= stop) break;
            if (end > offset && (piece.seg.use_count() != 1)) total += std::min(end, stop) - std::max(begin, offset);
            begin = end;
        }
        return total;
    }
    // Peak storage for resize to `n`, optionally followed by a suffix write.
    // Charge shared bytes rewritten by that write and both directories while
    // reserve replaces one. Pure prefix trims do not allocate.
    size_t allocation_peak(size_t n, size_t rewrite_from = SIZE_MAX) const {
        size_t total = bytes(), count = pieces_.size();
        if (n > size_) {
            size_t spare = 0;
            if (!pieces_.empty() && (pieces_.back().seg.use_count() == 1))
                spare = pieces_.back().seg->capacity() - pieces_.back().start - pieces_.back().len;
            const size_t extra = n - size_ - std::min(n - size_, spare);
            const size_t full = extra / segment_bytes, tail = extra % segment_bytes;
            if (!add(total, full * segment_bytes)) return SIZE_MAX;
            if (tail && !add(total, segment_capacity(tail, full ? segment_bytes :
                    pieces_.empty() || (pieces_.back().seg.use_count() != 1) ? 0 : pieces_.back().seg->capacity(),
                    size_ != 0, size_))) return SIZE_MAX;
            if (!add(count, full + (tail != 0))) return SIZE_MAX;
            if (count > pieces_.capacity() &&
                (count > SIZE_MAX / sizeof(Piece) || !add(total, count * sizeof(Piece)))) return SIZE_MAX;
        } else if (n < size_) {
            count = 0;
            size_t begin = 0;
            for (const auto& piece : pieces_) {
                if (begin >= n) break;
                ++count;
                begin += piece.len;
            }
        }
        if (rewrite_from >= n) return total;
        bool split = false;
        size_t begin = 0;
        for (const auto& piece : pieces_) {
            const size_t end = std::min(begin + piece.len, n);
            if (begin >= end) break;
            if (end > rewrite_from && (piece.seg.use_count() != 1)) {
                if (!add(total, end - std::max(begin, rewrite_from))) return SIZE_MAX;
                split = split || (begin < rewrite_from && rewrite_from < end);
            }
            begin += piece.len;
        }
        if (split) {
            size_t needed = count;
            if (!add(needed, 1)) return SIZE_MAX;
            if (needed > std::max(pieces_.capacity(), count) &&
                (needed > SIZE_MAX / sizeof(Piece) || !add(total, needed * sizeof(Piece)))) return SIZE_MAX;
        }
        return total;
    }
    void resize(size_t n, uint8_t value = 0) {
        if (n > size_) { grow(n, value); return; }
        while (!pieces_.empty() && size_ - pieces_.back().len >= n) {
            size_ -= pieces_.back().len;
            pieces_.pop_back();
        }
        if (size_ > n) {
            pieces_.back().len -= size_ - n;
            size_ = n;
        }
    }
    void pop_back() { resize(size_ - 1); }

    // Adopt `other`'s first `prefix_bytes` bytes without copying.  The source keeps
    // its own tail: the piece that holds the prefix's end is shared too, so neither
    // side copies a byte until it writes into that piece.
    void share_prefix(const ConversationBuffer& other, size_t prefix_bytes) {
        const size_t n = std::min(prefix_bytes, other.size_);
        if (&other == this) { resize(n); return; }
        pieces_.clear();
        size_ = 0;
        for (const auto& piece : other.pieces_) {
            if (size_ >= n) break;
            Piece adopted = piece;
            adopted.len = std::min(piece.len, n - size_);
            size_ += adopted.len;
            pieces_.push_back(adopted);
        }
    }

    // fn(pointer, count, absolute_offset); false aborts a transfer immediately.
    // The non-const form detaches the shared bytes it exposes, and only those.
    template<class Fn> bool visit(size_t offset, size_t count, Fn fn) {
        if (offset > size_ || count > size_ - offset) return false;
        detach(offset, count);
        return visit_impl(*this, offset, count, fn);
    }
    template<class Fn> bool visit(size_t offset, size_t count, Fn fn) const {
        return visit_impl(*this, offset, count, fn);
    }
    bool read(void* target, size_t offset, size_t count) const {
        return visit(offset, count, [&](const uint8_t* p, size_t n, size_t at) {
            std::memcpy(static_cast<uint8_t*>(target) + at - offset, p, n);
            return true;
        });
    }
    bool operator==(const ConversationBuffer& other) const {
        if (size_ != other.size_) return false;
        size_t at = 0, a = 0, b = 0, x = 0, y = 0;
        while (at < size_) {
            const auto& left = *pieces_[a].seg;
            const auto& right = *other.pieces_[b].seg;
            const size_t n = std::min(std::min(pieces_[a].len - x, other.pieces_[b].len - y), size_ - at);
            if (std::memcmp(left.data() + pieces_[a].start + x,
                            right.data() + other.pieces_[b].start + y, n)) return false;
            at += n; x += n; y += n;
            if (x == pieces_[a].len) { ++a; x = 0; }
            if (y == other.pieces_[b].len) { ++b; y = 0; }
        }
        return true;
    }
    bool operator!=(const ConversationBuffer& other) const { return !(*this == other); }

private:
    using Segment = std::vector<uint8_t>;
    // A byte range within one physical allocation.
    struct Piece {
        std::shared_ptr<Segment> seg;
        size_t start = 0, len = 0;
    };
    // A fresh capture allocates exactly its payload. Later appends reserve
    // geometrically growing segments, bounded by 16 MiB. Small chat turns then
    // extend the final allocation instead of adding one transfer per turn.
    // The reserve is also bounded by an eighth of what the buffer held before
    // the append (at least 64 KiB): a parked conversation carries it in every
    // K/V buffer, and at up to 16 MiB each it came to 0.4-0.5 GB of a ~1 GB
    // conversation (Qwen3.8-Flash-Next), counted against the cache's budget.
    static size_t segment_capacity(size_t count, size_t previous, bool growing, size_t held) {
        if (!growing) return count;
        const size_t geometric = std::min(segment_bytes, std::max<size_t>(65536, std::min(previous, segment_bytes/2)*2));
        return std::max(count, std::min(geometric, std::max<size_t>(65536, held / 8)));
    }
    static bool add(size_t& n, size_t extra) {
        if (extra > SIZE_MAX - n) return false;
        n += extra;
        return true;
    }
    // Split a piece so a range boundary falls on a piece boundary: both halves keep
    // sharing the allocation until one of them is written.
    void split_piece(size_t index, size_t len) {
        Piece tail = pieces_[index];
        tail.len -= len;
        tail.start += len;
        pieces_[index].len = len;
        pieces_.insert(pieces_.begin() + (std::ptrdiff_t) index + 1, tail);
    }
    // Give this piece its own allocation, holding its own bytes and nothing else:
    // the prefix, the spare capacity and the other owner's tail are not copied.
    void unshare_piece(size_t index) {
        Piece& piece = pieces_[index];
        if ((piece.seg.use_count() == 1)) return;
        auto copy = std::make_shared<Segment>();
        copy->reserve(piece.len);
        copy->assign(piece.seg->begin() + piece.start, piece.seg->begin() + piece.start + piece.len);
        piece.seg = std::move(copy);
        piece.start = 0;
    }
    void detach(size_t offset, size_t count) {
        if (!count) return;
        const size_t stop = offset + count;
        size_t begin = 0, splits = 0;
        for (const auto& piece : pieces_) {
            const size_t end = begin + piece.len;
            if ((piece.seg.use_count() != 1)) {
                splits += begin < offset && offset < end;
                splits += begin < stop && stop < end;
            }
            begin = end;
            if (begin >= stop) break;
        }
        if (splits) pieces_.reserve(pieces_.size() + splits);
        begin = 0;
        for (size_t i = 0; i < pieces_.size() && begin < stop;) {
            const size_t len = pieces_[i].len;
            if (begin + len <= offset || (pieces_[i].seg.use_count() == 1)) { begin += len; ++i; continue; }
            if (offset > begin) { split_piece(i, offset - begin); continue; }
            if (len > stop - begin) { split_piece(i, stop - begin); continue; }
            unshare_piece(i);
            begin += pieces_[i].len;
            ++i;
        }
    }
    void grow(size_t n, uint8_t value) {
        const bool growing = size_ != 0;
        const size_t held = size_;
        size_t extend = 0;
        if (!pieces_.empty() && (pieces_.back().seg.use_count() == 1)) {
            Piece& last = pieces_.back();
            // Bytes past this buffer's end are not its own: dropping them keeps a
            // rewind + regrow filling the suffix, as an unshared buffer does.
            if (last.seg->size() != last.start + last.len) last.seg->resize(last.start + last.len);
            extend = std::min(n - size_, last.seg->capacity() - last.start - last.len);
        }
        const size_t extra = n - size_ - extend;
        pieces_.reserve(pieces_.size() + extra / segment_bytes + (extra % segment_bytes != 0));
        if (extend) {
            pieces_.back().seg->resize(pieces_.back().start + pieces_.back().len + extend, value);
            pieces_.back().len += extend;
            size_ += extend;
        }
        while (size_ < n) {
            const size_t count = std::min(segment_bytes, n - size_);
            const size_t previous = pieces_.empty() || (pieces_.back().seg.use_count() != 1) ?
                0 : pieces_.back().seg->capacity();
            auto segment = std::make_shared<Segment>();
            segment->reserve(segment_capacity(count, previous, growing, held));
            segment->resize(count, value);
            pieces_.push_back(Piece{std::move(segment), 0, count});
            size_ += count;
        }
    }
    template<class Self, class Fn> static bool visit_impl(Self& self, size_t offset, size_t count, Fn fn) {
        if (offset > self.size_ || count > self.size_ - offset) return false;
        size_t begin = 0;
        for (auto& piece : self.pieces_) {
            if (!count) break;
            const size_t end = begin + piece.len;
            if (offset < end) {
                const size_t n = std::min(count, end - offset);
                using Byte = std::conditional_t<std::is_const_v<Self>, const uint8_t, uint8_t>;
                Byte* data = piece.seg->data() + piece.start + offset - begin;
                if (!fn(data, n, offset)) return false;
                offset += n; count -= n;
            }
            begin = end;
        }
        return count == 0;
    }
    std::vector<Piece> pieces_;
    size_t size_ = 0;
};

// A captured recurrent-state section (gdn/ple/tails/dead/block_pos).  A copy
// shares the bytes, so forking an archive costs no state bytes at all, and the
// first write makes this payload private again - the original is never touched.
// `vec()` is the plain byte vector behind it, for the disk serializers.
class ConversationPayload {
public:
    ConversationPayload() = default;
    ConversationPayload(std::initializer_list<uint8_t> values)
        : block_(values.size() ? std::make_shared<std::vector<uint8_t>>(values) : nullptr) {}
    // Assigning a plain vector is the same as writing the payload: it becomes
    // this payload's own allocation.
    ConversationPayload& operator=(const std::vector<uint8_t>& values) {
        block_ = values.empty() ? nullptr : std::make_shared<std::vector<uint8_t>>(values);
        return *this;
    }
    ConversationPayload& operator=(std::vector<uint8_t>&& values) {
        block_ = values.empty() ? nullptr : std::make_shared<std::vector<uint8_t>>(std::move(values));
        return *this;
    }
    ConversationPayload& operator=(std::initializer_list<uint8_t> values) {
        block_ = values.size() ? std::make_shared<std::vector<uint8_t>>(values) : nullptr;
        return *this;
    }

    size_t size() const { return block_ ? block_->size() : 0; }
    bool empty() const { return !block_ || block_->empty(); }
    size_t capacity() const { return block_ ? block_->capacity() : 0; }
    const uint8_t* data() const { return block_ ? block_->data() : nullptr; }
    uint8_t* data() { return vec().data(); }
    const std::vector<uint8_t>& vec() const {
        static const std::vector<uint8_t> none;
        return block_ ? *block_ : none;
    }
    // The writable form for a caller that overwrites every byte (conversation_checkpoint_save's device copy
    // does): a block shared with another holder (an archived checkpoint, a batch slot) is replaced by a fresh
    // one of the same size WITHOUT copying the bytes that are about to be overwritten.  The shared holder keeps
    // its immutable copy; a checkpoint pool that reuses a dropped checkpoint's vectors therefore never pays a
    // copy-on-write of the whole running state.
    uint8_t* writable_all() {
        if (!block_) return nullptr;
        if ((block_.use_count() != 1)) block_ = std::make_shared<std::vector<uint8_t>>(block_->size());
        return block_->data();
    }
    // The writable form detaches this payload from every other holder first.
    std::vector<uint8_t>& vec() {
        if (block_ && (block_.use_count() != 1)) block_ = std::make_shared<std::vector<uint8_t>>(*block_);
        if (!block_) block_ = std::make_shared<std::vector<uint8_t>>();
        return *block_;
    }
    void resize(size_t n, uint8_t value = 0) {
        if (n == 0) { block_.reset(); return; }
        if (block_ && block_->size() == n) return;
        vec().resize(n, value);
    }
    void assign(size_t n, uint8_t value) { vec().assign(n, value); }
    void clear() { block_.reset(); }
    void push_back(uint8_t value) { vec().push_back(value); }
    void pop_back() { if (block_) vec().pop_back(); }
    uint8_t operator[](size_t index) const { return vec()[index]; }
    uint8_t& operator[](size_t index) { return vec()[index]; }
    bool operator==(const ConversationPayload& other) const { return vec() == other.vec(); }
    bool operator!=(const ConversationPayload& other) const { return !(*this == other); }
    template<class Fn> void visit_allocations(Fn&& fn) const {
        if (block_) fn((const void*) block_.get(), block_->capacity());
    }
private:
    std::shared_ptr<std::vector<uint8_t>> block_;
};

// A payload compares equal to a plain byte vector of the same bytes, so a caller
// that reads or builds one does not need to know it is shared.
inline bool operator==(const std::vector<uint8_t>& left, const ConversationPayload& right) {
    return left == right.vec();
}
inline bool operator==(const ConversationPayload& left, const std::vector<uint8_t>& right) {
    return left.vec() == right;
}
inline bool operator!=(const std::vector<uint8_t>& left, const ConversationPayload& right) {
    return !(left == right);
}
inline bool operator!=(const ConversationPayload& left, const std::vector<uint8_t>& right) {
    return !(left == right);
}

// Exact unique physical-RAM accounting for shared buffers.  Every allocation is
// charged once while any cache entry holds it, however many entries share it at
// the same time; a view the caller keeps outside the cache neither adds nor
// removes a charge, and dropping one of two branches leaves the shared prefix
// charged to the other.  The table holds one small record per allocation and is
// updated on cache mutations, so no query ever scans a payload.
class ConversationLedger {
public:
    size_t bytes() const { return bytes_; }
    bool holds(const void* allocation) const { return charges_.find(allocation) != charges_.end(); }
    template<class Image> size_t shared_bytes(const Image& image) const {
        if (++query_ == 0) {
            for (const auto& [allocation, charge] : charges_) charge.query = 0;
            ++query_;
        }
        size_t total = 0;
        image.visit_allocations([&](const void* allocation, size_t) {
            const auto it = charges_.find(allocation);
            if (it != charges_.end() && it->second.query != query_) {
                it->second.query = query_;
                total += it->second.bytes;
            }
        });
        return total;
    }
    // Charge `bytes` to `allocation`, once, however many holders adopt it.
    void adopt(const void* allocation, size_t bytes) {
        Charge& charge = charges_[allocation];
        if (charge.holders++ == 0) { charge.bytes = bytes; bytes_ += bytes; }
    }
    // Release one holder; the charge goes when the last one does.
    void release(const void* allocation) {
        const auto it = charges_.find(allocation);
        if (it == charges_.end()) return;
        if (--it->second.holders != 0) return;
        bytes_ -= it->second.bytes;
        charges_.erase(it);
    }
    template<class Image> void adopt(const Image& image) {
        size_t adopted = 0;
        try {
            image.visit_allocations([&](const void* allocation, size_t bytes) {
                adopt(allocation, bytes);
                ++adopted;
            });
        } catch (...) {
            // Allocation visitors have the same order on both walks.
            size_t visited = 0;
            image.visit_allocations([&](const void* allocation, size_t) {
                if (visited++ < adopted) release(allocation);
            });
            throw;
        }
    }
    template<class Image> void release(const Image& image) {
        image.visit_allocations([&](const void* allocation, size_t) { release(allocation); });
    }

private:
    struct Charge {
        size_t bytes = 0;
        size_t holders = 0;
        mutable uint64_t query = 0;
    };
    std::unordered_map<const void*, Charge> charges_;
    size_t bytes_ = 0;
    mutable uint64_t query_ = 0;
};

} // namespace strata::core
