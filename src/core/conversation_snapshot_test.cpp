#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/conversation_file.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_tq.hpp"
#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <cstdlib>
#include <limits>
#include <vector>

using namespace strata::core;
using namespace strata::kernels;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
void cuda_check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e)); std::exit(1); }
}
struct Fixture {
    ModelGeometry g;
    QsaState state;
    std::vector<void*> device, host;
    std::array<void*,5> sources{};
    std::array<size_t,5> sizes{};

    template<class T> void alloc(T*& p, size_t n, bool pinned = false) {
        if (!n) return;
        void* raw = nullptr;
        if (pinned) {
            cuda_check(cudaHostAlloc(&raw, n, cudaHostAllocMapped));
            host.push_back(raw);
            void* mapped = nullptr;
            cuda_check(cudaHostGetDevicePointer(&mapped, raw, 0));
            p = static_cast<T*>(mapped);
        } else {
            cuda_check(cudaMalloc(&raw, n)); device.push_back(raw); p = static_cast<T*>(raw);
        }
    }
    Fixture(int fmt, int mode, int bits = 0) {
        auto& st = state;
        st.kv_mode = mode; st.kv_int8 = fmt==kKvInt8; st.kv_q4 = fmt==kKvQ4; st.kv_hybrid = fmt==3;
        st.kv_tq = fmt==kKvTq; st.kv_tq_bits = st.kv_tq ? bits : 0; st.kv_rot = st.kv_tq;
        st.n_pages = 24; st.n_slots = mode ? 4 : st.n_pages;
        st.max_cells = st.n_pages * 4; st.idx_pooled_rows = mode==2 ? 2 : st.max_cells/4+2;
        const size_t per = st.kv_tq ? kv_tq_bytes_per_head(bits, (int) g.head_dim)
                          : fmt==kKvQ4 ? kv_q4_bytes_per_head((int)g.head_dim)
                                       : g.head_dim*((st.kv_int8 || st.kv_hybrid) ? 1:2);
        const size_t rows = st.max_cells*g.n_head_kv, slot_rows = st.n_slots*4*g.n_head_kv;
        sizes = {rows*per, rows*per, fmt==kKvInt8 ? rows*(g.head_dim/64)*2:0,
                 fmt==kKvInt8 ? rows*(g.head_dim/64)*2:0, (size_t)st.idx_pooled_rows*g.idx_key_dim*4};
        if (fmt==3) {
            sizes = {rows*per, rows*kv_q4_bytes_per_head((int)g.head_dim), rows*(g.head_dim/64)*2, 0,
                     (size_t)st.idx_pooled_rows*g.idx_key_dim*4};
            alloc(st.k_q,sizes[0]); alloc(st.v_q4,sizes[1]); alloc(st.k_scale,sizes[2]);
            sources[0]=st.k_q; sources[1]=st.v_q4; sources[2]=st.k_scale;
        } else if (fmt==kKvTq) {
            // TurboQuant is resident-only (mode 0) and has no host copy: the VRAM arrays are authoritative.
            alloc(st.k_tq,slot_rows*per); alloc(st.v_tq,slot_rows*per);
            sources[0]=st.k_tq; sources[1]=st.v_tq;
        } else if (fmt==kKvQ4) {
            alloc(st.k_q4,slot_rows*per); alloc(st.v_q4,slot_rows*per);
            if (mode) {alloc(st.host.k_q4,sizes[0],true); alloc(st.host.v_q4,sizes[1],true);}
            sources[0]=mode?st.host.k_q4:st.k_q4; sources[1]=mode?st.host.v_q4:st.v_q4;
        } else if (fmt==kKvInt8) {
            alloc(st.k_q,slot_rows*per); alloc(st.v_q,slot_rows*per);
            alloc(st.k_scale,slot_rows*(g.head_dim/64)*2); alloc(st.v_scale,slot_rows*(g.head_dim/64)*2);
            if (mode) {
                alloc(st.host.k_q,sizes[0],true); alloc(st.host.v_q,sizes[1],true);
                alloc(st.host.k_scale,sizes[2],true); alloc(st.host.v_scale,sizes[3],true);
            }
            sources[0]=mode?st.host.k_q:st.k_q; sources[1]=mode?st.host.v_q:st.v_q;
            sources[2]=mode?st.host.k_scale:st.k_scale; sources[3]=mode?st.host.v_scale:st.v_scale;
        } else {
            alloc(st.k_pool,slot_rows*per); alloc(st.v_pool,slot_rows*per);
            if (mode) {alloc(st.host.k_pool,sizes[0],true); alloc(st.host.v_pool,sizes[1],true);}
            sources[0]=mode?st.host.k_pool:st.k_pool; sources[1]=mode?st.host.v_pool:st.v_pool;
        }
        alloc(st.idx_pooled,sizes[4]); sources[4]=st.idx_pooled;
        alloc(st.page_table,st.n_pages*4);
        if (mode==1) {
            auto& m=st.map;
            m.page_table=st.page_table; m.n_blocks=st.n_pages; m.n_slots=st.n_slots;
            alloc(m.slot_block,st.n_slots*4); alloc(m.slot_stamp,st.n_slots*4); alloc(m.slot_ref,st.n_slots*4);
            alloc(m.miss_block,st.n_slots*4); alloc(m.miss_slot,st.n_slots*4); alloc(m.ctl,kKvCtlInts*4);
        } else if (mode==2) {
            kv_ring_table(st.page_table,st.n_pages,st.n_slots,nullptr);
        }
    }
    void fill(uint8_t salt) {
        for (size_t i=0;i<sources.size();++i) {
            std::vector<uint8_t> data(sizes[i]);
            for (size_t j=0;j<data.size();++j) data[j]=(uint8_t)(salt+i*31+j*7+j/257);
            if (!data.empty()) cuda_check(cudaMemcpy(sources[i],data.data(),data.size(),cudaMemcpyDefault));
        }
    }
    void fill_after(uint8_t salt, int64_t first_dirty) {
        for (size_t i=0;i<sources.size();++i) {
            const size_t offset = i == 4 ? size_t(first_dirty/4)*g.idx_key_dim*4
                                         : (sizes[i]/size_t(state.max_cells))*size_t((first_dirty/4)*4);
            if (sizes[i] > offset)
                cuda_check(cudaMemset(static_cast<uint8_t*>(sources[i])+offset, salt, sizes[i]-offset));
        }
    }
    ~Fixture() { for (void* p:device) cudaFree(p); for (void* p:host) cudaFreeHost(p); }
};
bool equal(const ConversationKv& a,const ConversationKv& b) {
    return a.k==b.k && a.v==b.v && a.k_scale==b.k_scale && a.v_scale==b.v_scale && a.pooled==b.pooled;
}

void full_session(int fmt, int mode, int experts, int bits = 0) {
    Fixture main(fmt,mode,bits), draft((fmt==3 || fmt==kKvTq)?kKvInt8:fmt,2);
    auto& g=main.g;
    g.n_layers=4; g.n_expert=experts;
    g.ssm_state_size=2; g.ssm_v_heads=2; g.ssm_conv_channels=8;
    SessionState ss;
    ss.max_cells=96; ss.qsa_states=&main.state;
    ss.layer_hi=g.n_layers; ss.gdn_alloc=g.n_gdn_layers(); ss.qsa_alloc=g.n_qsa_layers();   // the whole-model carve
    std::string err;
    ConversationStateSizes sizes;
    check(conversation_state_sizes(g,sizes,err),"whole-session geometry sizes");
    main.alloc(ss.gdn_state,sizes.gdn); main.alloc(ss.ple_hist,sizes.ple);
    main.alloc(main.state.idx_tail,sizes.tail); main.alloc(main.state.idx_dead,sizes.dead);
    main.alloc(main.state.idx_block_pos,sizes.block_pos);
    std::vector<int32_t> ids(65);
    for (size_t i=0;i<ids.size();++i) ids[i]=(int32_t)i+1;
    std::vector<ConversationImageKey> images;
    std::vector<ConversationCheckpoint> checkpoints;
    auto fill=[&](uint8_t salt) {
        main.fill(salt); draft.fill(salt);
        for (const auto& [p,n] : std::vector<std::pair<void*,size_t>>{
                 {ss.gdn_state,sizes.gdn},{ss.ple_hist,sizes.ple},{main.state.idx_tail,sizes.tail},
                 {main.state.idx_dead,sizes.dead},{main.state.idx_block_pos,sizes.block_pos}})
            cuda_check(cudaMemset(p,salt,n));
        // A real indexer maintains its spare pooled row as a copy of idx_dead.
        cuda_check(cudaMemcpy(main.state.idx_pooled+(ids.size()/4)*g.idx_key_dim,
                              main.state.idx_dead,sizes.dead,cudaMemcpyDeviceToDevice));
        cuda_check(cudaDeviceSynchronize());
    };
    fill(13);
    ConversationCheckpoint checkpoint;
    checkpoint.ids.assign(ids.begin(),ids.begin()+3);
    check(conversation_checkpoint_save(checkpoint,ss,g,err),"save complete running checkpoint");
    checkpoint.used = 17;
    checkpoints.push_back(std::move(checkpoint));
    const ConversationView view{ids,images,checkpoints,true};
    SavedConversation a,b,restored;
    check(conversation_snapshot_save(a,view,ss,g,draft.state,err),"capture complete A");
    check(a.checkpoints[0].used == 17,"upstream checkpoint LRU stamp survives capture");
    fill(177);
    check(conversation_snapshot_save(b,view,ss,g,draft.state,err),"capture complete B");
    {
        // The session has advanced to B. Demotion must use the saved recurrent
        // state at the older checkpoint, not B's current recurrent state.
        const auto& historical = a.checkpoints[0];
        const std::vector<ConversationCheckpoint> none;
        const ConversationView old_view{historical.ids, historical.imgs, none, true, &historical};
        SavedConversation archived;
        check(conversation_snapshot_save(archived, old_view, ss, g, draft.state, err),
              "demote a historical checkpoint with authoritative positional backing");
        check(conversation_snapshot_restore(archived, ss, g, draft.state, err) == ConversationRestore::restored,
              "restore the demoted checkpoint");
        ConversationCheckpoint actual;
        actual.ids = historical.ids;
        check(conversation_checkpoint_save(actual, ss, g, err), "read back restored historical state");
        check(actual.gdn == historical.gdn && actual.ple == historical.ple &&
              actual.tails == historical.tails && actual.dead == historical.dead &&
              actual.block_pos == historical.block_pos,
              "historical restore recovers every recurrent state, not the later session state");
        fill(177);
    }
    {
        // disk save path: metadata + streamed K/V give the same file as the captured image
        SavedConversation meta;
        std::vector<SessionKvSource> sources;
        check(conversation_snapshot_sources(meta,sources,view,ss,g,draft.state,err),"disk-save sources");
        check(meta.kv.empty() && sources.size()==b.kv.size(),"one source per K/V layer, none captured");
        const SessionFileIdentity id{1,2};
        namespace fs=std::filesystem;
        const fs::path dir=fs::temp_directory_path()/("strata-snap-test-"+std::to_string(experts)+"-"+
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(dir);
        const std::string p1=(dir/"streamed.bin").string(), p2=(dir/"captured.bin").string();
        auto slurp=[](const std::string& p){ std::ifstream f(p,std::ios::binary);
            return std::vector<char>((std::istreambuf_iterator<char>(f)),std::istreambuf_iterator<char>()); };
        size_t n1=0,n2=0;
        check(session_file_write(p1,meta,sources,id,n1,err),"streamed session write");
        check(session_file_write(p2,b,id,n2,err),"captured session write");
        SavedConversation x,y;
        check(session_file_read(p1,id,x,n1,err) && session_file_read(p2,id,y,n2,err) && n1==n2,"both files read");
        check(x.live.gdn==y.live.gdn && x.live.dead==y.live.dead && x.live.ids==y.live.ids &&
              x.checkpoints.size()==y.checkpoints.size() && equal(x.kv[0],y.kv[0]) && equal(x.kv.back(),y.kv.back()),
              "streamed file equals captured file");
        check(slurp(p1)==slurp(p2),"streamed file is byte-identical to the captured file");
        std::error_code ec;
        fs::remove_all(dir,ec);
        auto empty_ids=std::vector<int32_t>{};
        const ConversationView none{empty_ids,images,checkpoints,true};
        check(!conversation_snapshot_sources(meta,sources,none,ss,g,draft.state,err),"empty session refused");
    }
    check(a.live.dead!=b.live.dead,"different opening-state spare keys in regression fixture");
    auto bad=a; bad.kv.back().k.pop_back();
    check(conversation_snapshot_restore(bad,ss,g,draft.state,err)==ConversationRestore::invalid,
          "reject invalid late draft before any main-layer write");
    check(conversation_snapshot_save(restored,view,ss,g,draft.state,err),"capture B after refused restore");
    check(restored.live.gdn==b.live.gdn && restored.live.dead==b.live.dead &&
          equal(restored.kv[0],b.kv[0]) && equal(restored.kv[1],b.kv[1]),"refusal preserves all tested state");
    check(conversation_snapshot_restore(a,ss,g,draft.state,err)==ConversationRestore::restored,"restore complete A");
    check(conversation_snapshot_save(restored,view,ss,g,draft.state,err),"capture restored A");
    check(restored.live.gdn==a.live.gdn && restored.live.ple==a.live.ple && restored.live.tails==a.live.tails &&
          restored.live.dead==a.live.dead && restored.live.block_pos==a.live.block_pos &&
          equal(restored.kv[0],a.kv[0]) && equal(restored.kv[1],a.kv[1]),"whole-session A/B/A exactness including spare key");
    check(ss.ple_prev[0]==64 && ss.ple_prev[1]==65,"PLE token window reconstructed");
    {
        // A shared prefix (ConversationCache::fork): the K/V of A's first tokens and A's checkpoint there, A
        // untouched - the same bytes as restoring A whole and rewinding to that checkpoint
        const int64_t at = (int64_t) a.checkpoints[0].ids.size();
        check(conversation_snapshot_restore(a,ss,g,draft.state,err)==ConversationRestore::restored,"restore A whole");
        check(conversation_checkpoint_restore(a.checkpoints[0],ss,g,err),"rewind it to its checkpoint");
        ConversationKv want_main,want_draft,got_main,got_draft;
        ConversationCheckpoint want_state,got_state;
        want_state.ids=got_state.ids=a.checkpoints[0].ids;
        check(conversation_kv_save(want_main,main.state,g,at,true,err) &&
              conversation_kv_save(want_draft,draft.state,g,at,false,err) &&
              conversation_checkpoint_save(want_state,ss,g,err),"capture the rewound prefix");
        fill(201);   // another conversation in the session
        const SavedConversation untouched=a;
        check(conversation_snapshot_restore_prefix(a,(int64_t)a.live.ids.size()+1,ss,g,draft.state,err)==
                  ConversationRestore::invalid &&
              conversation_snapshot_restore_prefix(a,0,ss,g,draft.state,err)==ConversationRestore::invalid,
              "a prefix outside the image is refused");
        check(conversation_snapshot_restore_prefix(a,at,ss,g,draft.state,err)==ConversationRestore::restored,
              "restore A's prefix only");
        check(conversation_checkpoint_restore(a.checkpoints[0],ss,g,err),"mount the checkpoint over it");
        check(conversation_kv_save(got_main,main.state,g,at,true,err) &&
              conversation_kv_save(got_draft,draft.state,g,at,false,err) &&
              conversation_checkpoint_save(got_state,ss,g,err),"capture the borrowed prefix");
        check(equal(got_main,want_main) && equal(got_draft,want_draft),
              "borrowed K/V equals the whole restore's prefix");
        check(got_state.gdn==want_state.gdn && got_state.ple==want_state.ple && got_state.tails==want_state.tails &&
              got_state.dead==want_state.dead && got_state.block_pos==want_state.block_pos,"the same running state");
        check(untouched.live.ids==a.live.ids && equal(untouched.kv[0],a.kv[0]) && equal(untouched.kv[1],a.kv[1]),
              "the parked image is unchanged");
    }
    for (int64_t dirty : {65, 3, 0}) {
        check(conversation_snapshot_restore(a,ss,g,draft.state,err)==ConversationRestore::restored,"restore growth fixture base");
        ConversationKvReuse reuse{a.kv,65,dirty};
        const uint8_t* original = nullptr;
        reuse.kv[0].k.visit(0,1,[&](const uint8_t* p,size_t,size_t){original=p;return true;});
        main.fill_after(91,dirty); draft.fill_after(91,std::max<int64_t>(0,dirty-1));
        cuda_check(cudaMemset(ss.gdn_state,91,sizes.gdn));
        ids.resize(70);
        for (size_t i=65;i<ids.size();++i) ids[i]=int32_t(i+1);
        cuda_check(cudaMemcpy(main.state.idx_pooled+(ids.size()/4)*g.idx_key_dim,
                              main.state.idx_dead,sizes.dead,cudaMemcpyDeviceToDevice));
        SavedConversation fresh,incremental;
        check(conversation_snapshot_save(fresh,view,ss,g,draft.state,err),"full capture reference after growth or rewind");
        size_t peak=0,reused=0;
        check(conversation_snapshot_capture_bytes(reuse,view,ss,g,draft.state,peak,err),"admit incremental capture peak");
        // The retained K/V is adopted and the checkpoints are adopted by reference: only the
        // allocations on top of what `a` already holds are new physical RAM, and the admission
        // (peak minus the retained capacity it already counts) has to cover them.
        const size_t retained=reuse.bytes();
        ConversationLedger held;
        for (const auto& layer : a.kv) held.adopt(layer);
        for (const auto& c : checkpoints) held.adopt(c);
        check(conversation_snapshot_save(incremental,view,ss,g,draft.state,err,std::move(reuse),&reused),"capture with retained pages");
        size_t fresh_bytes=0;
        incremental.visit_allocations([&](const void* allocation,size_t bytes){
            if (!held.holds(allocation)) fresh_bytes+=bytes; });
        check(peak>=retained && fresh_bytes<=peak-retained,
              "the capture's new physical allocation stays within the admitted bound");
        check(incremental.live.gdn==fresh.live.gdn && incremental.live.ple==fresh.live.ple &&
              incremental.live.dead==fresh.live.dead && equal(incremental.kv[0],fresh.kv[0]) &&
              equal(incremental.kv[1],fresh.kv[1]),"incremental capture equals full capture after growth or rewind");
        check((dirty>=4)==(reused>0),"only complete unchanged pages or rows are retained");
        incremental.kv[0].k.visit(0,1,[&](const uint8_t* p,size_t,size_t){
            check(p==original,"growth never reallocates the retained payload");return true;
        });
        check(conversation_snapshot_restore(incremental,ss,g,draft.state,err)==ConversationRestore::restored,"restore segmented incremental snapshot");
        uint64_t fingerprint=0;
        check(conversation_kv_verify(incremental.kv.back(),draft.state,g,70,false,fingerprint,err),"incremental draft authoritative and ring read-back");
        ids.resize(65);
    }
    {
        // a layer split's later stage: the same image WITHOUT the draft layer's K/V (draft == nullptr)
        check(conversation_snapshot_restore(a,ss,g,draft.state,err)==ConversationRestore::restored,"restore A before stage images");
        size_t with_draft=0,without=0;
        check(conversation_snapshot_bytes(view,ss,g,draft.state,with_draft,err) &&
              conversation_snapshot_bytes(view,ss,g,nullptr,without,err) && without<with_draft,
              "a stage image's estimate leaves the draft out");
        SavedConversation stage,stage_b,stage_back;
        check(conversation_snapshot_save(stage,view,ss,g,nullptr,err),"capture a stage image");
        check(stage.kv.size()==a.kv.size()-1 && equal(stage.kv[0],a.kv[0]) && stage.live.gdn==a.live.gdn,
              "a stage image holds the session's own layers, no draft");
        check(stage.bytes()<a.bytes(),"a stage image is smaller by the draft ring");
        check(conversation_snapshot_validate(stage,ss,g,nullptr,err),"validate a stage image without a draft");
        check(!conversation_snapshot_validate(stage,ss,g,draft.state,err),"a stage image is refused where a draft is expected");
        check(!conversation_snapshot_validate(a,ss,g,nullptr,err),"an image with a draft is refused where none is expected");
        fill(201);
        check(conversation_snapshot_save(stage_b,view,ss,g,nullptr,err),"capture another stage state");
        check(conversation_snapshot_restore(a,ss,g,nullptr,err)==ConversationRestore::invalid,
              "restoring a draft image as a stage image is refused before any write");
        check(conversation_snapshot_save(stage_back,view,ss,g,nullptr,err) && stage_back.live.gdn==stage_b.live.gdn &&
              equal(stage_back.kv[0],stage_b.kv[0]),"the refusal left the stage untouched");
        check(conversation_snapshot_restore(stage,ss,g,nullptr,err)==ConversationRestore::restored,"restore a stage image");
        check(conversation_snapshot_save(stage_back,view,ss,g,nullptr,err) && stage_back.live.gdn==stage.live.gdn &&
              stage_back.live.dead==stage.live.dead && equal(stage_back.kv[0],stage.kv[0]),"stage A/B/A exactness");
        // its retained K/V: the next park copies only what changed, and equals a full capture
        ConversationKvReuse reuse{stage.kv,65,65,{}};
        SavedConversation fresh,incremental;
        size_t peak=0,reused=0;
        check(conversation_snapshot_save(fresh,view,ss,g,nullptr,err),"full stage capture reference");
        check(conversation_snapshot_capture_bytes(reuse,view,ss,g,nullptr,peak,err),"admit an incremental stage capture");
        const size_t retained=reuse.bytes();
        ConversationLedger held;
        for (const auto& layer : stage.kv) held.adopt(layer);
        for (const auto& c : checkpoints) held.adopt(c);
        check(conversation_snapshot_save(incremental,view,ss,g,nullptr,err,std::move(reuse),&reused),"incremental stage capture");
        size_t fresh_bytes=0;
        incremental.visit_allocations([&](const void* allocation,size_t bytes){
            if (!held.holds(allocation)) fresh_bytes+=bytes; });
        check(reused>0 && peak>=retained && fresh_bytes<=peak-retained &&
              equal(incremental.kv[0],fresh.kv[0]) && incremental.live.gdn==fresh.live.gdn,
              "an incremental stage capture reuses pages within the admitted bound and equals a full one");
        SavedConversation split_image=stage;
        split_image.stage_images.push_back(stage);
        check(!conversation_snapshot_validate(split_image,ss,g,draft.state,err),
              "a layer split's image is refused by the whole-session form");
        ConversationKvReuse wrong{a.kv,65,65,{}};
        check(!conversation_snapshot_capture_bytes(wrong,view,ss,g,nullptr,peak,err),"a draft image's K/V is not a stage's reuse");
    }
    check(conversation_snapshot_restore(a,ss,g,draft.state,err)==ConversationRestore::restored,"restore A after stage images");
    check(conversation_checkpoint_restore(a.checkpoints[0],ss,g,err),"restore early running checkpoint");
    std::vector<uint8_t> spare(sizes.dead);
    cuda_check(cudaMemcpy(spare.data(),main.state.idx_pooled,sizes.dead,cudaMemcpyDeviceToHost));
    check(spare==a.checkpoints[0].dead,"checkpoint rebuilds spare row over a later completed block");
    check(ss.ple_prev[0]==2 && ss.ple_prev[1]==3,"checkpoint PLE token window");
}
}

// The real TurboQuant state path, not the snapshot fixture: qsa_state_bytes sizes the arena, qsa_state_init
// carves the two arrays out of it and qsa_attn_pools exposes them.  At every width the carved K/V extents must
// stay inside the bytes init reports (so kv_pool_bytes' TQ arm and init agree) and each array must cover exactly
// rows * kv_tq_bytes_per_head, touching nothing else in the arena.
void tq_state_sizing() {
    ModelGeometry g;   // head_dim 256 == kernels::QK_TQ, the only width layout() accepts
    const int64_t cells = 96;
    uint64_t last_arena = 0;
    for (int bits : {3,4,6,8}) {
        qsa_set_kv_tq(bits);
        const uint64_t arena = qsa_state_bytes(g, cells);
        check(arena > last_arena, "a wider TurboQuant block needs more state bytes");
        last_arena = arena;
        const QsaShapes s = qsa_shapes(g);
        const uint64_t row = kv_tq_bytes_per_head(bits, (int) s.head_dim);
        const uint64_t pages = (uint64_t) ((cells + s.page_size - 1) / s.page_size);
        // Guard slack: if qsa_state_bytes' pools undersized what qsa_state_init carves, init must land in
        // mapped memory so the used<=arena assertion reports it instead of faulting first.
        uint8_t* base = nullptr;
        cuda_check(cudaMalloc(&base, arena + 2 * pages * s.n_head_kv * s.page_size * row));
        QsaState st;
        const uint64_t used = qsa_state_init(g, cells, base, st);
        check(used > 0 && used <= arena, "TQ state fits the arena qsa_state_bytes sized");
        check(st.kv_tq && st.kv_tq_bits == bits && st.kv_mode == 0, "TQ state initializes resident at its width");
        const uint64_t rows = (uint64_t) st.n_slots * s.n_head_kv * s.page_size;
        check(st.k_tq != nullptr && st.v_tq != nullptr, "TQ state carves both K/V arrays");
        const size_t ko = (size_t) ((const uint8_t*) st.k_tq - base);
        const size_t vo = (size_t) ((const uint8_t*) st.v_tq - base);
        check(ko + rows*row <= used && vo + rows*row <= used,
              "both TQ arrays fit the state bytes qsa_state_init reports");
        check(ko + rows*row <= vo || vo + rows*row <= ko, "TQ K and V arrays do not overlap");
        const auto pools = qsa_attn_pools(st);
        check(pools.k_tq == st.k_tq && pools.v_tq == st.v_tq && pools.tq_bits == bits,
              "qsa_attn_pools exposes the TQ arrays and width");
        // Canary the whole arena, write each array's full extent, and read it back: only the two carved
        // ranges may change, so a short carve or an over-wide row cannot pass unnoticed.
        std::vector<uint8_t> canary((size_t) arena, 0xa5), got((size_t) arena);
        cuda_check(cudaMemcpy(base, canary.data(), canary.size(), cudaMemcpyHostToDevice));
        cuda_check(cudaMemset(st.k_tq, 0x5a, rows*row));
        cuda_check(cudaMemset(st.v_tq, 0x3c, rows*row));
        cuda_check(cudaMemcpy(got.data(), base, got.size(), cudaMemcpyDeviceToHost));
        bool exact = true;
        for (size_t i = 0; i < got.size() && exact; ++i)
            exact = got[i] == (i >= ko && i < ko + rows*row ? 0x5a
                              : i >= vo && i < vo + rows*row ? 0x3c : 0xa5);
        check(exact, "the two TQ arrays cover exactly their carved extents");
        cuda_check(cudaFree(base));
    }
    qsa_set_kv_tq(0);
}

int main() {
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) return 77;
    tq_state_sizing();
    for (int fmt : std::array<int,5>{kKvF16,kKvInt8,kKvQ4,3,kKvTq})
     for (int bits : fmt==kKvTq ? std::vector<int>{3,4,6,8} : std::vector<int>{0})
      for (int mode : {0,1,2}) {
        if ((fmt==3 || fmt==kKvTq) && mode!=0) continue;
        Fixture f(fmt,mode,bits);
        if (fmt==3 || fmt==kKvTq) {
            std::string err;
            ConversationKv invalid;
            for (int unsupported : {1,2}) {
                f.state.kv_mode=unsupported;
                check(conversation_kv_bytes(f.state,f.g,9,true)==0,
                      fmt==3 ? "hybrid streaming/ring rejected without block-mover abort"
                             : "TurboQuant streaming/ring rejected without block-mover abort");
                check(!conversation_kv_save(invalid,f.state,f.g,9,true,err),
                      fmt==3 ? "hybrid streaming/ring capture fails closed"
                             : "TurboQuant streaming/ring capture fails closed");
            }
            f.state.kv_mode=0;
        }
        {
            std::string err;
            ConversationKv invalid;
            for (const int64_t bad : {-1LL, 97LL, (long long) std::numeric_limits<int64_t>::max()}) {
                check(!conversation_kv_save(invalid,f.state,f.g,bad,mode!=2,err),"reject invalid save extent before allocation");
                check(!conversation_kv_restore(invalid,f.state,f.g,bad,mode!=2,err),"reject invalid restore extent before copying");
                check(conversation_kv_bytes(f.state,f.g,bad,mode!=2)==0,"invalid extent cannot overflow byte estimate");
            }
        }
        for (int64_t upto : {0,1,3,4,5,63,64,65,96}) {
            const bool index=mode!=2;
            std::string err;
            f.fill(13);
            ConversationKv a,b,restored;
            cuda_check(cudaDeviceSynchronize());
            check(conversation_kv_save(a,f.state,f.g,upto,index,err),"save A");
            check(a.bytes()==conversation_kv_bytes(f.state,f.g,upto,index),"size estimate equals snapshot payload");
            if (upto >= 9 && upto < 96) {
                const int64_t anchor = (upto / 2 / 4) * 4;
                f.fill_after(91, anchor);
                bool same = false;
                check(conversation_kv_prefix_matches(a, f.state, f.g, upto, index, same, err) && !same,
                      "rewritten reply cells invalidate the terminal state");
                check(conversation_kv_prefix_matches(a, f.state, f.g, anchor, index, same, err) && same,
                      "rewritten reply cells do not invalidate earlier historical checkpoints");
                ConversationKv incremental = a, fresh;
                size_t reused = 0;
                check(conversation_kv_save(fresh, f.state, f.g, upto, index, err) &&
                      conversation_kv_save(incremental, f.state, f.g, upto, index, err, anchor, &reused),
                      "replace rewritten reply cells in the existing backing");
                check(reused > 0 && equal(incremental, fresh),
                      "suffix replacement retains historical pages and equals a full capture");
                f.fill(13);
            }
            if (upto > 0 && upto % 4) {
                bool same = false;
                const size_t per = f.sizes[0] / size_t(f.state.max_cells * f.g.n_head_kv);
                const size_t page = size_t(upto / 4) * 4 * f.g.n_head_kv * per;
                const size_t head = size_t(f.g.n_head_kv - 1) * 4 * per;
                cuda_check(cudaMemset(static_cast<uint8_t*>(f.sources[0]) + page + head + size_t(upto % 4) * per,
                                     99, per));
                check(conversation_kv_prefix_matches(a, f.state, f.g, upto, index, same, err) && same,
                      "unused cells in a partial multi-head page do not invalidate the prefix");
                cuda_check(cudaMemset(static_cast<uint8_t*>(f.sources[0]) + page + head + size_t(upto % 4 - 1) * per,
                                     99, per));
                check(conversation_kv_prefix_matches(a, f.state, f.g, upto, index, same, err) && !same,
                      "the last live cell of the last head participates in exact prefix validation");
                f.fill(13);
            }
            if (upto > 4 && upto < 96) {
                ConversationCache cache(1 << 24, 4);
                SavedConversation archive;
                archive.live.ids.resize(upto, 1);
                archive.checkpoint_archive = true;
                archive.unchanged_tokens = upto;
                archive.kv.push_back(a);
                check(cache.put(std::move(archive)), "park branch before full prefill");
                cache.limit_reuse(0);
                ConversationCheckpoint next;
                next.ids.resize(upto + 1, 1);
                const auto match = cache.archive_for(next, true);
                check(match.tokens == upto, "full prefill keeps the matching archive discoverable");
                auto previous = cache.take(match.index);
                bool same = false;
                check(conversation_kv_prefix_matches(previous.kv[0], f.state, f.g, upto, index, same, err) && same,
                      "exact prefix validation recovers archive reuse after a full prefill");
                size_t reused = 0;
                ConversationKv fresh;
                check(conversation_kv_save(fresh, f.state, f.g, upto + 1, index, err),
                      "capture full-prefill continuation reference");
                check(conversation_kv_save(previous.kv[0], f.state, f.g, upto + 1, index, err, upto, &reused),
                      "append to recovered archive");
                check(reused > 0 && equal(previous.kv[0], fresh),
                      "recovered append retains pages and equals a fresh capture");
                previous.checkpoints.push_back(std::move(previous.live));
                previous.live = next;
                previous.unchanged_tokens = upto + 1;
                check(cache.put(std::move(previous)) && cache.size() == 1 && cache.evictions() == 0,
                      "full-prefill continuation grows one archive without a spill");
                f.fill(177);
                check(conversation_kv_prefix_matches(a, f.state, f.g, upto, index, same, err) && !same,
                      "matching tokens do not authorize reuse of different K/V bytes");
                f.fill(13);
            }
            f.fill(177);
            check(conversation_kv_save(b,f.state,f.g,upto,index,err),"save B");
            check(upto==0 || !equal(a,b),"fixture changes state");
            check(conversation_kv_restore(a,f.state,f.g,upto,index,err),"restore A over B");
            cuda_check(cudaDeviceSynchronize());
            check(conversation_kv_save(restored,f.state,f.g,upto,index,err),"read back restored A");
            check(equal(a,restored),"A/B/A byte-exact K/V and indexer state");
            uint64_t fingerprint = 0;
            check(conversation_kv_verify(a,f.state,f.g,upto,index,fingerprint,err),"verify restored authoritative and resident bytes without rewriting them");
            {
                // disk save streams the same bytes as a captured image, without the host copy
                SessionKvSource src;
                check(conversation_kv_source(src,f.state,f.g,upto,index,err),"K/V source for a disk save");
                check(src.format==a.format && src.cells==a.cells && src.heads==a.heads && src.head_dim==a.head_dim &&
                      src.page_size==a.page_size && src.pooled_rows==a.pooled_rows && src.idx_dim==a.idx_dim,
                      "K/V source metadata equals the captured image");
                const std::array<const ConversationBuffer*,5> parts={&a.k,&a.v,&a.k_scale,&a.v_scale,&a.pooled};
                for (size_t i=0;i<5;++i) {
                    check(src.sizes[i]==parts[i]->size(),"K/V source part size");
                    std::vector<uint8_t> got(src.sizes[i]), want(src.sizes[i]);
                    const size_t half=src.sizes[i]/2;   // two reads: offsets are honoured
                    check(src.read(i,0,got.data(),half) && src.read(i,half,got.data()+half,src.sizes[i]-half),
                          "K/V source reads");
                    check(parts[i]->read(want.data(),0,want.size()) && got==want,"K/V source bytes equal the image");
                }
                SessionKvSource none;
                check(!conversation_kv_source(none,f.state,f.g,-1,index,err),"K/V source rejects an invalid extent");
            }
            if (mode==1) {
                std::vector<int32_t> table((size_t)f.state.n_pages);
                cuda_check(cudaMemcpy(table.data(),f.state.page_table,table.size()*4,cudaMemcpyDeviceToHost));
                check(std::all_of(table.begin(),table.end(),[](int32_t x){return x==-1;}),"stale VRAM pages invalidated");
            }
            if (mode==2 && upto>0) {
                const auto s=qsa_real_shapes();
                const int64_t b1=(upto+s.page_size-1)/s.page_size,b0=std::max<int64_t>(0,b1-f.state.n_slots);
                const size_t block=kv_block_bytes(s,fmt)/2;
                // Codes and scales are separate in INT8; check code bytes here.
                const size_t code_block=fmt==kKvInt8 ? s.page_size*s.n_head_kv*s.head_dim : block;
                const void* slots=fmt==kKvQ4?(void*)f.state.k_q4:fmt==kKvInt8?(void*)f.state.k_q:(void*)f.state.k_pool;
                std::vector<uint8_t> got(code_block);
                for (int64_t page=b0;page<b1;++page) {
                    cuda_check(cudaMemcpy(got.data(),(const uint8_t*)slots+(page%f.state.n_slots)*code_block,code_block,cudaMemcpyDeviceToHost));
                    std::vector<uint8_t> expected(code_block);
                    check(a.k.read(expected.data(), page*code_block, code_block),"read segmented expected draft page");
                    check(got == expected,"draft ring contains restored pages");
                }
            }
            auto bad=a; bad.head_dim++;
            check(!conversation_kv_restore(bad,f.state,f.g,upto,index,err),"reject incompatible geometry");
            if (fmt==kKvInt8) {
                // #293: INT8 K/V saved through the Hadamard rotation (STRATA_KV_ROT=1) never restores into a state
                // without it, nor the reverse
                f.state.kv_rot=true;
                ConversationKv rotated;
                check(conversation_kv_save(rotated,f.state,f.g,upto,index,err),"save rotated INT8");
                check(!conversation_kv_restore(a,f.state,f.g,upto,index,err),"unrotated image into a rotated state rejected");
                f.state.kv_rot=false;
                check(!conversation_kv_restore(rotated,f.state,f.g,upto,index,err),"rotated image into an unrotated state rejected");
                check(conversation_kv_restore(a,f.state,f.g,upto,index,err),"unrotated image still restores");
            }
            if (!a.k.empty()) {
                bad=a; bad.k.pop_back();
                check(!conversation_kv_restore(bad,f.state,f.g,upto,index,err),"reject malformed payload");
            }
            if (fmt==kKvTq && upto>0) {
                // The width is deliberately NOT in the format tag: a tq3 image restored into a tq6 state has to
                // fail the payload-size check instead of decoding into the wrong codebook.
                const int other = bits==3 ? 4 : bits==4 ? 6 : bits==6 ? 8 : 3;
                f.state.kv_tq_bits = other;
                check(!conversation_kv_restore(a,f.state,f.g,upto,index,err),
                      "TurboQuant image of another width is refused by its payload size");
                f.state.kv_tq_bits = bits;
                check(conversation_kv_restore(a,f.state,f.g,upto,index,err),
                      "the same-width TurboQuant image still restores");
            }
        }
    }
    for (int fmt : {kKvF16,kKvInt8,kKvQ4}) for (int mode : {0,1}) for (int experts : {256,512})
        full_session(fmt,mode,experts);
    for (int experts : {256,512}) full_session(3,0,experts);
    for (int bits : {3,4,6,8}) for (int experts : {256,512}) full_session(kKvTq,0,experts,bits);
    std::printf("conversation_snapshot_test: %d checks passed\n",checks);
}
