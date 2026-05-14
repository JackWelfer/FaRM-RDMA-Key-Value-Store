#pragma once

#include <cstddef>      // std::size_t
#include <cstdint>      // fixed-width ints (optional)
#include <algorithm>    // std::min
#include <chrono>
#include <functional>   // std::hash
#include <stdexcept>    // std::runtime_error / std::out_of_range
#include <thread>
#include <vector>

#include <remus/remus.h>


constexpr int SLOTS_PER_BUCKET = 16;
constexpr int SLOTS_PER_BLOCK = 2;

template <typename Key, typename Value>
class hopscotch_hashtable {

    struct KVSlot {
        Key key;
        Value value;
        bool occupied;
    };
    
    struct OverflowBlock {
        KVSlot slots[SLOTS_PER_BLOCK];
        remus::rdma_ptr<OverflowBlock> next;
    };
    
    struct Bucket {
        uint64_t version;
        KVSlot slots[SLOTS_PER_BUCKET];
        remus::rdma_ptr<OverflowBlock> overflow_head;
    };

    /// `N` contiguous remote `Bucket`s in one `Read<NeighborhoodBatch<N>>` (Remus uses
    /// registered staging; sizeof matches a remote slice of `N * sizeof(Bucket)`).
    template <int N>
    struct NeighborhoodBatch {
        Bucket buckets[N];
    };

    struct HashTableNode {
        remus::rdma_ptr<Bucket> bucket_array;
        int node_id;
    };
    struct HashTableRoot
    {
        remus::rdma_ptr<HashTableNode> node_array;
        int num_nodes;
        uint64_t num_buckets; 
        int H;
    };
    struct BucketLocation {
        uint64_t global_bucket_idx;
        int node_id;
        uint64_t local_bucket_idx;
    };
    public:
        void Init(remus::ComputeThread *ct, int num_nodes, int node_id, int H, uint64_t num_buckets)
        {
            if (node_id == 0)
            {
                auto root = CreateRoot(ct, num_nodes, H, num_buckets);
                ct->set_root(root);
            }
            ct->arrive_control_barrier(num_nodes);
            root = ct->get_root<HashTableRoot>();
            InitLocal(ct, node_id, root);
            ct->arrive_control_barrier(num_nodes);
            CacheBucketArraysForAllNodes(ct);
        }

    private:
        remus::rdma_ptr<HashTableRoot> CreateRoot(remus::ComputeThread *ct, int num_nodes, int H, uint64_t num_buckets)
        {
            REMUS_ASSERT(num_nodes > 0, "CreateRoot: num_nodes must be positive, got {}", num_nodes);
            REMUS_ASSERT(num_buckets % static_cast<uint64_t>(num_nodes) == 0,
                         "CreateRoot: num_buckets ({}) must be divisible by num_nodes ({})",
                         num_buckets, num_nodes);
            auto root = ct->allocate<HashTableRoot>();
            auto node_array = ct->allocate<HashTableNode>(num_nodes);
            ct->Write<uint64_t>(
                remus::rdma_ptr<uint64_t>(root.raw() + offsetof(HashTableRoot, node_array)),
                node_array.raw());
            ct->Write<int>(remus::rdma_ptr<int>(root.raw() + offsetof(HashTableRoot, num_nodes)), num_nodes);
            ct->Write<uint64_t>(remus::rdma_ptr<uint64_t>(root.raw() + offsetof(HashTableRoot, num_buckets)), num_buckets);
            ct->Write<int>(remus::rdma_ptr<int>(root.raw() + offsetof(HashTableRoot, H)), H);
            return root;
        }
        void InitLocal(remus::ComputeThread *ct, int node_id, remus::rdma_ptr<HashTableRoot> root) 
        {
            //cache the root
            this->root = root;

            const int num_nodes =
                ct->Read<int>(remus::rdma_ptr<int>(root.raw() + offsetof(HashTableRoot, num_nodes)));
            REMUS_ASSERT(num_nodes > 0, "InitLocal: num_nodes must be positive, got {}", num_nodes);
            REMUS_ASSERT(node_id >= 0 && node_id < num_nodes,
                         "InitLocal: node_id ({}) out of range for num_nodes ({})", node_id, num_nodes);

            const uint64_t num_buckets =
                ct->Read<uint64_t>(remus::rdma_ptr<uint64_t>(root.raw() + offsetof(HashTableRoot, num_buckets)));
            REMUS_ASSERT(num_buckets % static_cast<uint64_t>(num_nodes) == 0,
                         "InitLocal: num_buckets ({}) must be divisible by num_nodes ({})",
                         num_buckets, num_nodes);
            const int H =
                ct->Read<int>(remus::rdma_ptr<int>(root.raw() + offsetof(HashTableRoot, H)));
            REMUS_ASSERT(H > 0, "InitLocal: H must be positive, got {}", H);

            // Cache frequently used metadata for fast bucket location.
            this->num_nodes_ = num_nodes;
            this->num_buckets_ = num_buckets;
            this->H_ = H;
            this->num_bucket_per_node_ =
                num_buckets / static_cast<uint64_t>(num_nodes);

            const uint64_t num_bucket_per_node = num_bucket_per_node_;
            auto bucket_array = ct->allocate<Bucket>(num_bucket_per_node);
            for (uint64_t i = 0; i < num_bucket_per_node; ++i) {
                ct->Write<uint64_t>(
                    remus::rdma_ptr<uint64_t>(bucket_array.raw() + sizeof(Bucket) * i +
                                              offsetof(Bucket, version)),
                    static_cast<uint64_t>(0));
                for (int j = 0; j < SLOTS_PER_BUCKET; j++) {
                    KVSlot slot = {};
                    slot.occupied = false;
                    
                    ct->Write<KVSlot>(remus::rdma_ptr<KVSlot>(bucket_array.raw() + sizeof(Bucket) * i + offsetof(Bucket, slots) + sizeof(KVSlot) * j), slot);
                }
                const uint64_t zero = 0;
                ct->Write<uint64_t>(
                    remus::rdma_ptr<uint64_t>(bucket_array.raw() + sizeof(Bucket) * i +
                                              offsetof(Bucket, overflow_head)),
                    zero);
            }
         
            remus::rdma_ptr<HashTableNode> node_array(
                ct->Read<uint64_t>(
                    remus::rdma_ptr<uint64_t>(
                        root.raw() + offsetof(HashTableRoot, node_array))));
            this->node_array_ = node_array;
            ct->Write<uint64_t>(
                remus::rdma_ptr<uint64_t>(
                    node_array.raw() + sizeof(HashTableNode) * node_id +
                    offsetof(HashTableNode, bucket_array)),
                bucket_array.raw());
            ct->Write<int>(remus::rdma_ptr<int>(node_array.raw() + sizeof(HashTableNode) * node_id + offsetof(HashTableNode, node_id)), node_id);
        }
   
    
        private:
            remus::rdma_ptr<HashTableRoot> root;
            remus::rdma_ptr<HashTableNode> node_array_;
            std::vector<remus::rdma_ptr<Bucket>> bucket_arrays_by_node_;
            int num_nodes_ = 0;
            uint64_t num_buckets_ = 0;
            uint64_t num_bucket_per_node_ = 0;
            int H_ = 0;
            void CacheBucketArraysForAllNodes(remus::ComputeThread *ct)
            {
                REMUS_ASSERT(node_array_.raw() != 0, "CacheBucketArraysForAllNodes: node_array_ is not initialized");
                bucket_arrays_by_node_.resize(static_cast<size_t>(num_nodes_));
                for (int nid = 0; nid < num_nodes_; ++nid) {
                    bucket_arrays_by_node_[static_cast<size_t>(nid)] =
                        remus::rdma_ptr<Bucket>(
                            ct->Read<uint64_t>(
                                remus::rdma_ptr<uint64_t>(
                                    node_array_.raw() + sizeof(HashTableNode) * nid +
                                    offsetof(HashTableNode, bucket_array))));
                }
            }
            uint64_t HashToGlobalBucket(const Key &key, uint64_t num_buckets) const
            {
                REMUS_ASSERT(num_buckets > 0, "HashToGlobalBucket: num_buckets must be > 0");
                return static_cast<uint64_t>(std::hash<Key>{}(key)) % num_buckets;
            }
            BucketLocation LocateBucket(const Key &key) const
            {
                REMUS_ASSERT(root.raw() != 0, "LocateBucket: root is not initialized");
                REMUS_ASSERT(num_nodes_ > 0, "LocateBucket: cached num_nodes is not initialized");
                REMUS_ASSERT(num_buckets_ > 0, "LocateBucket: cached num_buckets is not initialized");
                REMUS_ASSERT(num_bucket_per_node_ > 0,
                             "LocateBucket: cached num_bucket_per_node is not initialized");
                             
                const uint64_t global_bucket_idx = HashToGlobalBucket(key, num_buckets_);
                BucketLocation loc;
                loc.global_bucket_idx = global_bucket_idx;
                loc.node_id = static_cast<int>(global_bucket_idx / num_bucket_per_node_);
                loc.local_bucket_idx = global_bucket_idx % num_bucket_per_node_;
                return loc;
            }
            BucketLocation LocateBucketByGlobalIndex(uint64_t global_bucket_idx) const
            {
                REMUS_ASSERT(num_bucket_per_node_ > 0,
                             "LocateBucketByGlobalIndex: num_bucket_per_node_ not initialized");
                BucketLocation loc;
                loc.global_bucket_idx = global_bucket_idx;
                loc.node_id = static_cast<int>(global_bucket_idx / num_bucket_per_node_);
                loc.local_bucket_idx = global_bucket_idx % num_bucket_per_node_;
                return loc;
            }
            remus::rdma_ptr<Bucket> ReadBucketArrayPtrForNode(int node_id) const
            {
                REMUS_ASSERT(node_id >= 0 && node_id < num_nodes_,
                             "ReadBucketArrayPtrForNode: node_id {} out of range [0, {})",
                             node_id, num_nodes_);
                REMUS_ASSERT(bucket_arrays_by_node_.size() == static_cast<size_t>(num_nodes_),
                             "ReadBucketArrayPtrForNode: cached bucket map is not initialized");
                return bucket_arrays_by_node_[static_cast<size_t>(node_id)];
            }
            remus::rdma_ptr<Bucket> GetBucketPtr(const BucketLocation &loc) const
            {
                auto bucket_array = ReadBucketArrayPtrForNode(loc.node_id);
                return remus::rdma_ptr<Bucket>(bucket_array.raw() +
                                               sizeof(Bucket) * loc.local_bucket_idx);
            }
            remus::rdma_ptr<KVSlot> GetSlotPtr(const remus::rdma_ptr<Bucket> &bucket_ptr,
                                               int slot_idx) const
            {
                return remus::rdma_ptr<KVSlot>(bucket_ptr.raw() + offsetof(Bucket, slots) +
                                               sizeof(KVSlot) * slot_idx);
            }
            remus::rdma_ptr<uint64_t> GetBucketVersionPtr(
                const remus::rdma_ptr<Bucket> &bucket_ptr) const
            {
                return remus::rdma_ptr<uint64_t>(bucket_ptr.raw() + offsetof(Bucket, version));
            }
            remus::rdma_ptr<remus::rdma_ptr<OverflowBlock>>
            GetOverflowHeadFieldPtr(const remus::rdma_ptr<Bucket> &bucket_ptr) const
            {
                return remus::rdma_ptr<remus::rdma_ptr<OverflowBlock>>(
                    bucket_ptr.raw() + offsetof(Bucket, overflow_head));
            }
            remus::rdma_ptr<remus::rdma_ptr<OverflowBlock>>
            GetOverflowNextFieldPtr(const remus::rdma_ptr<OverflowBlock> &block_ptr) const
            {
                return remus::rdma_ptr<remus::rdma_ptr<OverflowBlock>>(
                    block_ptr.raw() + offsetof(OverflowBlock, next));
            }
            remus::rdma_ptr<KVSlot> GetOverflowSlotPtr(
                const remus::rdma_ptr<OverflowBlock> &block_ptr, int slot_idx) const
            {
                return remus::rdma_ptr<KVSlot>(block_ptr.raw() +
                                               offsetof(OverflowBlock, slots) +
                                               sizeof(KVSlot) * slot_idx);
            }
            void InitializeOverflowBlock(remus::ComputeThread *ct,
                                        const remus::rdma_ptr<OverflowBlock> &block_ptr) const
            {
                KVSlot empty_slot{};
                empty_slot.occupied = false;
                for (int i = 0; i < SLOTS_PER_BLOCK; ++i) {
                    ct->Write<KVSlot>(GetOverflowSlotPtr(block_ptr, i), empty_slot);
                }
                const uint64_t zero = 0;
                ct->Write<uint64_t>(
                    remus::rdma_ptr<uint64_t>(GetOverflowNextFieldPtr(block_ptr).raw()),
                    zero);
            }
            // Per-op retry loops abort after this many backoffs (true infinite spin
            // usually means a bucket version stuck odd, e.g. lost unlock).
            static constexpr uint32_t kMaxSpinRetries = 500000U;

            [[noreturn]] void FatalSpinExceeded_(const char *op, const Key &key,
                                                 uint32_t retries) const {
                REMUS_FATAL(
                    "{}: exceeded spin limit ({}) for key={} — likely stuck odd "
                    "version (crash mid-lock) or extreme livelock",
                    op, retries, key);
            }

            void CheckSpinLimit_(const char *op, const Key &key, uint32_t retries) const {
                if (retries > kMaxSpinRetries) {
                    FatalSpinExceeded_(op, key, retries);
                }
            }

            bool TryLockBucket(remus::ComputeThread *ct,
                               const remus::rdma_ptr<Bucket> &bucket_ptr,
                               uint64_t *locked_even_version) const
            {
                auto version_ptr = GetBucketVersionPtr(bucket_ptr);
                // Read only the version word: pairs with CAS on the same field and
                // avoids a full-bucket RDMA read before every lock attempt.
                const uint64_t observed = ct->Read<uint64_t>(version_ptr);
                if ((observed & 1ULL) != 0) {
                    return false;
                }
                const uint64_t prior =
                    ct->CompareAndSwap<uint64_t>(version_ptr, observed, observed + 1);
                if (prior == observed) {
                    *locked_even_version = observed;
                    // REMUS_INFO(
                    //     "TryLockBucket got_lock tid={} bucket=0x{:x} version={} "
                    //     "(memory_version_word={})",
                    //     ct->get_tid(), bucket_ptr.raw(), observed, observed + 1);
                    return true;
                }
                return false;
            }
            void UnlockBucket(remus::ComputeThread *ct,
                              const remus::rdma_ptr<Bucket> &bucket_ptr,
                              uint64_t locked_even_version) const
            {
                auto version_ptr = GetBucketVersionPtr(bucket_ptr);
                const uint64_t expect_odd = locked_even_version + 1;
                const uint64_t swap_even = locked_even_version + 2;
                (void)expect_odd;
                // CAS unlock (use when Write local-fast-path breaks RDMA Read visibility):
                // const uint64_t cas_prior =
                //     ct->CompareAndSwap<uint64_t>(version_ptr, expect_odd, swap_even);
                // REMUS_INFO(
                //     "UnlockBucket tid={} bucket=0x{:x} locked_at_even_version={} "
                //     "unlock_cas_expected_odd={} unlock_cas_swap_even={} cas_return_prior={} "
                //     "(on success version is unlock_cas_swap_even; cas_return_prior should "
                //     "equal unlock_cas_expected_odd)",
                //     ct->get_tid(), bucket_ptr.raw(), locked_even_version, expect_odd,
                //     swap_even, cas_prior);
                ct->Write<uint64_t>(version_ptr, swap_even, true, sizeof(uint64_t), false);
                // const bool is_local_addr = ct->is_local(version_ptr);
                // const uint64_t cpu_read_after_write =
                //     is_local_addr
                //         ? *reinterpret_cast<volatile uint64_t *>(
                //               version_ptr.address())
                //         : 0ULL;
                // const uint64_t rdma_read_after_write =
                //     ct->Read<uint64_t>(version_ptr);
                // REMUS_INFO(
                //     "UnlockBucket tid={} bucket=0x{:x} locked_at_even_version={} "
                //     "intent_swap_even={} is_local={} cpu_read_after_write={} "
                //     "rdma_read_after_write={}",
                //     ct->get_tid(), bucket_ptr.raw(), locked_even_version, swap_even,
                //     is_local_addr, cpu_read_after_write, rdma_read_after_write);
            }
            uint64_t ReadHomeBucketVersion_(remus::ComputeThread *ct,
                                            const BucketLocation &home_loc) const
            {
                return ct->Read<uint64_t>(GetBucketVersionPtr(GetBucketPtr(home_loc)));
            }
            void Backoff(uint32_t retry_count) const
            {
                // Testing: old backoff (yield + capped exponential ns + jitter).
                if (retry_count < 8) {
                    std::this_thread::yield();
                    return;
                }
                const uint32_t capped = std::min<uint32_t>(retry_count, 22);
                uint64_t pause_ns = (1ULL << (capped - 8)) * 200ULL;
                pause_ns +=
                    static_cast<uint64_t>((retry_count * 2654435761u) >> 20) & 4095ULL;
                pause_ns = std::min<uint64_t>(pause_ns, 5'000'000ULL);
                std::this_thread::sleep_for(std::chrono::nanoseconds(pause_ns));

                // Previous policy: uniform random 1–5 s (re-enable for livelock experiments).
                // thread_local std::mt19937 rng = [] {
                //     std::random_device rd;
                //     return std::mt19937(rd());
                // }();
                // std::uniform_int_distribution<int> ms(1000, 5000);
                // std::this_thread::sleep_for(std::chrono::milliseconds(ms(rng)));
            }

            /// Steps forward on the bucket ring from `from_global` to `to_global`.
            uint64_t ForwardBucketDistance_(uint64_t from_global, uint64_t to_global) const
            {
                return (to_global + num_buckets_ - from_global) % num_buckets_;
            }

            /// True iff `bucket_global` is one of the H_ buckets starting at `home_global`
            /// (same rule as neighborhood lookup).
            bool BucketInNeighborhood_(uint64_t home_global, uint64_t bucket_global) const
            {
                return ForwardBucketDistance_(home_global, bucket_global) <
                       static_cast<uint64_t>(H_);
            }

            /// Move one occupied slot to an empty slot under per-bucket version locks.
            ///
            /// Flow:
            /// 1. Same bucket: lock once, re-read both slots, verify source occupied and dest
            ///    empty, copy source → dest, clear source, unlock.
            /// 2. Two buckets: lock in ascending `global_bucket_index` (always the same order
            ///    regardless of source/dest roles) so concurrent moves cannot deadlock; then same
            ///    verify → copy → clear; unlock higher index first, then lower.
            ///
            /// False return: could not lock, or snapshot did not match hopscotch move preconditions
            /// (caller retries the bubble).
            bool TryMoveSlotBetweenBuckets_(remus::ComputeThread *ct, uint64_t source_bucket_global,
                                           int source_slot_index, uint64_t destination_bucket_global,
                                           int destination_slot_index) const
            {
                const auto source_bucket_pointer =
                    GetBucketPtr(LocateBucketByGlobalIndex(source_bucket_global));
                const auto destination_bucket_pointer =
                    GetBucketPtr(LocateBucketByGlobalIndex(destination_bucket_global));

                // --- Path A: both indices refer to the same bucket (different slots).
                if (source_bucket_global == destination_bucket_global) {
                    uint64_t locked_even_version = 0;
                    if (!TryLockBucket(ct, source_bucket_pointer, &locked_even_version)) {
                        return false;
                    }
                    const KVSlot source_slot =
                        ct->Read<KVSlot>(GetSlotPtr(source_bucket_pointer, source_slot_index));
                    const KVSlot destination_slot =
                        ct->Read<KVSlot>(GetSlotPtr(source_bucket_pointer, destination_slot_index));
                    if (!source_slot.occupied || destination_slot.occupied) {
                        UnlockBucket(ct, source_bucket_pointer, locked_even_version);
                        return false;
                    }
                    ct->Write<KVSlot>(GetSlotPtr(source_bucket_pointer, destination_slot_index),
                                      source_slot);
                    KVSlot cleared{};
                    cleared.occupied = false;
                    ct->Write<KVSlot>(GetSlotPtr(source_bucket_pointer, source_slot_index), cleared);
                    UnlockBucket(ct, source_bucket_pointer, locked_even_version);
                    return true;
                }

                // --- Path B: two buckets — establish a global lock order (min index first).
                remus::rdma_ptr<Bucket> first_locked_bucket_pointer;
                remus::rdma_ptr<Bucket> second_locked_bucket_pointer;
                if (source_bucket_global < destination_bucket_global) {
                    first_locked_bucket_pointer = source_bucket_pointer;
                    second_locked_bucket_pointer = destination_bucket_pointer;
                } else {
                    first_locked_bucket_pointer = destination_bucket_pointer;
                    second_locked_bucket_pointer = source_bucket_pointer;
                }

                uint64_t first_locked_even_version = 0;
                uint64_t second_locked_even_version = 0;
                if (!TryLockBucket(ct, first_locked_bucket_pointer, &first_locked_even_version)) {
                    return false;
                }
                if (!TryLockBucket(ct, second_locked_bucket_pointer, &second_locked_even_version)) {
                    UnlockBucket(ct, first_locked_bucket_pointer, first_locked_even_version);
                    return false;
                }

                // Under both locks: confirm move still valid, then write dest then clear source.
                const KVSlot source_slot =
                    ct->Read<KVSlot>(GetSlotPtr(source_bucket_pointer, source_slot_index));
                const KVSlot destination_slot =
                    ct->Read<KVSlot>(GetSlotPtr(destination_bucket_pointer, destination_slot_index));
                if (!source_slot.occupied || destination_slot.occupied) {
                    UnlockBucket(ct, second_locked_bucket_pointer, second_locked_even_version);
                    UnlockBucket(ct, first_locked_bucket_pointer, first_locked_even_version);
                    return false;
                }

                ct->Write<KVSlot>(GetSlotPtr(destination_bucket_pointer, destination_slot_index),
                                  source_slot);
                KVSlot cleared{};
                cleared.occupied = false;
                ct->Write<KVSlot>(GetSlotPtr(source_bucket_pointer, source_slot_index), cleared);

                UnlockBucket(ct, second_locked_bucket_pointer, second_locked_even_version);
                UnlockBucket(ct, first_locked_bucket_pointer, first_locked_even_version);
                return true;
            }

            enum class BubbleInsertOutcome {
                Inserted,
                NeedOverflow,
                Retry,
            };

            /// Hopscotch bubble insert: slide a free slot into `home`’s H_-neighborhood by
            /// repeatedly moving one entry one step “backward” along the ring (toward lower
            /// indices mod N), then write `(key,value)` into the hole once it lands in-range.
            ///
            /// Flow:
            /// 1. Find *some* empty `(bucket,slot)` anywhere (linear probe from `home`; wide reads).
            /// 2. Loop: if that hole is already in `insert_home`’s neighborhood → lock bucket,
            ///    re-verify slot empty, write new entry, done.
            /// 3. Else “bubble”: for d = 1 .. H_-1, look at bucket `empty - d` (mod N). Scan its
            ///    slots (optimistic bucket read). For each occupied entry, if that key’s *home*
            ///    neighborhood contains the current hole, the entry may legally move into the hole
            ///    (`TryMoveSlotBetweenBuckets_` validates under locks). One successful move shifts
            ///    the hole backward to where the entry came from; repeat the outer loop.
            /// 4. No candidate at any d, or step cap exceeded → NeedOverflow. Lock / probe
            ///    contention → Retry (caller backs off).
            BubbleInsertOutcome HopscotchBubbleInsert_(remus::ComputeThread *ct, const Key &key,
                                                       const Value &value,
                                                       const BucketLocation &home) const
            {
                const uint64_t insert_home_global = home.global_bucket_idx;
                uint64_t empty_bucket_global = 0;
                int empty_slot_index = 0;
                bool empty_probe_retry = false;
                // Any table-wide empty is enough; we will drag it toward `insert_home` by moves.
                if (!FindFirstEmptySlotLinearProbe_(ct, insert_home_global, &empty_bucket_global,
                                                    &empty_slot_index, &empty_probe_retry)) {
                    if (empty_probe_retry) {
                        return BubbleInsertOutcome::Retry;
                    }
                    return BubbleInsertOutcome::NeedOverflow;
                }

                const uint32_t bubble_step_limit =
                    static_cast<uint32_t>(num_buckets_) * static_cast<uint32_t>(SLOTS_PER_BUCKET) +
                    8U;
                uint32_t bubble_steps = 0;

                while (true) {
                    // Hole reached the insert key’s neighborhood: lock and insert (same as Put).
                    if (BucketInNeighborhood_(insert_home_global, empty_bucket_global)) {
                        const auto insert_bucket_pointer =
                            GetBucketPtr(LocateBucketByGlobalIndex(empty_bucket_global));
                        uint64_t locked_even_version = 0;
                        if (!TryLockBucket(ct, insert_bucket_pointer, &locked_even_version)) {
                            return BubbleInsertOutcome::Retry;
                        }
                        const KVSlot slot_check =
                            ct->Read<KVSlot>(GetSlotPtr(insert_bucket_pointer, empty_slot_index));
                        if (slot_check.occupied) {
                            UnlockBucket(ct, insert_bucket_pointer, locked_even_version);
                            return BubbleInsertOutcome::Retry;
                        }
                        KVSlot new_entry{};
                        new_entry.occupied = true;
                        new_entry.key = key;
                        new_entry.value = value;
                        ct->Write<KVSlot>(GetSlotPtr(insert_bucket_pointer, empty_slot_index),
                                          new_entry);
                        UnlockBucket(ct, insert_bucket_pointer, locked_even_version);
                        return BubbleInsertOutcome::Inserted;
                    }

                    if (++bubble_steps > bubble_step_limit) {
                        return BubbleInsertOutcome::NeedOverflow;
                    }

                    // Pull the hole backward: find an entry 1..H_-1 steps behind that is allowed
                    // to occupy the current hole (its home’s neighborhood contains the hole).
                    bool moved_one_entry = false;
                    for (int backward_distance = 1; backward_distance < H_; ++backward_distance) {
                        const uint64_t candidate_bucket_global =
                            (empty_bucket_global + num_buckets_ -
                             static_cast<uint64_t>(backward_distance)) %
                            num_buckets_;
                        const auto candidate_bucket_pointer =
                            GetBucketPtr(LocateBucketByGlobalIndex(candidate_bucket_global));
                        const Bucket candidate_snapshot = ct->Read<Bucket>(candidate_bucket_pointer);

                        for (int candidate_slot_index = 0; candidate_slot_index < SLOTS_PER_BUCKET;
                             ++candidate_slot_index) {
                            if (!candidate_snapshot.slots[candidate_slot_index].occupied) {
                                continue;
                            }
                            const Key &candidate_key =
                                candidate_snapshot.slots[candidate_slot_index].key;
                            const uint64_t candidate_entry_home_global =
                                HashToGlobalBucket(candidate_key, num_buckets_);
                            if (!BucketInNeighborhood_(candidate_entry_home_global,
                                                       empty_bucket_global)) {
                                continue;
                            }
                            if (!TryMoveSlotBetweenBuckets_(ct, candidate_bucket_global,
                                                            candidate_slot_index,
                                                            empty_bucket_global,
                                                            empty_slot_index)) {
                                return BubbleInsertOutcome::Retry;
                            }
                            // Hole is now where that entry used to sit; continue outer loop.
                            empty_bucket_global = candidate_bucket_global;
                            empty_slot_index = candidate_slot_index;
                            moved_one_entry = true;
                            break;
                        }
                        if (moved_one_entry) {
                            break;
                        }
                    }

                    // No legal backward hop at any distance: table too tight for this H_ / layout.
                    if (!moved_one_entry) {
                        return BubbleInsertOutcome::NeedOverflow;
                    }
                }
            }

            /// Copies `bucket_count_in_segment` contiguous remote buckets starting at
            /// `segment_base`. Uses one `Read<NeighborhoodBatch<N>>` when the count matches
            /// `N` for `1 <= N <= 32` (wide read into Remus staging); otherwise per-bucket
            /// `Read<Bucket>`. Extend the switch if you need larger neighborhoods.
            static void ReadSegmentBuckets_(remus::ComputeThread *ct,
                                            remus::rdma_ptr<Bucket> segment_base,
                                            size_t bucket_count_in_segment,
                                            Bucket *destination_buckets)
            {
                switch (static_cast<int>(bucket_count_in_segment)) {
#define HOP_NB_(n)                                                                               \
    case n: {                                                                                    \
        const remus::rdma_ptr<NeighborhoodBatch<n>> batch_ptr(segment_base.raw());              \
        const NeighborhoodBatch<n> pack = ct->Read<NeighborhoodBatch<n>>(batch_ptr);             \
        for (int bucket_index_in_batch = 0; bucket_index_in_batch < n; ++bucket_index_in_batch) { \
            destination_buckets[bucket_index_in_batch] = pack.buckets[bucket_index_in_batch];   \
        }                                                                                        \
    } break
                    HOP_NB_(1);
                    HOP_NB_(2);
                    HOP_NB_(3);
                    HOP_NB_(4);
                    HOP_NB_(5);
                    HOP_NB_(6);
                    HOP_NB_(7);
                    HOP_NB_(8);
                    HOP_NB_(9);
                    HOP_NB_(10);
                    HOP_NB_(11);
                    HOP_NB_(12);
                    HOP_NB_(13);
                    HOP_NB_(14);
                    HOP_NB_(15);
                    HOP_NB_(16);
                    HOP_NB_(17);
                    HOP_NB_(18);
                    HOP_NB_(19);
                    HOP_NB_(20);
                    HOP_NB_(21);
                    HOP_NB_(22);
                    HOP_NB_(23);
                    HOP_NB_(24);
                    HOP_NB_(25);
                    HOP_NB_(26);
                    HOP_NB_(27);
                    HOP_NB_(28);
                    HOP_NB_(29);
                    HOP_NB_(30);
                    HOP_NB_(31);
                    HOP_NB_(32);
#undef HOP_NB_
                default:
                    for (size_t bucket_index_in_segment = 0;
                         bucket_index_in_segment < bucket_count_in_segment;
                         ++bucket_index_in_segment) {
                        destination_buckets[bucket_index_in_segment] = ct->Read<Bucket>(
                            remus::rdma_ptr<Bucket>(
                                segment_base.raw() + bucket_index_in_segment * sizeof(Bucket)));
                    }
                    break;
                }
            }

            /// Contiguous remote runs for one logical H_-bucket neighborhood + one RDMA
            /// snapshot per run (same layout as lookup).
            struct NeighborhoodRdSnapshot_ {
                struct ContiguousSegment {
                    remus::rdma_ptr<Bucket> segment_base;
                    size_t bucket_count_in_segment;
                };
                std::vector<ContiguousSegment> contiguous_memory_segments;
                std::vector<size_t> segment_index_for_neighborhood_step;
                std::vector<size_t> bucket_index_in_segment_for_neighborhood_step;
                std::vector<std::vector<Bucket>> buckets_per_segment;
            };

            /// Builds segment layout and fetches each segment once (`ReadSegmentBuckets_`).
            /// Returns false only if H_ == 0.
            bool LoadNeighborhoodRdSnapshot_(remus::ComputeThread *ct, const BucketLocation &home_loc,
                                               NeighborhoodRdSnapshot_ *out) const
            {
                const uint64_t neighborhood_bucket_count = static_cast<uint64_t>(H_);
                const uint64_t home_global_bucket_index = home_loc.global_bucket_idx;
                out->contiguous_memory_segments.clear();
                out->segment_index_for_neighborhood_step.clear();
                out->bucket_index_in_segment_for_neighborhood_step.clear();
                out->buckets_per_segment.clear();
                if (neighborhood_bucket_count == 0ULL) {
                    return false;
                }
                out->segment_index_for_neighborhood_step.resize(
                    static_cast<size_t>(neighborhood_bucket_count));
                out->bucket_index_in_segment_for_neighborhood_step.resize(
                    static_cast<size_t>(neighborhood_bucket_count));

                uint64_t neighborhood_step = 0;
                while (neighborhood_step < neighborhood_bucket_count) {
                    const uint64_t global_bucket_index =
                        (home_global_bucket_index + neighborhood_step) % num_buckets_;
                    const auto segment_base =
                        GetBucketPtr(LocateBucketByGlobalIndex(global_bucket_index));
                    size_t buckets_in_this_segment = 1;
                    while (neighborhood_step + buckets_in_this_segment < neighborhood_bucket_count) {
                        const uint64_t next_global_bucket_index =
                            (home_global_bucket_index + neighborhood_step + buckets_in_this_segment) %
                            num_buckets_;
                        const auto next_bucket_pointer =
                            GetBucketPtr(LocateBucketByGlobalIndex(next_global_bucket_index));
                        const auto expected_next_bucket_pointer = remus::rdma_ptr<Bucket>(
                            segment_base.raw() + buckets_in_this_segment * sizeof(Bucket));
                        if (next_bucket_pointer.raw() != expected_next_bucket_pointer.raw()) {
                            break;
                        }
                        ++buckets_in_this_segment;
                    }
                    const size_t new_segment_index = out->contiguous_memory_segments.size();
                    typename NeighborhoodRdSnapshot_::ContiguousSegment contiguous_segment{};
                    contiguous_segment.segment_base = segment_base;
                    contiguous_segment.bucket_count_in_segment = buckets_in_this_segment;
                    out->contiguous_memory_segments.push_back(contiguous_segment);
                    for (size_t step_within_segment = 0; step_within_segment < buckets_in_this_segment;
                         ++step_within_segment) {
                        out->segment_index_for_neighborhood_step[static_cast<size_t>(
                            neighborhood_step + step_within_segment)] = new_segment_index;
                        out->bucket_index_in_segment_for_neighborhood_step[static_cast<size_t>(
                            neighborhood_step + step_within_segment)] = step_within_segment;
                    }
                    neighborhood_step += static_cast<uint64_t>(buckets_in_this_segment);
                }

                REMUS_ASSERT(
                    out->contiguous_memory_segments.size() <= 2,
                    "LoadNeighborhoodRdSnapshot_: neighborhood spans {} contiguous memory regions "
                    "(max 2). Adjust num_buckets / H / node layout.",
                    out->contiguous_memory_segments.size());

                out->buckets_per_segment.resize(out->contiguous_memory_segments.size());
                for (size_t segment_index = 0; segment_index < out->contiguous_memory_segments.size();
                     ++segment_index) {
                    out->buckets_per_segment[segment_index].resize(
                        out->contiguous_memory_segments[segment_index].bucket_count_in_segment);
                    ReadSegmentBuckets_(
                        ct, out->contiguous_memory_segments[segment_index].segment_base,
                        out->contiguous_memory_segments[segment_index].bucket_count_in_segment,
                        out->buckets_per_segment[segment_index].data());
                }
                return true;
            }

            /// First free slot in `home_loc`'s neighborhood (bucket-major slot order), one
            /// snapshot pass. Odd bucket version → `*out_retry` and false.
            bool TryFindFirstFreeInNeighborhood_(remus::ComputeThread *ct,
                                                   const BucketLocation &home_loc,
                                                   uint64_t *out_bucket_global, int *out_slot_index,
                                                   bool *out_retry) const
            {
                *out_retry = false;
                NeighborhoodRdSnapshot_ snap;
                if (!LoadNeighborhoodRdSnapshot_(ct, home_loc, &snap)) {
                    return false;
                }
                const uint64_t home_global_bucket_index = home_loc.global_bucket_idx;
                const uint64_t neighborhood_bucket_count = static_cast<uint64_t>(H_);
                for (uint64_t neighborhood_step = 0; neighborhood_step < neighborhood_bucket_count;
                     ++neighborhood_step) {
                    const size_t segment_index =
                        snap.segment_index_for_neighborhood_step[static_cast<size_t>(
                            neighborhood_step)];
                    const size_t bucket_index_in_segment =
                        snap.bucket_index_in_segment_for_neighborhood_step[static_cast<size_t>(
                            neighborhood_step)];
                    const Bucket &bucket_from_snapshot =
                        snap.buckets_per_segment[segment_index][bucket_index_in_segment];
                    if ((bucket_from_snapshot.version & 1ULL) != 0) {
                        *out_retry = true;
                        return false;
                    }
                    const uint64_t global_bucket_index =
                        (home_global_bucket_index + neighborhood_step) % num_buckets_;
                    for (int slot_index_in_bucket = 0; slot_index_in_bucket < SLOTS_PER_BUCKET;
                         ++slot_index_in_bucket) {
                        if (!bucket_from_snapshot.slots[slot_index_in_bucket].occupied) {
                            *out_bucket_global = global_bucket_index;
                            *out_slot_index = slot_index_in_bucket;
                            return true;
                        }
                    }
                }
                return false;
            }

            /// First empty slot in bucket-major order starting at `start_home_global`, using
            /// wide neighborhood reads when `H_ < num_buckets_` (same chunk order as a linear
            /// walk). If `H_ >= num_buckets_`, falls back to one `Read<Bucket>` per bucket.
            bool FindFirstEmptySlotLinearProbe_(remus::ComputeThread *ct, uint64_t start_home_global,
                                                uint64_t *out_bucket_global, int *out_slot_index,
                                                bool *out_retry) const
            {
                *out_retry = false;
                if (static_cast<uint64_t>(H_) >= num_buckets_) {
                    for (uint64_t bucket_offset = 0; bucket_offset < num_buckets_; ++bucket_offset) {
                        const uint64_t global_bucket_index =
                            (start_home_global + bucket_offset) % num_buckets_;
                        const auto bucket_pointer =
                            GetBucketPtr(LocateBucketByGlobalIndex(global_bucket_index));
                        const Bucket bucket_snapshot = ct->Read<Bucket>(bucket_pointer);
                        if ((bucket_snapshot.version & 1ULL) != 0) {
                            *out_retry = true;
                            return false;
                        }
                        for (int slot_index = 0; slot_index < SLOTS_PER_BUCKET; ++slot_index) {
                            if (!bucket_snapshot.slots[slot_index].occupied) {
                                *out_bucket_global = global_bucket_index;
                                *out_slot_index = slot_index;
                                return true;
                            }
                        }
                    }
                    return false;
                }
                for (uint64_t chunk_index = 0;
                     chunk_index * static_cast<uint64_t>(H_) < num_buckets_;
                     ++chunk_index) {
                    const uint64_t neighborhood_home_global =
                        (start_home_global + chunk_index * static_cast<uint64_t>(H_)) % num_buckets_;
                    bool chunk_retry = false;
                    if (TryFindFirstFreeInNeighborhood_(
                            ct, LocateBucketByGlobalIndex(neighborhood_home_global), out_bucket_global,
                            out_slot_index, &chunk_retry)) {
                        return true;
                    }
                    if (chunk_retry) {
                        *out_retry = true;
                        return false;
                    }
                }
                return false;
            }

            /// Loads the H_-bucket neighborhood by contiguous memory runs (1–2 segments).
            /// Each segment is fetched with `ReadSegmentBuckets_` (one wide `Read` when
            /// segment length matches a `NeighborhoodBatch<N>` case, else per-bucket reads).
            bool TryFindInNeighborhood(remus::ComputeThread *ct,
                                       const Key &key, const BucketLocation &home_loc,
                                       Value *out_value, remus::rdma_ptr<KVSlot> *out_slot_ptr,
                                       KVSlot *out_slot,
                                       remus::rdma_ptr<KVSlot> *out_first_free_slot_ptr,
                                       remus::rdma_ptr<Bucket> *out_found_bucket_ptr,
                                       remus::rdma_ptr<Bucket> *out_first_free_bucket_ptr,
                                       bool *out_retry) const
            {
                *out_retry = false;
                const uint64_t neighborhood_bucket_count = static_cast<uint64_t>(H_);
                const uint64_t home_global_bucket_index = home_loc.global_bucket_idx;
                if (neighborhood_bucket_count == 0ULL) {
                    return false;
                }

                NeighborhoodRdSnapshot_ snap;
                if (!LoadNeighborhoodRdSnapshot_(ct, home_loc, &snap)) {
                    return false;
                }

                // --- Scan neighborhood in logical order using only the first snapshot (no
                // RDMA in the inner slot loop until we find a matching key below).
                bool free_recorded = false;
                for (uint64_t neighborhood_step = 0; neighborhood_step < neighborhood_bucket_count;
                     ++neighborhood_step) {
                    const size_t segment_index =
                        snap.segment_index_for_neighborhood_step[static_cast<size_t>(
                            neighborhood_step)];
                    const size_t bucket_index_in_segment =
                        snap.bucket_index_in_segment_for_neighborhood_step[static_cast<size_t>(
                            neighborhood_step)];
                    const Bucket &bucket_from_first_snapshot =
                        snap.buckets_per_segment[segment_index][bucket_index_in_segment];
                    if ((bucket_from_first_snapshot.version & 1ULL) != 0) {
                        // REMUS_INFO(
                        //     "TryFindNeighborhood initial_read_locked tid={} key={} "
                        //     "home_g={} off={} failed_version={}",
                        //     ct->get_tid(), key, home_global_bucket_index, neighborhood_step,
                        //     bucket_from_first_snapshot.version);
                        *out_retry = true;
                        return false;
                    }
                    const uint64_t global_bucket_index =
                        (home_global_bucket_index + neighborhood_step) % num_buckets_;
                    auto bucket_location = LocateBucketByGlobalIndex(global_bucket_index);
                    auto bucket_pointer = GetBucketPtr(bucket_location);
                    for (int slot_index_in_bucket = 0; slot_index_in_bucket < SLOTS_PER_BUCKET;
                         ++slot_index_in_bucket) {
                        const KVSlot &slot = bucket_from_first_snapshot.slots[slot_index_in_bucket];
                        if (!slot.occupied && out_first_free_slot_ptr != nullptr &&
                            !free_recorded) {
                            *out_first_free_slot_ptr =
                                GetSlotPtr(bucket_pointer, slot_index_in_bucket);
                            if (out_first_free_bucket_ptr != nullptr) {
                                *out_first_free_bucket_ptr = bucket_pointer;
                            }
                            free_recorded = true;
                        }
                        if (slot.occupied && slot.key == key) {
                            if (out_value != nullptr) {
                                *out_value = slot.value;
                            }
                            if (out_slot_ptr != nullptr) {
                                *out_slot_ptr = GetSlotPtr(bucket_pointer, slot_index_in_bucket);
                            }
                            if (out_slot != nullptr) {
                                *out_slot = slot;
                            }
                            if (out_found_bucket_ptr != nullptr) {
                                *out_found_bucket_ptr = bucket_pointer;
                            }
                            // Re-read this bucket alone: first snapshot may be stale if a
                            // writer locked/changed the bucket during our read.
                            const Bucket bucket_after_verify_read =
                                ct->Read<Bucket>(bucket_pointer);
                            if (bucket_from_first_snapshot.version != bucket_after_verify_read.version ||
                                (bucket_after_verify_read.version & 1ULL) != 0) {
                                *out_retry = true;
                                return false;
                            }
                            return true;
                        }
                    }
                }

                // Key not seen in the single neighborhood snapshot above. No second full
                // segment re-read: that doubled RDMA on every miss (bad vs a ~1 read/lookup
                // budget). Callers (Get/Put retries, lock + re-read before mutate) handle races.
                return false;
            }
            bool TryFindInOverflow(remus::ComputeThread *ct,
                                   const Key &key, const BucketLocation &home_loc,
                                   Value *out_value, remus::rdma_ptr<KVSlot> *out_slot_ptr,
                                   KVSlot *out_slot, bool *out_retry) const
            {
                auto home_bucket_ptr = GetBucketPtr(home_loc);
                *out_retry = false;
                const Bucket b1 = ct->Read<Bucket>(home_bucket_ptr);
                if ((b1.version & 1ULL) != 0) {
                    *out_retry = true;
                    return false;
                }
                auto head_ptr = b1.overflow_head;
                auto cur = head_ptr;
                while (cur.raw() != 0) {
                    for (int s = 0; s < SLOTS_PER_BLOCK; ++s) {
                        auto slot_ptr = GetOverflowSlotPtr(cur, s);
                        KVSlot slot = ct->Read<KVSlot>(slot_ptr);
                        if (slot.occupied && slot.key == key) {
                            if (out_value != nullptr) {
                                *out_value = slot.value;
                            }
                            if (out_slot_ptr != nullptr) {
                                *out_slot_ptr = slot_ptr;
                            }
                            if (out_slot != nullptr) {
                                *out_slot = slot;
                            }
                            const Bucket b2 = ct->Read<Bucket>(home_bucket_ptr);
                            if (b1.version != b2.version || (b2.version & 1ULL) != 0) {
                                *out_retry = true;
                                return false;
                            }
                            return true;
                        }
                    }
                    cur = remus::rdma_ptr<OverflowBlock>(
                        ct->Read<uint64_t>(
                            remus::rdma_ptr<uint64_t>(GetOverflowNextFieldPtr(cur).raw())));
                }
                const Bucket b2 = ct->Read<Bucket>(home_bucket_ptr);
                if (b1.version != b2.version || (b2.version & 1ULL) != 0) {
                    *out_retry = true;
                    return false;
                }
                return false;
            }
            bool AppendToOverflow(remus::ComputeThread *ct,
                                  const Key &key, const Value &value,
                                  const BucketLocation &home_loc) const
            {
                auto home_bucket_ptr = GetBucketPtr(home_loc);
                auto head_field_ptr = GetOverflowHeadFieldPtr(home_bucket_ptr);
                auto head = remus::rdma_ptr<OverflowBlock>(
                    ct->Read<uint64_t>(remus::rdma_ptr<uint64_t>(head_field_ptr.raw())));
                auto cur = head;
                remus::rdma_ptr<OverflowBlock> prev{};

                while (cur.raw() != 0) {
                    for (int s = 0; s < SLOTS_PER_BLOCK; ++s) {
                        auto slot_ptr = GetOverflowSlotPtr(cur, s);
                        KVSlot slot = ct->Read<KVSlot>(slot_ptr);
                        if (!slot.occupied) {
                            slot.occupied = true;
                            slot.key = key;
                            slot.value = value;
                            ct->Write<KVSlot>(slot_ptr, slot);
                            return true;
                        }
                    }
                    prev = cur;
                    cur = remus::rdma_ptr<OverflowBlock>(
                        ct->Read<uint64_t>(
                            remus::rdma_ptr<uint64_t>(GetOverflowNextFieldPtr(cur).raw())));
                }

                auto new_block = ct->allocate<OverflowBlock>();
                InitializeOverflowBlock(ct, new_block);
                KVSlot new_slot{};
                new_slot.occupied = true;
                new_slot.key = key;
                new_slot.value = value;
                ct->Write<KVSlot>(GetOverflowSlotPtr(new_block, 0), new_slot);
                if (prev.raw() == 0) {
                    ct->Write<uint64_t>(
                        remus::rdma_ptr<uint64_t>(head_field_ptr.raw()),
                        new_block.raw());
                } else {
                    ct->Write<uint64_t>(
                        remus::rdma_ptr<uint64_t>(GetOverflowNextFieldPtr(prev).raw()),
                        new_block.raw());
                }
                return true;
            }

            /// Insert when key is not present (Put’s “insert path”). Re-runs find/lock
            /// because other nodes may have added the key after Put’s update-phase scans.
            bool PutInsertIfAbsent_(remus::ComputeThread *ct, const Key &key, const Value &value,
                                    const BucketLocation &home) const
            {
                uint32_t ins_retries = 0;
                while (true) {
                    CheckSpinLimit_("Put", key, ins_retries);
                    // const uint64_t home_ver_ins = ReadHomeBucketVersion_(ct, home);
                    // REMUS_INFO(
                    //     "PutInsertIfAbsent tid={} key={} retries={} home_bucket_version={}",
                    //     ct->get_tid(), key, ins_retries, home_ver_ins);
                    // if ((ins_retries % 100U) == 0U) {
                    //     REMUS_INFO(
                    //         "PutInsertIfAbsent retry_milestone tid={} key={} retries={} "
                    //         "home_bucket_version={}",
                    //         ct->get_tid(), key, ins_retries, home_ver_ins);
                    // }
                    bool ins_retry = false;
                    remus::rdma_ptr<KVSlot> free_slot_ptr{};
                    remus::rdma_ptr<Bucket> free_bucket_ptr{};
                    if (TryFindInNeighborhood(ct, key, home, nullptr, nullptr, nullptr,
                                              &free_slot_ptr, nullptr, &free_bucket_ptr,
                                              &ins_retry)) {
                        return false;
                    }
                    if (ins_retry) {
                        Backoff(++ins_retries);
                        continue;
                    }
                    if (TryFindInOverflow(ct, key, home, nullptr, nullptr, nullptr,
                                          &ins_retry)) {
                        return false;
                    }
                    if (ins_retry) {
                        Backoff(++ins_retries);
                        continue;
                    }

                    if (free_slot_ptr.raw() != 0 && free_bucket_ptr.raw() != 0) {
                        uint64_t locked_even = 0;
                        if (!TryLockBucket(ct, free_bucket_ptr, &locked_even)) {
                            Backoff(++ins_retries);
                            continue;
                        }
                        KVSlot current = ct->Read<KVSlot>(free_slot_ptr);
                        if (!current.occupied) {
                            KVSlot to_write{};
                            to_write.occupied = true;
                            to_write.key = key;
                            to_write.value = value;
                            ct->Write<KVSlot>(free_slot_ptr, to_write);
                            UnlockBucket(ct, free_bucket_ptr, locked_even);
                            return true;
                        }
                        UnlockBucket(ct, free_bucket_ptr, locked_even);
                        Backoff(++ins_retries);
                        continue;
                    }

                    // Neighborhood buckets are full (no free slot in the H_ window): try
                    // hopscotch bubbling—move entries backward along the ring so a hole migrates
                    // into `home`’s neighborhood (bucket-major slot scan, same order as lookup).
                    // If no legal move exists (dense table / H too small), append to overflow.
                    const BubbleInsertOutcome bubble_result =
                        HopscotchBubbleInsert_(ct, key, value, home);
                    if (bubble_result == BubbleInsertOutcome::Inserted) {
                        return true;
                    }
                    if (bubble_result == BubbleInsertOutcome::Retry) {
                        Backoff(++ins_retries);
                        continue;
                    }

                    auto home_bucket_ptr = GetBucketPtr(home);
                    uint64_t locked_even = 0;
                    if (!TryLockBucket(ct, home_bucket_ptr, &locked_even)) {
                        Backoff(++ins_retries);
                        continue;
                    }
                    const bool ok = AppendToOverflow(ct, key, value, home);
                    UnlockBucket(ct, home_bucket_ptr, locked_even);
                    return ok;
                }
            }

        public:
            struct OccupancyStats {
                uint64_t total_buckets = 0;
                uint64_t occupied_buckets = 0;
                uint64_t buckets_with_overflow_chains = 0;
                uint64_t total_overflow_chain_blocks = 0;
                uint64_t occupied_base_slots = 0;
                uint64_t occupied_overflow_slots = 0;
            };

            void DebugPrintKeyWindow(remus::ComputeThread *ct, const Key &key,
                                     uint64_t max_buckets = 4) const
            {
                REMUS_ASSERT(ct != nullptr, "DebugPrintKeyWindow: ct is null");
                auto home = LocateBucket(key);
                const uint64_t window =
                    std::min<uint64_t>(max_buckets,
                                       std::min<uint64_t>(num_buckets_, static_cast<uint64_t>(H_)));
                (void)window;
                for (uint64_t off = 0; off < window; ++off) {
                    const uint64_t gidx = (home.global_bucket_idx + off) % num_buckets_;
                    auto loc = LocateBucketByGlobalIndex(gidx);
                    auto bucket_ptr = GetBucketPtr(loc);
                    int occupied_slots = 0;
                    for (int s = 0; s < SLOTS_PER_BUCKET; ++s) {
                        const KVSlot slot = ct->Read<KVSlot>(GetSlotPtr(bucket_ptr, s));
                        if (slot.occupied) {
                            ++occupied_slots;
                        }
                    }

                    int overflow_occupied = 0;
                    int overflow_blocks = 0;
                    auto cur = remus::rdma_ptr<OverflowBlock>(
                        ct->Read<uint64_t>(
                            remus::rdma_ptr<uint64_t>(
                                GetOverflowHeadFieldPtr(bucket_ptr).raw())));
                    while (cur.raw() != 0) {
                        ++overflow_blocks;
                        for (int s = 0; s < SLOTS_PER_BLOCK; ++s) {
                            const KVSlot slot = ct->Read<KVSlot>(GetOverflowSlotPtr(cur, s));
                            if (slot.occupied) {
                                ++overflow_occupied;
                            }
                        }
                        cur = remus::rdma_ptr<OverflowBlock>(
                            ct->Read<uint64_t>(
                                remus::rdma_ptr<uint64_t>(
                                    GetOverflowNextFieldPtr(cur).raw())));
                    }

                    (void)gidx;
                    (void)loc;
                    (void)occupied_slots;
                    (void)overflow_occupied;
                    (void)overflow_blocks;
                }
            }

            OccupancyStats CollectOccupancyStats(remus::ComputeThread *ct) const
            {
                REMUS_ASSERT(ct != nullptr, "CollectOccupancyStats: ct is null");
                OccupancyStats stats{};
                stats.total_buckets = num_buckets_;

                for (uint64_t global_bucket_index = 0; global_bucket_index < num_buckets_;
                     ++global_bucket_index) {
                    const auto bucket_ptr =
                        GetBucketPtr(LocateBucketByGlobalIndex(global_bucket_index));
                    const Bucket bucket_snapshot = ct->Read<Bucket>(bucket_ptr);

                    bool bucket_has_occupied_slot = false;
                    for (int slot_index = 0; slot_index < SLOTS_PER_BUCKET; ++slot_index) {
                        if (bucket_snapshot.slots[slot_index].occupied) {
                            bucket_has_occupied_slot = true;
                            ++stats.occupied_base_slots;
                        }
                    }
                    if (bucket_has_occupied_slot) {
                        ++stats.occupied_buckets;
                    }

                    uint64_t chain_blocks = 0;
                    auto cur = bucket_snapshot.overflow_head;
                    while (cur.raw() != 0) {
                        ++chain_blocks;
                        for (int slot_index = 0; slot_index < SLOTS_PER_BLOCK; ++slot_index) {
                            const KVSlot overflow_slot =
                                ct->Read<KVSlot>(GetOverflowSlotPtr(cur, slot_index));
                            if (overflow_slot.occupied) {
                                ++stats.occupied_overflow_slots;
                            }
                        }
                        cur = remus::rdma_ptr<OverflowBlock>(
                            ct->Read<uint64_t>(
                                remus::rdma_ptr<uint64_t>(
                                    GetOverflowNextFieldPtr(cur).raw())));
                    }
                    if (chain_blocks != 0) {
                        ++stats.buckets_with_overflow_chains;
                        stats.total_overflow_chain_blocks += chain_blocks;
                    }
                }

                return stats;
            }

            /// Scalar read of the home bucket's version word for `key` (benchmark / logging).
            uint64_t ReadHomeBucketVersionForKey(remus::ComputeThread *ct,
                                                  const Key &key) const
            {
                REMUS_ASSERT(ct != nullptr, "ReadHomeBucketVersionForKey: ct is null");
                return ReadHomeBucketVersion_(ct, LocateBucket(key));
            }

            bool Put(remus::ComputeThread *ct, const Key &key, const Value &value)
            {
                REMUS_ASSERT(ct != nullptr, "Put: ct is null");
                auto home = LocateBucket(key);
                uint32_t retries = 0;
                while (true) {
                    CheckSpinLimit_("Put", key, retries);
                    // const uint64_t home_ver_put = ReadHomeBucketVersion_(ct, home);
                    // REMUS_INFO("Put tid={} key={} retries={} home_bucket_version={}",
                    //            ct->get_tid(), key, retries, home_ver_put);
                    // if ((retries % 100U) == 0U) {
                    //     REMUS_INFO(
                    //         "Put retry_milestone tid={} key={} retries={} home_bucket_version={}",
                    //         ct->get_tid(), key, retries, home_ver_put);
                    // }
                    bool retry = false;
                    remus::rdma_ptr<KVSlot> found_ptr{};
                    remus::rdma_ptr<Bucket> found_bucket{};
                    KVSlot found{};
                    if (TryFindInNeighborhood(ct, key, home, nullptr, &found_ptr, &found,
                                              nullptr, &found_bucket, nullptr, &retry)) {
                        uint64_t locked_even = 0;
                        if (!TryLockBucket(ct, found_bucket, &locked_even)) {
                            Backoff(++retries);
                            continue;
                        }
                        KVSlot cur = ct->Read<KVSlot>(found_ptr);
                        if (cur.occupied && cur.key == key) {
                            cur.value = value;
                            ct->Write<KVSlot>(found_ptr, cur);
                            UnlockBucket(ct, found_bucket, locked_even);
                            return true;
                        }
                        UnlockBucket(ct, found_bucket, locked_even);
                        Backoff(++retries);
                        continue;
                    }
                    if (retry) {
                        Backoff(++retries);
                        continue;
                    }
                    if (TryFindInOverflow(ct, key, home, nullptr, &found_ptr, &found,
                                          &retry)) {
                        auto home_bucket_ptr = GetBucketPtr(home);
                        uint64_t locked_even = 0;
                        if (!TryLockBucket(ct, home_bucket_ptr, &locked_even)) {
                            Backoff(++retries);
                            continue;
                        }
                        KVSlot cur = ct->Read<KVSlot>(found_ptr);
                        if (cur.occupied && cur.key == key) {
                            cur.value = value;
                            ct->Write<KVSlot>(found_ptr, cur);
                            UnlockBucket(ct, home_bucket_ptr, locked_even);
                            return true;
                        }
                        UnlockBucket(ct, home_bucket_ptr, locked_even);
                        Backoff(++retries);
                        continue;
                    }
                    if (retry) {
                        Backoff(++retries);
                        continue;
                    }

                    return PutInsertIfAbsent_(ct, key, value, home);
                }
            }
            bool Get(remus::ComputeThread *ct, const Key &key, Value &value)
            {
                REMUS_ASSERT(ct != nullptr, "Get: ct is null");
                auto home = LocateBucket(key);
                uint32_t retries = 0;
                while (true) {
                    CheckSpinLimit_("Get", key, retries);
                    // const uint64_t home_ver_get = ReadHomeBucketVersion_(ct, home);
                    // REMUS_INFO("Get tid={} key={} retries={} home_bucket_version={}",
                    //            ct->get_tid(), key, retries, home_ver_get);
                    // if ((retries % 100U) == 0U) {
                    //     REMUS_INFO(
                    //         "Get retry_milestone tid={} key={} retries={} home_bucket_version={}",
                    //         ct->get_tid(), key, retries, home_ver_get);
                    // }
                    bool retry = false;
                    if (TryFindInNeighborhood(ct, key, home, &value, nullptr, nullptr,
                                              nullptr, nullptr, nullptr, &retry)) {
                        return true;
                    }
                    if (retry) {
                        Backoff(++retries);
                        continue;
                    }
                    if (TryFindInOverflow(ct, key, home, &value, nullptr, nullptr,
                                          &retry)) {
                        return true;
                    }
                    if (retry) {
                        Backoff(++retries);
                        continue;
                    }
                    value = Value{};
                    return false;
                }
            }
            Value Remove(const Key &key)
            {
                (void)key;
                return Value{};
            }
            
        

};