#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <remus/remus.h>
#include <thread>
#include <vector>

#include "cloudlab.h"
#include "../src/hopscotch_hashtable.h"
#include "../src/message_buffer.h"

namespace {

// Opaque: same raw address as hopscotch HashTableRoot (private in class).
struct HashtableRootOpaque {};

// Hopscotch uses SLOTS_PER_BUCKET == 16 (see hopscotch_hashtable.h).
constexpr int kSlotsPerBucket = 16;
constexpr int kNeighborhoodSize = 32;
constexpr uint64_t kBucketsPerComputeNode = 512;

// Do not drive inserts past this fraction of total slot capacity (default 90%).
constexpr int kTargetLoadPercent = 90;
// Progress log when local insert phase crosses each N% of the per-thread budget.
constexpr int kProgressLogEveryPercent = 10;
// After the insert phase, each thread runs this many mixed ops (Get vs Put-update).
constexpr uint32_t kPostFillMixedOpsPerThread = 250000;
// Mixed phase: uniform draw in [0,100); values < this → Get, else → Put-update.
constexpr int kMixedPhaseGetPercent = 5;
// Safety: stop insert attempts after this many times the per-thread insert budget.
constexpr uint32_t kMaxInsertAttemptsFactor = 40;

struct SplitMix64 {
  uint64_t state_;
  explicit SplitMix64(uint64_t seed) : state_(seed + 0x9E3779B97F4A7C15ULL) {}
  uint64_t Next() {
    uint64_t z = (state_ += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  uint64_t NextBounded(uint64_t bound) {
    if (bound == 0) {
      return 0;
    }
    return Next() % bound;
  }
};

struct ThreadBenchStats {
  uint64_t insert_ok = 0;
  uint64_t insert_fail = 0;
  uint64_t insert_attempts = 0;
  uint64_t phase2_ops = 0;
  uint64_t get_hits = 0;
  double phase1_seconds = 0;
  double phase2_seconds = 0;
};

struct BenchRoot {
  ThreadBenchStats per_thread[MAX_THREADS];
};

}  // namespace

int main(int argc, char **argv) {
  remus::INIT();

  auto args = std::make_shared<remus::ArgMap>();
  args->import(remus::ARGS);
  args->parse(argc, argv);
  if (args->bget(remus::HELP)) {
    args->usage();
    return 0;
  }
  args->report_config();

  const uint64_t id = args->uget(remus::NODE_ID);
  const uint64_t m0 = args->uget(remus::FIRST_MN_ID);
  const uint64_t mn = args->uget(remus::LAST_MN_ID);
  const uint64_t c0 = args->uget(remus::FIRST_CN_ID);
  const uint64_t cn = args->uget(remus::LAST_CN_ID);
  const uint64_t threads = args->uget(remus::CN_THREADS);

  std::vector<remus::MachineInfo> memnodes;
  for (uint64_t i = m0; i <= mn; ++i) {
    memnodes.emplace_back(i, id_to_dns_name(i));
  }

  remus::MachineInfo self(id, id_to_dns_name(id));

  std::unique_ptr<remus::MemoryNode> memory_node;
  if (id >= m0 && id <= mn) {
    memory_node.reset(new remus::MemoryNode(self, args));
  }

  std::shared_ptr<remus::ComputeNode> compute_node;
  if (id >= c0 && id <= cn) {
    compute_node.reset(new remus::ComputeNode(self, args));
    if (memory_node.get() != nullptr) {
      compute_node->connect_local(memnodes, memory_node->get_local_rkeys());
    }
    compute_node->connect_remote(memnodes);
  }

  if (memory_node) {
    memory_node->init_done();
  }

  if (id >= c0 && id <= cn) {
    std::vector<std::shared_ptr<remus::ComputeThread>> compute_threads;
    const uint64_t total_threads = (cn - c0 + 1) * threads;
    REMUS_ASSERT(total_threads <= static_cast<uint64_t>(MAX_THREADS),
                 "Total threads ({}) cannot exceed MAX_THREADS ({})", total_threads,
                 MAX_THREADS);
    REMUS_ASSERT(total_threads <= static_cast<uint64_t>(INT32_MAX),
                 "Total threads ({}) exceeds INT32_MAX", total_threads);

    const uint64_t num_compute_nodes = (cn - c0 + 1);
    REMUS_ASSERT(num_compute_nodes >= 1, "launch needs at least one compute node");

    for (uint64_t i = 0; i < threads; ++i) {
      compute_threads.push_back(
          std::make_shared<remus::ComputeThread>(id, compute_node, args));
    }

    const uint64_t node_id = id - c0;
    const uint64_t total_buckets = num_compute_nodes * kBucketsPerComputeNode;
    const uint64_t total_slots = total_buckets * static_cast<uint64_t>(kSlotsPerBucket);
    const uint64_t slot_budget_global =
        (total_slots * static_cast<uint64_t>(kTargetLoadPercent)) / 100ULL;
    const uint64_t insert_budget_per_thread =
        std::max<uint64_t>(1ULL, slot_budget_global / total_threads);
    const uint64_t max_insert_attempts =
        insert_budget_per_thread * static_cast<uint64_t>(kMaxInsertAttemptsFactor);

    hopscotch_hashtable<uint64_t, uint64_t> table;
    table.Init(compute_threads[0].get(), static_cast<int>(num_compute_nodes),
               static_cast<int>(node_id), kNeighborhoodSize, total_buckets);

    if (id == c0) {
      REMUS_INFO(
          "launch hashtable bench: nodes={} threads_per_node={} total_threads={} "
          "buckets={} slots={} target_load_pct={} insert_budget_per_thread={} "
          "post_fill_ops_per_thread={} mixed_get_pct={}",
          num_compute_nodes, threads, total_threads, total_buckets, total_slots,
          kTargetLoadPercent, insert_budget_per_thread, kPostFillMixedOpsPerThread,
          kMixedPhaseGetPercent);
    }

    remus::rdma_ptr<HashtableRootOpaque> saved_hash_root;
    if (id == c0) {
      saved_hash_root = compute_threads[0]->get_root<HashtableRootOpaque>();
    }

    remus::rdma_ptr<BenchRoot> bench_root;
    if (id == c0) {
      bench_root = compute_threads[0]->allocate<BenchRoot>();
      BenchRoot cleared{};
      compute_threads[0]->Write<BenchRoot>(bench_root, cleared);
      compute_threads[0]->set_root(bench_root);
    }

    compute_threads[0]->arrive_control_barrier(static_cast<int>(num_compute_nodes));
    if (id != c0) {
      bench_root = compute_threads[0]->get_root<BenchRoot>();
    }
    REMUS_ASSERT(bench_root.raw() != 0, "bench root not visible on CN{}", id);

    auto run_bench = [&](uint64_t local_tid, remus::ComputeThread *ct) {
      const uint64_t global_tid = node_id * threads + local_tid;
      SplitMix64 rng(0x9E3779B97F4A7C15ULL ^ (global_tid * 0xD6E8FEB8669FD5B1ULL));

      std::vector<uint64_t> keys;
      keys.reserve(static_cast<size_t>(
          std::min<uint64_t>(insert_budget_per_thread + 1024, 1ULL << 20)));

      uint64_t insert_ok = 0;
      uint64_t insert_fail = 0;
      uint64_t insert_attempts = 0;
      int last_logged_pct = -kProgressLogEveryPercent;

      ct->arrive_control_barrier(static_cast<int>(total_threads));
      const auto phase1_start = std::chrono::steady_clock::now();

      while (insert_ok < insert_budget_per_thread &&
             insert_attempts < max_insert_attempts) {
        ++insert_attempts;
        const uint64_t key =
            (global_tid << 48) ^ (insert_attempts << 16) ^ (rng.Next() & 0xffffULL);
        const uint64_t val = rng.Next() ^ (insert_ok << 1);
        if (table.Put(ct, key, val)) {
          ++insert_ok;
          keys.push_back(key);
          const int pct = static_cast<int>((insert_ok * 100) / insert_budget_per_thread);
          if (pct >= last_logged_pct + kProgressLogEveryPercent) {
            last_logged_pct = (pct / kProgressLogEveryPercent) * kProgressLogEveryPercent;
            REMUS_INFO(
                "launch progress CN{} tid={} phase=insert inserts_ok={}/{} "
                "attempts={} fail={} pct_of_budget={}",
                id, local_tid, insert_ok, insert_budget_per_thread, insert_attempts,
                insert_fail, pct);
          }
        } else {
          ++insert_fail;
        }
      }

      const auto phase1_end = std::chrono::steady_clock::now();
      const double phase1_s =
          std::chrono::duration<double>(phase1_end - phase1_start).count();

      REMUS_INFO(
          "launch progress CN{} tid={} phase=insert_done inserts_ok={}/{} "
          "attempts={} fail={} phase1_s={:.3f}",
          id, local_tid, insert_ok, insert_budget_per_thread, insert_attempts,
          insert_fail, phase1_s);

      uint64_t phase2_ops = 0;
      uint64_t get_hits = 0;
      int last_phase2_pct = -kProgressLogEveryPercent;
      const auto phase2_start = std::chrono::steady_clock::now();

      for (uint32_t i = 0; i < kPostFillMixedOpsPerThread; ++i) {
        ++phase2_ops;
        if (keys.empty()) {
          break;
        }
        if (static_cast<int>(rng.NextBounded(100)) < kMixedPhaseGetPercent) {
          const uint64_t key = keys[rng.NextBounded(keys.size())];
          uint64_t out{};
          if (table.Get(ct, key, out)) {
            ++get_hits;
          }
        } else {
          const uint64_t key = keys[rng.NextBounded(keys.size())];
          const uint64_t val = rng.Next() ^ static_cast<uint64_t>(i);
          (void)table.Put(ct, key, val);
        }
        const int pct = static_cast<int>((static_cast<uint64_t>(i + 1) * 100) /
                                           kPostFillMixedOpsPerThread);
        if (pct >= last_phase2_pct + kProgressLogEveryPercent) {
          last_phase2_pct = (pct / kProgressLogEveryPercent) * kProgressLogEveryPercent;
          REMUS_INFO(
              "launch progress CN{} tid={} phase=mixed ops={}/{} get_hits={} "
              "pct_of_phase2={}",
              id, local_tid, phase2_ops, kPostFillMixedOpsPerThread, get_hits, pct);
        }
      }

      const auto phase2_end = std::chrono::steady_clock::now();
      const double phase2_s =
          std::chrono::duration<double>(phase2_end - phase2_start).count();

      REMUS_INFO(
          "launch progress CN{} tid={} phase=done phase2_ops={} get_hits={} "
          "phase2_s={:.3f}",
          id, local_tid, phase2_ops, get_hits, phase2_s);

      ThreadBenchStats row{};
      row.insert_ok = insert_ok;
      row.insert_fail = insert_fail;
      row.insert_attempts = insert_attempts;
      row.phase2_ops = phase2_ops;
      row.get_hits = get_hits;
      row.phase1_seconds = phase1_s;
      row.phase2_seconds = phase2_s;

      const auto row_ptr = remus::rdma_ptr<ThreadBenchStats>(
          bench_root.raw() + offsetof(BenchRoot, per_thread) +
          global_tid * sizeof(ThreadBenchStats));
      ct->Write<ThreadBenchStats>(row_ptr, row);
      ct->arrive_control_barrier(static_cast<int>(total_threads));
    };

    std::vector<std::thread> workers;
    for (uint64_t ti = 1; ti < threads; ++ti) {
      workers.emplace_back(
          [&, ti]() { run_bench(ti, compute_threads[ti].get()); });
    }
    run_bench(0, compute_threads[0].get());
    for (auto &w : workers) {
      if (w.joinable()) {
        w.join();
      }
    }

    if (id == c0) {
      uint64_t sum_ins_ok = 0;
      uint64_t sum_ins_fail = 0;
      uint64_t sum_attempts = 0;
      uint64_t sum_p2 = 0;
      uint64_t sum_get_hits = 0;
      double max_p1 = 0.0;
      double max_p2 = 0.0;

      for (uint64_t gt = 0; gt < total_threads; ++gt) {
        const auto row_ptr = remus::rdma_ptr<ThreadBenchStats>(
            bench_root.raw() + offsetof(BenchRoot, per_thread) +
            gt * sizeof(ThreadBenchStats));
        const auto r = compute_threads[0]->Read<ThreadBenchStats>(row_ptr);
        sum_ins_ok += r.insert_ok;
        sum_ins_fail += r.insert_fail;
        sum_attempts += r.insert_attempts;
        sum_p2 += r.phase2_ops;
        sum_get_hits += r.get_hits;
        if (r.phase1_seconds > max_p1) {
          max_p1 = r.phase1_seconds;
        }
        if (r.phase2_seconds > max_p2) {
          max_p2 = r.phase2_seconds;
        }
      }

      const double insert_rate =
          max_p1 > 0.0 ? static_cast<double>(sum_ins_ok) / max_p1 : 0.0;
      const double mixed_rate =
          max_p2 > 0.0 ? static_cast<double>(sum_p2) / max_p2 : 0.0;
      REMUS_INFO(
          "launch hashtable bench summary: target_load_pct={} slot_budget_global={} "
          "sum_insert_ok={} sum_insert_fail={} sum_attempts={} phase1_max_s={:.6f} "
          "insert_ok_per_s={:.1f} phase2_ops={} get_hits={} phase2_max_s={:.6f} "
          "phase2_ops_per_s={:.1f}",
          kTargetLoadPercent, slot_budget_global, sum_ins_ok, sum_ins_fail,
          sum_attempts, max_p1, insert_rate, sum_p2, sum_get_hits, max_p2,
          mixed_rate);

      compute_threads[0]->set_root(saved_hash_root);
      compute_threads[0]->deallocate(bench_root);
    }

    compute_threads[0]->arrive_control_barrier(static_cast<int>(num_compute_nodes));
  }

  return 0;
}
