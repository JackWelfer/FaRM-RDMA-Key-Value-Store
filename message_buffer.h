#pragma once

#include <cstddef>      // std::size_t
#include <cstdint>      // fixed-width ints (optional)
#include <map>
#include <stdexcept>    // std::runtime_error / std::out_of_range
#include <vector>

#include <remus/remus.h>

constexpr size_t CAPACITY = 4096;
constexpr int MAX_THREADS = 256;

// Root object containing all buffers on receivers and all cached heads on senders
struct MessageBuffers {
  uint64_t pair_ring_buffer_ptrs[MAX_THREADS * MAX_THREADS];
  uint64_t pair_cached_head_ptrs[MAX_THREADS * MAX_THREADS];
};

// Message Struct
struct Message {
    uint64_t numba;
};

// for writing message to the buffer as a single write
struct WireMessage {
    uint64_t length;
    Message payload;
    uint8_t trailer;
  };

// Circular buffer initalized in Remote Memory (fixed-size message slots).
struct ring_buffer {
  WireMessage slots[CAPACITY];
};
// Sender head pointer in Remote Memory (slot index).
struct sender_cached_head {
  uint64_t cached_head;
};

// Local Object for sending messages
struct message_sender {
  remus::ComputeThread *ct;
  
  remus::rdma_ptr<ring_buffer> ring_buf;
  remus::rdma_ptr<sender_cached_head> cached_head_ptr;
  
  uint64_t tail = 0;
  
  uint64_t SlotBase() const {
    return ring_buf.raw() + offsetof(ring_buffer, slots);
  }
  
  uint64_t CachedHeadBase() const {
    return cached_head_ptr.raw() + offsetof(sender_cached_head, cached_head);
  }
  
  uint64_t ModSlot(uint64_t slot) const { return slot % CAPACITY; }
  remus::rdma_ptr<WireMessage> SlotPtr(uint64_t slot_idx) const {
    return remus::rdma_ptr<WireMessage>(SlotBase() + slot_idx * sizeof(WireMessage));
  }

  bool TrySendMessage(const Message &message) {
    // Read sender-owned cached copy of receiver head.
    uint64_t safe_head =
        ct->Read<uint64_t>(remus::rdma_ptr<uint64_t>(CachedHeadBase()));
        safe_head = ModSlot(safe_head);

    // Ring-space check in slots: reserve one empty slot to avoid full==empty ambiguity.
    const uint64_t used = (tail + CAPACITY - safe_head) % CAPACITY;
    const uint64_t free_slots = CAPACITY - 1 - used;
    if (free_slots == 0) {
      return false;
    }

    WireMessage wire{};
    wire.length = sizeof(Message);
    wire.payload = message;
    wire.trailer = 1;
    ct->Write<WireMessage>(SlotPtr(tail), wire, true);

    // Advance local tail in slot ring.
    tail = ModSlot(tail + 1);

    return true;
  }
};


// Local Object for receiving messages
struct message_receiver {
  remus::ComputeThread *ct;

  remus::rdma_ptr<ring_buffer> ring_buf;
  remus::rdma_ptr<sender_cached_head> sender_cached_head_ptr;
  uint64_t head = 0;

  uint64_t SlotBase() const {
    return ring_buf.raw() +
           offsetof(ring_buffer, slots);
  }

  uint64_t CachedHeadBase() const {
    return sender_cached_head_ptr.raw() + offsetof(sender_cached_head, cached_head);
  }

  uint64_t ModSlot(uint64_t slot) const { return slot % CAPACITY; }
  remus::rdma_ptr<WireMessage> SlotPtr(uint64_t slot_idx) const {
    return remus::rdma_ptr<WireMessage>(SlotBase() + slot_idx * sizeof(WireMessage));
  }

  Message TryReadMessage() {
    WireMessage wire = ct->Read<WireMessage>(SlotPtr(head));
    // If length != sizeof(Message), no message is available
    if (wire.length != sizeof(Message)) {
      return Message();
    }

    // If trailer is 0, message is not complete
    if (wire.trailer == 0) {
      return Message();
    }

    Message message = wire.payload;

    // Consume this slot.
    ct->Write<WireMessage>(SlotPtr(head), WireMessage{});

    // Update head in slot ring.
    head = ModSlot(head + 1);

    // Receiver occasionally updates sender's cached head copy.
    if (head == 0 || head == (CAPACITY / 2)) {
      ct->Write<uint64_t>(remus::rdma_ptr<uint64_t>(CachedHeadBase()), head);
    }

    return message;
  }
};
// The working set of message related objects a thread needs after initialization finishes
struct message_runtime {
  std::vector<remus::rdma_ptr<ring_buffer>> owned_ring_buffers;
  std::vector<remus::rdma_ptr<sender_cached_head>> owned_cached_heads;
  std::map<uint64_t, message_sender> message_senders;
  std::map<uint64_t, message_receiver> message_receivers;
};

inline remus::rdma_ptr<MessageBuffers>
CacheAndClearMessageRoot(remus::ComputeThread *t, uint64_t total_threads,
                         uint64_t global_thread_id) {
  auto message_buffers = t->get_root<MessageBuffers>();
  t->arrive_control_barrier(total_threads);
  if (global_thread_id == 0) {
    t->set_root(remus::rdma_ptr<MessageBuffers>(nullptr));
  }
  t->arrive_control_barrier(total_threads);
  return message_buffers;
}

inline message_runtime
InitializeMessageRuntime(remus::ComputeThread *t,
                         remus::rdma_ptr<MessageBuffers> message_buffers,
                         uint64_t id, uint64_t c0, uint64_t cn, uint64_t threads,
                         uint64_t local_thread_id, uint64_t global_thread_id,
                         uint64_t total_threads) {
  message_runtime runtime;

  // Allocate + publish per-pair ring/cached-head.
  for (uint64_t remote_cn = c0; remote_cn <= cn; ++remote_cn) {
    if (remote_cn == id) {
      continue;
    }

    uint64_t remote_global_thread_id =
        (remote_cn - c0) * threads + local_thread_id;

    ring_buffer rb{};
    auto inbound_ring_ptr = t->allocate<ring_buffer>();
    t->Write<ring_buffer>(inbound_ring_ptr, rb);
    runtime.owned_ring_buffers.push_back(inbound_ring_ptr);
    uint64_t in_pair_idx = remote_global_thread_id * MAX_THREADS + global_thread_id;
    auto in_ring_slot_ptr = remus::rdma_ptr<uint64_t>(
        message_buffers.raw() + offsetof(MessageBuffers, pair_ring_buffer_ptrs) +
        in_pair_idx * sizeof(uint64_t));
    t->Write<uint64_t>(in_ring_slot_ptr, inbound_ring_ptr.raw());

    sender_cached_head sch{};
    sch.cached_head = 0;
    auto out_cached_head_ptr = t->allocate<sender_cached_head>();
    t->Write<sender_cached_head>(out_cached_head_ptr, sch);
    runtime.owned_cached_heads.push_back(out_cached_head_ptr);
    uint64_t out_pair_idx = global_thread_id * MAX_THREADS + remote_global_thread_id;
    auto out_cached_slot_ptr = remus::rdma_ptr<uint64_t>(
        message_buffers.raw() + offsetof(MessageBuffers, pair_cached_head_ptrs) +
        out_pair_idx * sizeof(uint64_t));
    t->Write<uint64_t>(out_cached_slot_ptr, out_cached_head_ptr.raw());
  }

  t->arrive_control_barrier(total_threads);

  // Build sender/receiver maps.
  for (uint64_t remote_cn = c0; remote_cn <= cn; ++remote_cn) {
    if (remote_cn == id) {
      continue;
    }

    uint64_t remote_global_thread_id =
        (remote_cn - c0) * threads + local_thread_id;

    uint64_t out_pair_idx = global_thread_id * MAX_THREADS + remote_global_thread_id;
    auto out_ring_slot_ptr = remus::rdma_ptr<uint64_t>(
        message_buffers.raw() + offsetof(MessageBuffers, pair_ring_buffer_ptrs) +
        out_pair_idx * sizeof(uint64_t));
    uint64_t out_ring_raw = t->Read<uint64_t>(out_ring_slot_ptr);
    auto out_ring_buf_ptr = remus::rdma_ptr<ring_buffer>(out_ring_raw);

    auto out_cached_slot_ptr = remus::rdma_ptr<uint64_t>(
        message_buffers.raw() + offsetof(MessageBuffers, pair_cached_head_ptrs) +
        out_pair_idx * sizeof(uint64_t));
    uint64_t out_cached_raw = t->Read<uint64_t>(out_cached_slot_ptr);
    auto out_cached_head_ptr = remus::rdma_ptr<sender_cached_head>(out_cached_raw);

    runtime.message_senders.emplace(
        remote_cn, message_sender{t, out_ring_buf_ptr, out_cached_head_ptr});

    uint64_t in_pair_idx = remote_global_thread_id * MAX_THREADS + global_thread_id;
    auto in_ring_slot_ptr = remus::rdma_ptr<uint64_t>(
        message_buffers.raw() + offsetof(MessageBuffers, pair_ring_buffer_ptrs) +
        in_pair_idx * sizeof(uint64_t));
    uint64_t in_ring_raw = t->Read<uint64_t>(in_ring_slot_ptr);
    auto in_ring_buf_ptr = remus::rdma_ptr<ring_buffer>(in_ring_raw);

    auto in_cached_slot_ptr = remus::rdma_ptr<uint64_t>(
        message_buffers.raw() + offsetof(MessageBuffers, pair_cached_head_ptrs) +
        in_pair_idx * sizeof(uint64_t));
    uint64_t in_cached_raw = t->Read<uint64_t>(in_cached_slot_ptr);
    auto in_cached_head_ptr = remus::rdma_ptr<sender_cached_head>(in_cached_raw);

    runtime.message_receivers.emplace(
        remote_cn, message_receiver{t, in_ring_buf_ptr, in_cached_head_ptr, 0});
  }

  return runtime;
}

