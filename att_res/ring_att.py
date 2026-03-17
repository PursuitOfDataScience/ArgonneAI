"""
Ring Attention: Context Parallelism for Long Sequences

Distributes a long sequence across GPUs along the sequence dimension.
Each GPU holds a chunk of Q, K, V and passes KV blocks around a ring,
computing partial attention at each step and merging via online softmax.
Produces exact results (no approximation).

Usage for continued pretraining:
    1. Set block_size = full_context (e.g., 8192)
    2. Create context parallel group across GPUs
    3. DataLoader serves full-length sequences, each GPU takes its chunk
    4. Model uses ring_attention instead of standard attention

Reference: Liu et al., "Ring Attention with Blockwise Transformers
for Near-Infinite Context", ICLR 2024. arxiv.org/abs/2310.01889
"""

import math
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Tuple


# ============================================================
# Ring communication primitives
# ============================================================

def _ring_send_recv_kv(
    k_send: torch.Tensor,
    v_send: torch.Tensor,
    k_recv: torch.Tensor,
    v_recv: torch.Tensor,
    group=None,
):
    """
    Send KV to next rank in ring, receive from previous rank.
    Uses batched non-blocking P2P ops to avoid communicator instability seen
    with unbatched isend/recv on some NCCL setups.
    Ring direction: rank -> rank+1 (wraps around)
    """
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    send_to = (rank + 1) % world_size
    recv_from = (rank - 1) % world_size

    # Keep contiguous send buffers alive until requests complete.
    k_send_buf = k_send.contiguous()
    v_send_buf = v_send.contiguous()
    ops = [
        dist.P2POp(dist.isend, k_send_buf, send_to, group),
        dist.P2POp(dist.irecv, k_recv, recv_from, group),
        dist.P2POp(dist.isend, v_send_buf, send_to, group),
        dist.P2POp(dist.irecv, v_recv, recv_from, group),
    ]
    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()


# ============================================================
# Online softmax merge (Milakov & Gimelshein, 2018)
# ============================================================

def _online_softmax_merge(
    o_acc: torch.Tensor,     # [B, H, S, D] accumulated output
    lse_acc: torch.Tensor,   # [B, H, S, 1] log-sum-exp accumulator
    o_new: torch.Tensor,     # [B, H, S, D] new partial output
    lse_new: torch.Tensor,   # [B, H, S, 1] new partial lse
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Merge two partial attention results exactly.
    Handles -inf lse (fully masked chunks) correctly.
    """
    lse_max = torch.maximum(lse_acc, lse_new)

    # When both are -inf, avoid nan: set max to 0 (result will be 0 anyway)
    both_neg_inf = (lse_acc == float('-inf')) & (lse_new == float('-inf'))
    lse_max = torch.where(both_neg_inf, torch.zeros_like(lse_max), lse_max)

    exp_acc = torch.exp(lse_acc - lse_max)
    exp_new = torch.exp(lse_new - lse_max)

    # Replace nan from 0/0 with 0
    exp_acc = torch.nan_to_num(exp_acc, nan=0.0)
    exp_new = torch.nan_to_num(exp_new, nan=0.0)

    denom = exp_acc + exp_new
    denom = torch.clamp(denom, min=1e-12)  # avoid division by zero

    o_combined = (exp_acc * o_acc + exp_new * o_new) / denom
    lse_combined = lse_max + torch.log(denom)

    return o_combined, lse_combined


# ============================================================
# Chunk-level attention computation
# ============================================================

def _chunk_attention_fwd(
    q: torch.Tensor,         # [B, H, Sq, D]
    k: torch.Tensor,         # [B, H, Sk, D]
    v: torch.Tensor,         # [B, H, Sk, D]
    is_same_chunk: bool,     # causal mask within chunk
    is_future_chunk: bool,   # entirely masked (future tokens)
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute attention for Q against one KV chunk.
    Returns (output, log_sum_exp) for online softmax merging.
    """
    if is_future_chunk:
        # Future chunks: all masked, contribute nothing
        o = torch.zeros_like(q)
        lse = torch.full(
            (q.shape[0], q.shape[1], q.shape[2], 1),
            float('-inf'), device=q.device, dtype=q.dtype,
        )
        return o, lse

    # [B, H, Sq, Sk]
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if is_same_chunk:
        Sq, Sk = q.shape[-2], k.shape[-2]
        causal = torch.triu(
            torch.ones(Sq, Sk, dtype=torch.bool, device=q.device),
            diagonal=1,
        )
        scores = scores.masked_fill(causal, float('-inf'))

    lse = torch.logsumexp(scores, dim=-1, keepdim=True)  # [B, H, Sq, 1]
    attn = torch.exp(scores - lse)  # [B, H, Sq, Sk]

    # Handle nan from exp(-inf - (-inf)) = exp(nan)
    attn = torch.nan_to_num(attn, nan=0.0)

    o = torch.matmul(attn, v)  # [B, H, Sq, D]
    return o, lse


def _chunk_attention_bwd(
    grad_output: torch.Tensor,  # [B, H, Sq, D]
    q: torch.Tensor,            # [B, H, Sq, D]
    k: torch.Tensor,            # [B, H, Sk, D]
    v: torch.Tensor,            # [B, H, Sk, D]
    o_global: torch.Tensor,     # [B, H, Sq, D] full output from forward
    lse_global: torch.Tensor,   # [B, H, Sq, 1] global lse from forward
    is_same_chunk: bool,
    is_future_chunk: bool,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute gradients for one Q-KV chunk pair.
    Uses globally-normalized attention weights (via lse_global).
    Returns (dq, dk, dv).
    """
    if is_future_chunk:
        return (
            torch.zeros_like(q),
            torch.zeros_like(k),
            torch.zeros_like(v),
        )

    # Mixed precision safe path: compute chunk backward in fp32, then cast back.
    compute_dtype = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
    grad_output_f = grad_output.to(compute_dtype)
    q_f = q.to(compute_dtype)
    k_f = k.to(compute_dtype)
    v_f = v.to(compute_dtype)
    o_global_f = o_global.to(compute_dtype)
    lse_global_f = lse_global.to(compute_dtype)

    # Recompute scores
    scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale  # [B, H, Sq, Sk]

    if is_same_chunk:
        Sq, Sk = q_f.shape[-2], k_f.shape[-2]
        causal = torch.triu(
            torch.ones(Sq, Sk, dtype=torch.bool, device=q_f.device),
            diagonal=1,
        )
        scores = scores.masked_fill(causal, float('-inf'))

    # Globally-normalized weights: P = exp(scores - lse_global)
    P = torch.exp(scores - lse_global_f)  # [B, H, Sq, Sk]
    P = torch.nan_to_num(P, nan=0.0)

    # dv = P^T @ grad_output
    dv = torch.matmul(P.transpose(-2, -1), grad_output_f)  # [B, H, Sk, D]

    # dp = grad_output @ v^T
    dp_raw = torch.matmul(grad_output_f, v_f.transpose(-2, -1))  # [B, H, Sq, Sk]

    # Softmax backward: dscores = P * (dp_raw - sum(grad_output * o_global, dim=-1))
    correction = (grad_output_f * o_global_f).sum(dim=-1, keepdim=True)  # [B, H, Sq, 1]
    dscores = P * (dp_raw - correction)  # [B, H, Sq, Sk]

    # dq = dscores @ k * scale
    dq = torch.matmul(dscores, k_f) * scale  # [B, H, Sq, D]

    # dk = dscores^T @ q * scale
    dk = torch.matmul(dscores.transpose(-2, -1), q_f) * scale  # [B, H, Sk, D]

    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


# ============================================================
# GQA: repeat KV heads to match Q heads
# ============================================================

def _repeat_kv(x: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Repeat KV heads to match Q head count for GQA."""
    if num_groups == 1:
        return x
    B, H_kv, S, D = x.shape
    x = x[:, :, None, :, :].expand(B, H_kv, num_groups, S, D)
    return x.reshape(B, H_kv * num_groups, S, D)


def _unrepeat_kv_grad(dx: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Sum repeated head gradients back to KV head count."""
    if num_groups == 1:
        return dx
    B, H_q, S, D = dx.shape
    H_kv = H_q // num_groups
    dx = dx.reshape(B, H_kv, num_groups, S, D)
    return dx.sum(dim=2)


# ============================================================
# Ring Attention autograd function
# ============================================================

class RingAttentionFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,       # [B, H_q, S_local, D]
        k: torch.Tensor,       # [B, H_kv, S_local, D]
        v: torch.Tensor,       # [B, H_kv, S_local, D]
        num_kv_groups: int,     # H_q // H_kv for GQA
        cp_group,               # context parallel process group
    ) -> torch.Tensor:
        rank = dist.get_rank(cp_group)
        world_size = dist.get_world_size(cp_group)
        scale = q.shape[-1] ** -0.5

        # Repeat KV for GQA (communication uses compact KV, compute uses repeated)
        k_full = _repeat_kv(k, num_kv_groups)
        v_full = _repeat_kv(v, num_kv_groups)

        B, H_q, S_local, D = q.shape

        # Accumulators
        o_acc = torch.zeros_like(q)
        lse_acc = torch.full(
            (B, H_q, S_local, 1), float('-inf'),
            device=q.device, dtype=q.dtype,
        )

        # Current KV being processed (starts as local, repeated for GQA)
        k_cur = k_full
        v_cur = v_full

        # Buffers for ring (use compact KV for communication to save bandwidth)
        k_compact_cur = k.clone()
        v_compact_cur = v.clone()
        k_compact_recv = torch.empty_like(k)
        v_compact_recv = torch.empty_like(v)

        # Save all KV chunks (compact) for backward
        all_k_compact = []
        all_v_compact = []

        for step in range(world_size):
            source_rank = (rank - step) % world_size
            is_same = (source_rank == rank)
            is_future = (source_rank > rank)

            # Save compact KV for backward
            all_k_compact.append(k_compact_cur.clone())
            all_v_compact.append(v_compact_cur.clone())

            # Compute partial attention
            o_new, lse_new = _chunk_attention_fwd(
                q, k_cur, v_cur,
                is_same_chunk=is_same,
                is_future_chunk=is_future,
                scale=scale,
            )

            # Merge
            o_acc, lse_acc = _online_softmax_merge(o_acc, lse_acc, o_new, lse_new)

            # Ring pass compact KV (not on last step)
            if step < world_size - 1:
                _ring_send_recv_kv(
                    k_compact_cur, v_compact_cur,
                    k_compact_recv, v_compact_recv,
                    group=cp_group,
                )
                k_compact_cur = k_compact_recv.clone()
                v_compact_cur = v_compact_recv.clone()
                # Repeat for next step's compute
                k_cur = _repeat_kv(k_compact_cur, num_kv_groups)
                v_cur = _repeat_kv(v_compact_cur, num_kv_groups)

        # Save for backward
        ctx.save_for_backward(q, o_acc, lse_acc, *all_k_compact, *all_v_compact)
        ctx.world_size = world_size
        ctx.rank = rank
        ctx.cp_group = cp_group
        ctx.num_kv_groups = num_kv_groups
        ctx.scale = scale

        return o_acc

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        saved = ctx.saved_tensors
        q = saved[0]
        o_acc = saved[1]
        lse_acc = saved[2]
        world_size = ctx.world_size
        rank = ctx.rank
        cp_group = ctx.cp_group
        num_kv_groups = ctx.num_kv_groups
        scale = ctx.scale

        all_k_compact = list(saved[3 : 3 + world_size])
        all_v_compact = list(saved[3 + world_size : 3 + 2 * world_size])

        B, H_q, S_local, D = q.shape
        H_kv = H_q // num_kv_groups

        # Gradient accumulators
        dq = torch.zeros_like(q)

        # Compute dk/dv for each chunk, organized by destination rank
        dk_by_dest = [torch.zeros(B, H_kv, S_local, D, device=q.device, dtype=q.dtype)
                      for _ in range(world_size)]
        dv_by_dest = [torch.zeros(B, H_kv, S_local, D, device=q.device, dtype=q.dtype)
                      for _ in range(world_size)]

        for step in range(world_size):
            source_rank = (rank - step) % world_size
            is_same = (source_rank == rank)
            is_future = (source_rank > rank)

            # Repeat KV for GQA compute
            k_step = _repeat_kv(all_k_compact[step], num_kv_groups)
            v_step = _repeat_kv(all_v_compact[step], num_kv_groups)

            dq_step, dk_step, dv_step = _chunk_attention_bwd(
                grad_output, q, k_step, v_step,
                o_acc, lse_acc,
                is_same_chunk=is_same,
                is_future_chunk=is_future,
                scale=scale,
            )

            dq += dq_step

            # Un-repeat dk/dv back to KV head count
            dk_compact = _unrepeat_kv_grad(dk_step, num_kv_groups)
            dv_compact = _unrepeat_kv_grad(dv_step, num_kv_groups)

            # Route to the rank that owns this KV
            dk_by_dest[source_rank] += dk_compact
            dv_by_dest[source_rank] += dv_compact

        # Exchange dk/dv via all_to_all:
        # dk_by_dest[r] goes to rank r, we receive from rank r
        dk_received = [torch.empty_like(dk_by_dest[0]) for _ in range(world_size)]
        dv_received = [torch.empty_like(dv_by_dest[0]) for _ in range(world_size)]

        dist.all_to_all(dk_received, dk_by_dest, group=cp_group)
        dist.all_to_all(dv_received, dv_by_dest, group=cp_group)

        # Sum all received contributions for our local KV
        dk_local = torch.stack(dk_received).sum(dim=0)
        dv_local = torch.stack(dv_received).sum(dim=0)

        return dq, dk_local, dv_local, None, None


# ============================================================
# Public API
# ============================================================

def ring_attention(
    q: torch.Tensor,        # [B, H_q, S_local, D]
    k: torch.Tensor,        # [B, H_kv, S_local, D]
    v: torch.Tensor,        # [B, H_kv, S_local, D]
    num_kv_groups: int = 1,
    cp_group=None,
) -> torch.Tensor:
    """
    Ring attention with context parallelism.

    Each GPU holds S_local = S_full / world_size tokens.
    Produces exact same result as full attention over S_full tokens.

    Args:
        q: [B, num_heads, S_local, head_dim]
        k: [B, num_kv_heads, S_local, head_dim]
        v: [B, num_kv_heads, S_local, head_dim]
        num_kv_groups: num_heads // num_kv_heads (for GQA)
        cp_group: context parallel process group

    Returns:
        output: [B, num_heads, S_local, head_dim]
    """
    if cp_group is None or not dist.is_initialized() or dist.get_world_size(cp_group) <= 1:
        # Fallback to standard attention
        k_full = _repeat_kv(k, num_kv_groups)
        v_full = _repeat_kv(v, num_kv_groups)
        scale = q.shape[-1] ** -0.5
        scores = torch.matmul(q, k_full.transpose(-2, -1)) * scale
        S = q.shape[-2]
        mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=q.device), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v_full)

    return RingAttentionFunc.apply(q, k, v, num_kv_groups, cp_group)


# ============================================================
# Context Parallel Data Loader
# ============================================================

class ContextParallelDataLoader:
    """
    DataLoader for context parallelism.
    Loads full sequences and gives each GPU its chunk.

    Each sample is block_size tokens. With cp_world_size GPUs,
    each GPU gets block_size // cp_world_size tokens.

    Args:
        filename: path to binary token file
        batch_size: samples per GPU (typically 1 for long context)
        block_size: FULL context length (e.g., 8192)
        cp_rank: rank within context parallel group
        cp_world_size: number of GPUs in context parallel group
    """

    def __init__(self, filename, batch_size, block_size, cp_rank=0, cp_world_size=1):
        import numpy as np
        self.B = batch_size
        self.T = block_size  # full context
        self.T_local = block_size // cp_world_size  # per-GPU chunk
        self.cp_rank = cp_rank
        self.cp_world_size = cp_world_size

        # Load binary data
        with open(filename, "rb") as f:
            header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
            if header[0] != 20240801:
                raise ValueError(f"Unknown magic: {header[0]}")
        self.tokens = np.memmap(filename, dtype=np.uint32, mode='r', offset=256 * 4)
        self.current_position = 0
        self.epoch = 0

        if cp_rank == 0:
            print(f"ContextParallelDataLoader: {len(self.tokens):,} tokens, "
                  f"full_context={block_size}, local_chunk={self.T_local}")

    def next_batch(self):
        import numpy as np
        B, T = self.B, self.T

        # Check if we need to wrap around
        end = self.current_position + B * T + 1
        if end > len(self.tokens):
            self.current_position = 0
            self.epoch += 1
            if self.cp_rank == 0:
                print(f"\n*** Epoch {self.epoch} completed ***\n")
            end = self.current_position + B * T + 1

        # Load full sequence (all ranks read same data)
        buf = self.tokens[self.current_position : end]
        buf = torch.tensor(buf.astype(np.int64), dtype=torch.long).pin_memory()

        # Each GPU takes its chunk
        # x: tokens to predict from, y: tokens to predict
        start = self.cp_rank * self.T_local
        x = buf[start : start + self.T_local * B].view(B, self.T_local)
        y = buf[start + 1 : start + 1 + self.T_local * B].view(B, self.T_local)

        # Advance position (same for all ranks)
        self.current_position += B * T

        return x, y

    def get_position(self):
        return self.current_position

    def set_position(self, position):
        self.current_position = position


# ============================================================
# Context Parallel Group Setup
# ============================================================

def create_context_parallel_group(world_size=None):
    """
    Create a context parallel group containing all GPUs.

    Returns:
        cp_group: process group for context parallelism
    """
    if not dist.is_initialized():
        return None

    if world_size is None:
        world_size = dist.get_world_size()

    ranks = list(range(world_size))
    cp_group = dist.new_group(ranks)
    return cp_group


# ============================================================
# Position ID helper for RoPE
# ============================================================

def get_context_parallel_position_ids(
    seq_len_local: int,
    cp_rank: int,
    cp_world_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate global position IDs for this GPU's chunk.
    GPU 0: [0, 1, ..., S_local-1]
    GPU 1: [S_local, ..., 2*S_local-1]
    etc.

    This ensures RoPE encodes the correct absolute positions.
    """
    offset = cp_rank * seq_len_local
    return torch.arange(offset, offset + seq_len_local, device=device)


# ============================================================
# Loss computation across context parallel group
# ============================================================

def context_parallel_cross_entropy(
    logits: torch.Tensor,    # [B, S_local, V]
    labels: torch.Tensor,    # [B, S_local]
    cp_group=None,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute cross-entropy loss on local chunk, then average across CP group.
    Each GPU computes loss on its own chunk, then we all-reduce.

    Args:
        logits: [B, S_local, V]
        labels: [B, S_local]
        cp_group: context parallel process group
        chunk_size: optional sequence chunk size for memory-friendly CE.
    """
    if chunk_size is None or chunk_size <= 0 or logits.size(1) <= chunk_size:
        loss_local = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
            reduction='sum',
        )
    else:
        B, S, V = logits.shape
        loss_local = None
        for start in range(0, S, chunk_size):
            end = min(start + chunk_size, S)
            chunk_loss = torch.nn.functional.cross_entropy(
                logits[:, start:end, :].reshape(-1, V),
                labels[:, start:end].reshape(-1),
                ignore_index=-100,
                reduction='sum',
            )
            loss_local = chunk_loss if loss_local is None else (loss_local + chunk_loss)
        if loss_local is None:
            loss_local = torch.zeros([], device=logits.device, dtype=logits.dtype)

    # Count valid tokens (float for all_reduce)
    valid_tokens = (labels != -100).sum().float()

    if cp_group is not None and dist.is_initialized() and dist.get_world_size(cp_group) > 1:
        dist.all_reduce(loss_local, op=dist.ReduceOp.SUM, group=cp_group)
        dist.all_reduce(valid_tokens, op=dist.ReduceOp.SUM, group=cp_group)

    return loss_local / valid_tokens.clamp(min=1)
