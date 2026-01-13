import tqdm
import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


from custom_models.lact_swiglu.ttt_operation import zeropower_via_newtonschulz5


def block_prefix_causal_linear_attention_recurrent(
    state: torch.Tensor, # [b, dv, dk]
    q: torch.Tensor, # [b, l, dk]
    k: torch.Tensor, # [b, l, dk]
    v: torch.Tensor, # [b, l, dv]
    chunk_size: int = 2048,
    use_muon: bool = True,
) -> torch.Tensor: # [b, l, dv]
    """
    block i attends to all previous blocks (< i) + initial state

    block 0 attends to only initial state.

    no within-block attention
    """
    q = q.transpose(1, 2)  # [b, dk, l]
    v = v.transpose(1, 2)  # [b, dv, l]

    output = torch.zeros_like(v)

    e_index = 0
    seq_len = k.shape[1]
    for i in range(0, seq_len - chunk_size, chunk_size):
        s_index = i
        e_index = s_index + chunk_size

        # [b, l, dk]
        ki = k[:, s_index:e_index, :]
        # [b, dv, l]
        vi = v[:, :, s_index:e_index]
        # [b, dk, l]
        qi = q[:, :, s_index:e_index]

        # get the final output
        # [b, dv, dk] @ [b, dk, l] -> [b, dv, l] -> [b, l, dv]
        output[:, :, s_index:e_index] = torch.bmm(state, qi)

        # [b, dv, l] @ [b, l, dk] -> [b, dv, dk]
        dstate = torch.bmm(vi, ki)
        if use_muon:
            dstate = zeropower_via_newtonschulz5(dstate)

        state = state + dstate

    # for the last chunk, don't update the fast weights, directly apply the fast weights to the query.
    s_index = e_index
    e_index = seq_len

    qi = q[:, :, s_index:e_index]
    # get the final output
    # [b, dv, dk] @ [b, dk, l] -> [b, dv, l] -> [b, l, dv]
    output[:, :, s_index:e_index] = torch.bmm(state, qi)

    return output.transpose(1, 2)


def block_prefix_causal_linear_attention_parallel(
    s0: torch.Tensor,  # [b, dv, dk]
    q: torch.Tensor,   # [b, l, dk]
    k: torch.Tensor,   # [b, l, dk]
    v: torch.Tensor,   # [b, l, dv]
    chunk_size: int = 2048,
    use_muon: bool = True,
) -> torch.Tensor:     # [b, l, dv]
    """
    block i attends to all previous blocks (< i) + initial state
    block 0 attends to only initial state
    no within-block attention

    Parallel form via per-block outer-product reduction + prefix-sum over blocks.
    """
    b, l, dk = q.shape
    dv = v.shape[-1]
    assert s0.shape == (b, dv, dk)
    assert k.shape == (b, l, dk)
    assert v.shape == (b, l, dv)

    # Pad sequence to multiple of chunk_size (padding tokens contribute zero to updates)
    n_blocks = (l + chunk_size - 1) // chunk_size
    l_pad = n_blocks * chunk_size
    pad = l_pad - l
    if pad:
        q_pad = F.pad(q, (0, 0, 0, pad))  # pad length dim
        k_pad = F.pad(k, (0, 0, 0, pad))
        v_pad = F.pad(v, (0, 0, 0, pad))
    else:
        q_pad, k_pad, v_pad = q, k, v

    # [b, n, c, dk/dv]
    q_blk = q_pad.view(b, n_blocks, chunk_size, dk)
    k_blk = k_pad.view(b, n_blocks, chunk_size, dk)
    v_blk = v_pad.view(b, n_blocks, chunk_size, dv)

    # Per-block update: dS[j] = sum_t v_t k_t^T  -> [b, n, dv, dk]
    # einsum: (b n t dv) x (b n t dk) -> (b n dv dk)
    dS = torch.einsum("bntd,bntk->bndk", v_blk, k_blk)
    if use_muon:
        dS = zeropower_via_newtonschulz5(dS.reshape(-1, dv, dk)).reshape_as(dS)

    # Prefix state BEFORE each block:
    # S_before[0] = s0
    # S_before[i] = s0 + sum_{j < i} dS[j]
    # Compute exclusive prefix-sum by shifting the inclusive cumsum.
    prefix_inclusive = dS.cumsum(dim=1)  # [b, n, dv, dk]
    prefix_exclusive = prefix_inclusive - dS
    S_before = prefix_exclusive + s0.unsqueeze(1)  # [b, n, dv, dk]

    # Output per block: o = S_before @ q (with q as vector on dk)
    # (b n dv dk) x (b n t dk) -> (b n t dv)
    out_blk = torch.einsum("bndk,bntk->bntd", S_before, q_blk)

    out = out_blk.reshape(b, l_pad, dv)
    return out[:, :l, :]


# @triton.jit
# def block_prefix_parallel_kernel(
#     Q, K, V, S0, Out,
#     stride_qb, stride_ql, stride_qd,
#     stride_kb, stride_kl, stride_kd,
#     stride_vb, stride_vl, stride_vd,
#     stride_s0b, stride_s0v, stride_s0k,
#     stride_ob, stride_ol, stride_od,
#     B, L, DK, DV, CHUNK_SIZE: tl.constexpr,
#     BLOCK_DK: tl.constexpr, BLOCK_DV: tl.constexpr,
# ):
#     """
#     Parallel block prefix causal linear attention kernel.
#     Each program processes one block (chunk) for one batch element.
#     """
#     pid_b = tl.program_id(0)
#     pid_block = tl.program_id(1)
    
#     block_start = pid_block * CHUNK_SIZE
#     block_end = tl.minimum(block_start + CHUNK_SIZE, L)
#     actual_chunk = block_end - block_start
    
#     # Allocate accumulator for state: [DV, DK]
#     acc_state = tl.zeros((BLOCK_DV, BLOCK_DK), dtype=tl.float32)
    
#     # Load initial state s0 into accumulator
#     offs_dv = tl.arange(0, BLOCK_DV)
#     offs_dk = tl.arange(0, BLOCK_DK)
    
#     for dv_start in range(0, DV, BLOCK_DV):
#         for dk_start in range(0, DK, BLOCK_DK):
#             dv_idx = dv_start + offs_dv
#             dk_idx = dk_start + offs_dk
#             dv_mask = dv_idx < DV
#             dk_mask = dk_idx < DK
            
#             s0_ptrs = S0 + pid_b * stride_s0b + dv_idx[:, None] * stride_s0v + dk_idx[None, :] * stride_s0k
#             s0_val = tl.load(s0_ptrs, mask=dv_mask[:, None] & dk_mask[None, :], other=0.0)
#             acc_state = s0_val
            
#             # Compute prefix sum of dS for blocks before current block
#             for prev_block in range(pid_block):
#                 prev_start = prev_block * CHUNK_SIZE
#                 prev_end = tl.minimum(prev_start + CHUNK_SIZE, L)
                
#                 # Compute dS for prev_block: sum_t v_t @ k_t^T
#                 dS_block = tl.zeros((BLOCK_DV, BLOCK_DK), dtype=tl.float32)
                
#                 for t in range(prev_start, prev_end):
#                     t_mask = t < L
                    
#                     # Load k[t]: [DK]
#                     k_ptrs = K + pid_b * stride_kb + t * stride_kl + dk_idx * stride_kd
#                     k_val = tl.load(k_ptrs, mask=dk_mask & t_mask, other=0.0)
                    
#                     # Load v[t]: [DV]
#                     v_ptrs = V + pid_b * stride_vb + t * stride_vl + dv_idx * stride_vd
#                     v_val = tl.load(v_ptrs, mask=dv_mask & t_mask, other=0.0)
                    
#                     # Outer product: v[:, None] @ k[None, :]
#                     dS_block += v_val[:, None] * k_val[None, :]
                
#                 acc_state += dS_block
            
#             # Now acc_state contains S_before[pid_block]
#             # Compute output for current block: S_before @ q
#             for t in range(block_start, block_end):
#                 t_mask = t < L
                
#                 # Load q[t]: [DK]
#                 q_ptrs = Q + pid_b * stride_qb + t * stride_ql + dk_idx * stride_qd
#                 q_val = tl.load(q_ptrs, mask=dk_mask & t_mask, other=0.0)
                
#                 # Compute output: acc_state @ q -> [DV]
#                 out_val = tl.sum(acc_state * q_val[None, :], axis=1)
                
#                 # Store output
#                 out_ptrs = Out + pid_b * stride_ob + t * stride_ol + dv_idx * stride_od
#                 tl.store(out_ptrs, out_val, mask=dv_mask & t_mask)


# @triton.jit
# def block_prefix_recurrent_fwd_kernel(
#     Q, K, V, S0, Out, State_out,
#     stride_qb, stride_ql, stride_qd,
#     stride_kb, stride_kl, stride_kd,
#     stride_vb, stride_vl, stride_vd,
#     stride_s0b, stride_s0v, stride_s0k,
#     stride_ob, stride_ol, stride_od,
#     stride_sb, stride_sv, stride_sk,
#     B, L, DK, DV, CHUNK_SIZE: tl.constexpr,
#     BLOCK_DK: tl.constexpr, BLOCK_DV: tl.constexpr,
# ):
#     """
#     Recurrent block prefix causal linear attention kernel.
#     Each program processes one batch element sequentially through blocks.
#     """
#     pid = tl.program_id(0)
    
#     offs_dv = tl.arange(0, BLOCK_DV)
#     offs_dk = tl.arange(0, BLOCK_DK)
    
#     # Process each block sequentially
#     num_blocks = tl.cdiv(L, CHUNK_SIZE)
    
#     # Load initial state
#     state = tl.zeros((BLOCK_DV, BLOCK_DK), dtype=tl.float32)
    
#     for dv_idx in range(0, DV, BLOCK_DV):
#         for dk_idx in range(0, DK, BLOCK_DK):
#             dv_offs = dv_idx + offs_dv
#             dk_offs = dk_idx + offs_dk
#             dv_mask = dv_offs < DV
#             dk_mask = dk_offs < DK
            
#             s0_ptrs = S0 + pid * stride_s0b + dv_offs[:, None] * stride_s0v + dk_offs[None, :] * stride_s0k
#             state = tl.load(s0_ptrs, mask=dv_mask[:, None] & dk_mask[None, :], other=0.0)
            
#             for blk_idx in range(num_blocks):
#                 block_start = blk_idx * CHUNK_SIZE
#                 block_end = tl.minimum(block_start + CHUNK_SIZE, L)
                
#                 # Compute outputs for this block using current state
#                 for t in range(block_start, block_end):
#                     t_mask = t < L
                    
#                     # Load q[t]
#                     q_ptrs = Q + pid * stride_qb + t * stride_ql + dk_offs * stride_qd
#                     q_val = tl.load(q_ptrs, mask=dk_mask & t_mask, other=0.0)
                    
#                     # Output = state @ q
#                     out_val = tl.sum(state * q_val[None, :], axis=1)
                    
#                     # Store output
#                     out_ptrs = Out + pid * stride_ob + t * stride_ol + dv_offs * stride_od
#                     tl.store(out_ptrs, out_val, mask=dv_mask & t_mask)
                
#                 # Update state (except for last block)
#                 if blk_idx < num_blocks - 1:
#                     for t in range(block_start, block_end):
#                         t_mask = t < L
                        
#                         # Load k[t] and v[t]
#                         k_ptrs = K + pid * stride_kb + t * stride_kl + dk_offs * stride_kd
#                         k_val = tl.load(k_ptrs, mask=dk_mask & t_mask, other=0.0)
                        
#                         v_ptrs = V + pid * stride_vb + t * stride_vl + dv_offs * stride_vd
#                         v_val = tl.load(v_ptrs, mask=dv_mask & t_mask, other=0.0)
                        
#                         # Update state: state += v[:, None] @ k[None, :]
#                         state += v_val[:, None] * k_val[None, :]
            
#             # Store final state
#             if State_out is not None:
#                 state_ptrs = State_out + pid * stride_sb + dv_offs[:, None] * stride_sv + dk_offs[None, :] * stride_sk
#                 tl.store(state_ptrs, state, mask=dv_mask[:, None] & dk_mask[None, :])


# def block_prefix_causal_linear_attention_triton_recurrent(
#     state: torch.Tensor,  # [b, dv, dk]
#     q: torch.Tensor,      # [b, l, dk]
#     k: torch.Tensor,      # [b, l, dk]
#     v: torch.Tensor,      # [b, l, dv]
#     chunk_size: int = 2048,
# ) -> torch.Tensor:
#     b, l, dk = q.shape
#     dv = v.shape[-1]
    
#     output = torch.zeros(b, l, dv, device=q.device, dtype=q.dtype)
    
#     BLOCK_DK = triton.next_power_of_2(min(dk, 64))
#     BLOCK_DV = triton.next_power_of_2(min(dv, 64))
    
#     grid = (b,)
    
#     block_prefix_recurrent_fwd_kernel[grid](
#         q, k, v, state, output, None,
#         q.stride(0), q.stride(1), q.stride(2),
#         k.stride(0), k.stride(1), k.stride(2),
#         v.stride(0), v.stride(1), v.stride(2),
#         state.stride(0), state.stride(1), state.stride(2),
#         output.stride(0), output.stride(1), output.stride(2),
#         0, 0, 0,  # state_out strides (not used)
#         b, l, dk, dv, chunk_size,
#         BLOCK_DK, BLOCK_DV,
#     )
    
#     return output


# def block_prefix_causal_linear_attention_triton_parallel(
#     s0: torch.Tensor,  # [b, dv, dk]
#     q: torch.Tensor,   # [b, l, dk]
#     k: torch.Tensor,   # [b, l, dk]
#     v: torch.Tensor,   # [b, l, dv]
#     chunk_size: int = 2048,
# ) -> torch.Tensor:
#     b, l, dk = q.shape
#     dv = v.shape[-1]
    
#     output = torch.zeros(b, l, dv, device=q.device, dtype=q.dtype)
    
#     num_blocks = (l + chunk_size - 1) // chunk_size
    
#     BLOCK_DK = triton.next_power_of_2(min(dk, 64))
#     BLOCK_DV = triton.next_power_of_2(min(dv, 64))
    
#     grid = (b, num_blocks)
    
#     block_prefix_parallel_kernel[grid](
#         q, k, v, s0, output,
#         q.stride(0), q.stride(1), q.stride(2),
#         k.stride(0), k.stride(1), k.stride(2),
#         v.stride(0), v.stride(1), v.stride(2),
#         s0.stride(0), s0.stride(1), s0.stride(2),
#         output.stride(0), output.stride(1), output.stride(2),
#         b, l, dk, dv, chunk_size,
#         BLOCK_DK, BLOCK_DV,
#     )
    
#     return output


@torch.no_grad()
def test_equiv(
    device="cuda" if torch.cuda.is_available() else "cpu",
    dtype=torch.bfloat16,
    benchmark=False,
):
    torch.manual_seed(44)

    # multiple configs, including non-multiple-of-chunk (padding path)
    if benchmark:
        configs = [
            dict(b=1, l=32768, dk=64, dv=64, chunk=2048),
        ]
    else:
        configs = [
            dict(b=1, l=1,    dk=8,  dv=7,  chunk=4),
            dict(b=2, l=63,   dk=16, dv=32, chunk=8),
            dict(b=3, l=128,  dk=32, dv=16, chunk=32),
            dict(b=2, l=257,  dk=64, dv=48, chunk=64),
            dict(b=1, l=2048, dk=32, dv=32, chunk=256),
        ]

    for cfg in configs:
        print(f"Testing {cfg}")
        b, l, dk, dv, chunk = cfg["b"], cfg["l"], cfg["dk"], cfg["dv"], cfg["chunk"]

        s0 = torch.rand(b, dv, dk, device=device, dtype=dtype)
        q  = torch.rand(b, l, dk, device=device, dtype=dtype)
        k  = torch.rand(b, l, dk, device=device, dtype=dtype)
        v  = torch.rand(b, l, dv, device=device, dtype=dtype)

        for _ in tqdm.trange(3000, disable=not benchmark): # 789 it/s; use moun: 119 it/s
            y_ref = block_prefix_causal_linear_attention_recurrent(s0, q, k, v, chunk)
        for _ in tqdm.trange(30000, disable=not benchmark): # 6370 it/s; use muon: 1300 it/s
            y_par = block_prefix_causal_linear_attention_parallel(s0, q, k, v, chunk)
        # for _ in tqdm.trange(300, disable=not benchmark): # not efficient, only works for fp32
        #     y_tri = block_prefix_causal_linear_attention_triton_recurrent(s0, q, k, v, chunk)
        # for _ in tqdm.trange(300, disable=not benchmark): # not efficient, only works for fp32
        #     y_tri = block_prefix_causal_linear_attention_triton_parallel(s0, q, k, v, chunk)

        if not benchmark:
            # tolerance: fp16 on GPU needs looser
            if dtype in (torch.float16, torch.bfloat16):
                atol, rtol = 5e-2, 5e-2
            else:
                atol, rtol = 1e-5, 1e-5
            torch.testing.assert_close(y_ref, y_par, atol=atol, rtol=rtol)

    print("Forward equivalence: PASS")


if __name__ == "__main__":
    test_equiv(benchmark=False)
    test_equiv(benchmark=True)
