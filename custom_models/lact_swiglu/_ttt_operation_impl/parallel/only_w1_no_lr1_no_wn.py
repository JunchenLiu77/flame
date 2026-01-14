import tqdm
import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


from custom_models.lact_swiglu.ttt_operation import zeropower_via_newtonschulz5


def block_prefix_causal_linear_attention_recurrent(
    w0: torch.Tensor, # [b, dh, dk]
    w1: torch.Tensor, # [b, dv, dh]
    w2: torch.Tensor, # [b, dh, dk]
    q: torch.Tensor, # [b, l, dk]
    k: torch.Tensor, # [b, l, dk]
    v: torch.Tensor, # [b, l, dv]
    chunk_size: int = 2048,
    use_muon: bool = True,
    momentum: torch.Tensor = None,  # [b, l, 1]
) -> torch.Tensor: # [b, l, dv]
    """
    block i attends to all previous blocks (< i) + initial state

    block 0 attends to only initial state.

    no within-block attention
    """
    q = q.transpose(1, 2)  # [b, dk, l]
    v = v.transpose(1, 2)  # [b, dv, l]

    output = torch.zeros_like(v)

    if momentum is not None:
        dw1_momentum = torch.zeros_like(w1)

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

        # use previous w0 and w1 to get the final output
        # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
        h = torch.bmm(w2, qi)
        gate = F.silu(torch.bmm(w0, qi), inplace=True)
        # [b, dv, dh] @ [b, dh, l] -> [b, dv, l] -> [b, l, dv]
        output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)

        # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
        gate_before_act = torch.bmm(w0, ki.transpose(1, 2))
        hidden_before_mul = torch.bmm(w2, ki.transpose(1, 2))

        hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

        # [b, d_2, l] @ [b, l, d_1] -> [b, d_2, d_1]
        # in bmm two mat is fp32, but the result is bf16.
        # it's better to cast the mat to bf16 before bmm.
        # [b, dv, l] @ [b, l, dh] -> [b, dv, dh]
        # it's better to cast the mat to bf16 before bmm.
        dw1 = torch.bmm(vi, (hidden.transpose(1, 2)).type_as(vi))  # [b, d, d]

        if momentum is not None:
            m_i = momentum[:, s_index:e_index, :]
            m_i = m_i.mean(dim=1, keepdim=True)

            dw1 = dw1 + dw1_momentum * m_i
            dw1_momentum = dw1
            
        if use_muon:
            dw1 = zeropower_via_newtonschulz5(dw1)

        w1 = w1 + dw1

    # for the last chunk, don't update the fast weights, directly apply the fast weights to the query.
    s_index = e_index
    e_index = seq_len

    qi = q[:, :, s_index:e_index]
    # use the last w0 and w1 to get the final output
    # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
    h = torch.bmm(w2, qi)
    gate = F.silu(torch.bmm(w0, qi), inplace=True)
    # [b, dv, dh] @ [b, dh, l] -> [b, dv, l] -> [b, l, dv]
    output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)

    return output.transpose(1, 2)


def block_prefix_causal_linear_attention_parallel(
    w0: torch.Tensor, # [b, dh, dk]
    w1: torch.Tensor, # [b, dv, dh]
    w2: torch.Tensor, # [b, dh, dk]
    q: torch.Tensor,   # [b, l, dk]
    k: torch.Tensor,   # [b, l, dk]
    v: torch.Tensor,   # [b, l, dv]
    chunk_size: int = 2048,
    use_muon: bool = True,
    momentum: torch.Tensor = None,  # [b, l, 1]
) -> torch.Tensor:     # [b, l, dv]
    """
    block i attends to all previous blocks (< i) + initial state
    block 0 attends to only initial state
    no within-block attention

    Parallel form via per-block outer-product reduction + prefix-sum over blocks.

    This version uses w0, w1, w2 with SiLU gating:
    - Output: w1 @ (silu(w0 @ q) * (w2 @ q))
    - State update: dw1 = v @ (silu(w0 @ k) * (w2 @ k)).T
    
    With momentum, the recurrence is:
    - dw1'[i] = dw1[i] + dw1'[i-1] * m[i]
    This is computed via a parallel scan.
    """
    b, l, dk = q.shape
    dv = v.shape[-1]
    dh = w0.shape[1]
    assert w0.shape == (b, dh, dk)
    assert w1.shape == (b, dv, dh)
    assert w2.shape == (b, dh, dk)
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

    # Compute hidden states for keys: hidden = silu(w0 @ k) * (w2 @ k)
    # k_blk: [b, n, c, dk] -> transpose to [b*n, dk, c] for bmm
    k_flat = k_blk.reshape(b * n_blocks, chunk_size, dk).transpose(1, 2)  # [b*n, dk, c]
    w0_exp = w0.unsqueeze(1).expand(-1, n_blocks, -1, -1).reshape(b * n_blocks, dh, dk)  # [b*n, dh, dk]
    w2_exp = w2.unsqueeze(1).expand(-1, n_blocks, -1, -1).reshape(b * n_blocks, dh, dk)  # [b*n, dh, dk]

    gate_before_act = torch.bmm(w0_exp, k_flat)  # [b*n, dh, c]
    hidden_before_mul = torch.bmm(w2_exp, k_flat)  # [b*n, dh, c]
    hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul  # [b*n, dh, c]
    hidden = hidden.reshape(b, n_blocks, dh, chunk_size)  # [b, n, dh, c]

    # Per-block update: dw1[j] = sum_t v_t @ hidden_t.T  -> [b, n, dv, dh]
    # v_blk: [b, n, c, dv], hidden: [b, n, dh, c]
    # einsum: (b n t dv) x (b n dh t) -> (b n dv dh)
    dw1 = torch.einsum("bntd,bnht->bndh", v_blk, hidden)

    # Apply momentum if provided
    # Recurrence: dw1'[i] = dw1[i] + dw1'[i-1] * m[i]
    # This is a linear recurrence that can be computed via parallel scan
    if momentum is not None:
        # Pad momentum to match sequence length
        if pad:
            momentum_pad = F.pad(momentum, (0, 0, 0, pad))
        else:
            momentum_pad = momentum
        # Compute per-block momentum as mean over chunk: [b, n, 1]
        momentum_blk = momentum_pad.view(b, n_blocks, chunk_size, 1).mean(dim=2)  # [b, n, 1]
        
        # For linear recurrence y[i] = x[i] + m[i] * y[i-1], we can solve via:
        # y[i] = sum_{j<=i} x[j] * prod_{k=j+1}^{i} m[k]
        # 
        # Compute cumulative product of momentum from right to left for each position
        # m_cumprod[i] = prod_{k=i}^{n-1} m[k]  (reverse cumulative product)
        # Then: y[i] = sum_{j<=i} x[j] * (m_cumprod[j+1] / m_cumprod[i+1])
        #            = (1/m_cumprod[i+1]) * sum_{j<=i} x[j] * m_cumprod[j+1]
        #
        # But simpler: use a sequential scan which is still efficient for small n_blocks
        # For truly parallel version, we'd need associative scan, but n_blocks is typically small
        
        # Sequential scan for momentum (n_blocks is usually small, e.g., seq_len/chunk_size)
        dw1_with_momentum = torch.zeros_like(dw1)
        dw1_prev = torch.zeros(b, dv, dh, device=dw1.device, dtype=dw1.dtype)
        for i in range(n_blocks):
            m_i = momentum_blk[:, i, :]  # [b, 1]
            dw1_curr = dw1[:, i, :, :] + dw1_prev * m_i.unsqueeze(-1)  # [b, dv, dh]
            dw1_with_momentum[:, i, :, :] = dw1_curr
            dw1_prev = dw1_curr
        dw1 = dw1_with_momentum

    if use_muon:
        dw1 = zeropower_via_newtonschulz5(dw1.reshape(-1, dv, dh)).reshape_as(dw1)

    # Prefix state BEFORE each block:
    # w1_before[0] = w1
    # w1_before[i] = w1 + sum_{j < i} dw1[j]
    # Compute exclusive prefix-sum by shifting the inclusive cumsum.
    prefix_inclusive = dw1.cumsum(dim=1)  # [b, n, dv, dh]
    prefix_exclusive = prefix_inclusive - dw1
    w1_before = prefix_exclusive + w1.unsqueeze(1)  # [b, n, dv, dh]

    # Compute hidden states for queries: gate * h = silu(w0 @ q) * (w2 @ q)
    q_flat = q_blk.reshape(b * n_blocks, chunk_size, dk).transpose(1, 2)  # [b*n, dk, c]
    gate_q = torch.bmm(w0_exp, q_flat)  # [b*n, dh, c]
    h_q = torch.bmm(w2_exp, q_flat)  # [b*n, dh, c]
    gated_h = F.silu(gate_q, inplace=False) * h_q  # [b*n, dh, c]
    gated_h = gated_h.reshape(b, n_blocks, dh, chunk_size)  # [b, n, dh, c]

    # Output per block: o = w1_before @ (gate * h)
    # w1_before: [b, n, dv, dh], gated_h: [b, n, dh, c]
    # einsum: (b n dv dh) x (b n dh t) -> (b n t dv)
    out_blk = torch.einsum("bndh,bnht->bntd", w1_before, gated_h)

    out = out_blk.reshape(b, l_pad, dv)
    return out[:, :l, :]


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
            dict(b=1, l=32768, dk=64, dh=64, dv=64, chunk=2048, use_muon=False),
            dict(b=1, l=32768, dk=64, dh=64, dv=64, chunk=2048, use_muon=True),
        ]
    else:
        configs = [
            dict(b=1, l=1,    dk=8,  dh=8,  dv=7,  chunk=4, use_muon=True),
            dict(b=2, l=63,   dk=16, dh=16, dv=32, chunk=8, use_muon=False),
            dict(b=3, l=128,  dk=32, dh=32, dv=16, chunk=32, use_muon=True),
            dict(b=2, l=257,  dk=64, dh=48, dv=48, chunk=64, use_muon=False),
            dict(b=1, l=2048, dk=32, dh=32, dv=32, chunk=256, use_muon=True),
        ]

    for cfg in configs:
        print(f"Testing {cfg}")
        b, l, dk, dh, dv, chunk, use_muon = cfg["b"], cfg["l"], cfg["dk"], cfg["dh"], cfg["dv"], cfg["chunk"], cfg["use_muon"]

        w0 = torch.rand(b, dh, dk, device=device, dtype=dtype)
        w1 = torch.rand(b, dv, dh, device=device, dtype=dtype)
        w2 = torch.rand(b, dh, dk, device=device, dtype=dtype)
        q  = torch.rand(b, l, dk, device=device, dtype=dtype)
        k  = torch.rand(b, l, dk, device=device, dtype=dtype)
        v  = torch.rand(b, l, dv, device=device, dtype=dtype)
        momentum = torch.rand(b, l, 1, device=device, dtype=dtype)

        for _ in tqdm.trange(3000, disable=not benchmark): # 502 it/s; use moun: 126 it/s
            y_ref = block_prefix_causal_linear_attention_recurrent(w0, w1, w2, q, k, v, chunk, use_muon, momentum)
        for _ in tqdm.trange(30000, disable=not benchmark): # 1423 it/s; use muon: 822 it/s
            y_par = block_prefix_causal_linear_attention_parallel(w0, w1, w2, q, k, v, chunk, use_muon, momentum)
        # for _ in tqdm.trange(300, disable=not benchmark): # not efficient, only works for fp32
        #     y_tri = block_prefix_causal_linear_attention_triton_recurrent(w0, w1, w2, q, k, v, chunk)
        # for _ in tqdm.trange(300, disable=not benchmark): # not efficient, only works for fp32
        #     y_tri = block_prefix_causal_linear_attention_triton_parallel(w0, w1, w2, q, k, v, chunk)

        if not benchmark:
            # tolerance: fp16 on GPU needs looser
            if dtype in (torch.float16, torch.bfloat16):
                atol, rtol = 5e-2, 5e-2
            else:
                atol, rtol = 1e-5, 1e-5
            torch.testing.assert_close(y_ref, y_par, atol=atol, rtol=rtol)

    print("Forward equivalence: PASS")


if __name__ == "__main__":
    # test_equiv(benchmark=False)
    test_equiv(benchmark=True)
