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
    lr1: torch.Tensor, # [b, l, 1] fp32
    chunk_size: int = 2048,
    use_muon: bool = True,
    momentum: torch.Tensor = None,  # [b, l, 1]
) -> torch.Tensor: # [b, l, dv]
    """
    block i attends to all previous blocks (< i) + initial state

    block 0 attends to only initial state.

    no within-block attention
    """

    if momentum is not None:
        dw1_momentum = torch.zeros_like(w1)

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
        # [b, l, d/1] fp32
        lr1i = lr1[:, s_index:e_index, :]  # [b, l, d/1] fp32

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
        dw1 = torch.bmm(vi, (hidden.transpose(1, 2) * lr1i).type_as(vi))  # [b, d, d]

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
    lr1: torch.Tensor, # [b, l, d/1] fp32
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
    n_blocks = (l + chunk_size - 1) // chunk_size
    l_pad = n_blocks * chunk_size
    pad = l_pad - l

    # 1. Concatenate weights for single projection launch
    # [b, 2*dh, dk]
    w02 = torch.cat([w0, w2], dim=1)
    
    # 2. Prepare inputs (Flatten B and N)
    q_pad = F.pad(q, (0, 0, 0, pad)) if pad > 0 else q
    k_pad = F.pad(k, (0, 0, 0, pad)) if pad > 0 else k
    v_pad = F.pad(v, (0, 0, 0, pad)) if pad > 0 else v
    
    q_flat = q_pad.view(b * n_blocks, chunk_size, dk)
    k_flat = k_pad.view(b * n_blocks, chunk_size, dk)
    v_flat = v_pad.view(b * n_blocks, chunk_size, dv)

    # 3. Single BMM for Gate & Hidden (instead of two)
    # [b*n, dk, c] -> [b*n, 2*dh, c]
    w02_exp = w02.repeat_interleave(n_blocks, dim=0)
    z_all = torch.bmm(w02_exp, k_flat.transpose(1, 2))
    
    # Slice gate and hidden
    gate_k, hidden_k = z_all.chunk(2, dim=1)
    # Fused SiLU and Mul
    gated_k = F.silu(gate_k) * hidden_k # [b*n, dh, c]

    # 4. Compute dw1 using BMM (Replaces einsum)
    # Pre-multiply v by lr1 to simplify outer product
    lr1_pad = F.pad(lr1, (0, 0, 0, pad)) if pad > 0 else lr1
    v_scaled = v_flat * lr1_pad.view(b * n_blocks, chunk_size, 1)
    
    # [b*n, dv, c] @ [b*n, c, dh] -> [b*n, dv, dh]
    dw1 = torch.bmm(v_scaled.transpose(1, 2), gated_k.transpose(1, 2))
    dw1 = dw1.view(b, n_blocks, dv, dh)

    # 5. Momentum Recurrence
    if momentum is not None:
        m_pad = F.pad(momentum, (0, 0, 0, pad)) if pad > 0 else momentum
        m_blk = m_pad.view(b, n_blocks, chunk_size).mean(dim=2, keepdim=True)
        # Parallel scan via cumprod/cumsum
        m_prod = m_blk.cumprod(dim=1).unsqueeze(-1)
        dw1 = m_prod * torch.cumsum(dw1 / (m_prod + 1e-8), dim=1)

    if use_muon:
        # Use single flattened call for Muon
        dw1 = zeropower_via_newtonschulz5(dw1.reshape(-1, dv, dh)).reshape_as(dw1)

    # 6. Prefix-sum for w1
    prefix_exclusive = torch.cumsum(dw1, dim=1) - dw1
    w1_before = prefix_exclusive + w1.unsqueeze(1)

    # 7. Final Output Projection
    z_q = torch.bmm(w02_exp, q_flat.transpose(1, 2))
    gate_q, hidden_q = z_q.chunk(2, dim=1)
    gated_q = F.silu(gate_q) * hidden_q
    
    # [b*n, dv, dh] @ [b*n, dh, c] -> [b*n, dv, c]
    out = torch.bmm(w1_before.view(b * n_blocks, dv, dh), gated_q)
    
    out = out.transpose(1, 2).reshape(b, l_pad, dv)
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
        lr1 = torch.rand(b, l, 1, device=device, dtype=dtype)
        momentum = torch.rand(b, l, 1, device=device, dtype=dtype)

        for _ in tqdm.trange(3000, disable=not benchmark): # 480 it/s; use moun: 147 it/s
            y_ref = block_prefix_causal_linear_attention_recurrent(w0, w1, w2, q, k, v, lr1, chunk, use_muon, momentum)
        for _ in tqdm.trange(30000, disable=not benchmark): # 5285 it/s; use muon: 1891 it/s
            y_par = block_prefix_causal_linear_attention_parallel(w0, w1, w2, q, k, v, lr1, chunk, use_muon, momentum)
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
    test_equiv(benchmark=False)
    test_equiv(benchmark=True)
