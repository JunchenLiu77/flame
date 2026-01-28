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
    lr1: torch.Tensor, # [b, l, 1] fp32
    chunk_size: int = 2048,
    use_muon: bool = True,
    momentum: torch.Tensor = None,  # [b, l, 1]
) -> torch.Tensor: # [b, l, dv]
    """
    block i attends to all previous blocks (< i) + initial state

    block 0 attends to only initial state.

    no within-block attention
    
    Straight qk version with lr1 and momentum:
    - Output: state @ q
    - State update: dstate = v @ (k * lr1).T
    """

    if momentum is not None:
        dstate_momentum = torch.zeros_like(state)

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

        # get the final output
        # [b, dv, dk] @ [b, dk, l] -> [b, dv, l] -> [b, l, dv]
        output[:, :, s_index:e_index] = torch.bmm(state, qi)

        # [b, dv, l] @ [b, l, dk] -> [b, dv, dk]
        dstate = torch.bmm(vi, (ki * lr1i).type_as(vi))

        if momentum is not None:
            m_i = momentum[:, s_index:e_index, :]
            m_i = m_i.mean(dim=1, keepdim=True)

            dstate = dstate + dstate_momentum * m_i
            dstate_momentum = dstate
            
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

    Straight qk version with lr1 and momentum:
    - Output: state @ q
    - State update: dS = v @ (k * lr1).T
    
    With momentum, the recurrence is:
    - dS'[i] = dS[i] + dS'[i-1] * m[i]
    This is computed via a parallel scan.
    """
    b, l, dk = q.shape
    dv = v.shape[-1]
    n_blocks = (l + chunk_size - 1) // chunk_size
    l_pad = n_blocks * chunk_size
    pad = l_pad - l

    # 1. Prepare inputs (Flatten B and N)
    q_pad = F.pad(q, (0, 0, 0, pad)) if pad > 0 else q
    k_pad = F.pad(k, (0, 0, 0, pad)) if pad > 0 else k
    v_pad = F.pad(v, (0, 0, 0, pad)) if pad > 0 else v
    
    q_flat = q_pad.view(b * n_blocks, chunk_size, dk)
    k_flat = k_pad.view(b * n_blocks, chunk_size, dk)
    v_flat = v_pad.view(b * n_blocks, chunk_size, dv)

    # 2. Compute dS using BMM
    # Per-block update: dS[j] = sum_t v_t @ (k_t * lr1_t).T  -> [b*n, dv, dk]
    lr1_pad = F.pad(lr1, (0, 0, 0, pad)) if pad > 0 else lr1
    v_scaled = v_flat * lr1_pad.view(b * n_blocks, chunk_size, 1)
    
    # [b*n, dv, c] @ [b*n, c, dk] -> [b*n, dv, dk]
    dS = torch.bmm(v_scaled.transpose(1, 2), k_flat)
    dS = dS.view(b, n_blocks, dv, dk)

    # 3. Momentum Recurrence
    if momentum is not None:
        m_pad = F.pad(momentum, (0, 0, 0, pad)) if pad > 0 else momentum
        m_blk = m_pad.view(b, n_blocks, chunk_size).mean(dim=2, keepdim=True)
        # Parallel scan via cumprod/cumsum
        m_prod = m_blk.cumprod(dim=1).unsqueeze(-1)
        dS = m_prod * torch.cumsum(dS / (m_prod + 1e-8), dim=1)

    if use_muon:
        dS = zeropower_via_newtonschulz5(dS.reshape(-1, dv, dk)).reshape_as(dS)

    # 4. Prefix-sum for state
    # S_before[0] = s0
    # S_before[i] = s0 + sum_{j < i} dS[j]
    prefix_exclusive = torch.cumsum(dS, dim=1) - dS
    S_before = prefix_exclusive + s0.unsqueeze(1)  # [b, n, dv, dk]

    # 5. Final Output using BMM
    # Output per block: o = S_before @ q
    # [b*n, dv, dk] @ [b*n, dk, c] -> [b*n, dv, c]
    out = torch.bmm(S_before.view(b * n_blocks, dv, dk), q_flat.transpose(1, 2))
    
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
            dict(b=1, l=32768, dk=64, dv=64, chunk=2048, use_muon=False),
            dict(b=1, l=32768, dk=64, dv=64, chunk=2048, use_muon=True),
        ]
    else:
        configs = [
            dict(b=1, l=1,    dk=8,  dv=7,  chunk=4, use_muon=True),
            dict(b=2, l=63,   dk=16, dv=32, chunk=8, use_muon=False),
            dict(b=3, l=128,  dk=32, dv=16, chunk=32, use_muon=True),
            dict(b=2, l=257,  dk=64, dv=48, chunk=64, use_muon=False),
            dict(b=1, l=2048, dk=32, dv=32, chunk=256, use_muon=True),
        ]

    for cfg in configs:
        print(f"Testing {cfg}")
        b, l, dk, dv, chunk, use_muon = cfg["b"], cfg["l"], cfg["dk"], cfg["dv"], cfg["chunk"], cfg["use_muon"]

        s0 = torch.rand(b, dv, dk, device=device, dtype=dtype)
        q  = torch.rand(b, l, dk, device=device, dtype=dtype)
        k  = torch.rand(b, l, dk, device=device, dtype=dtype)
        v  = torch.rand(b, l, dv, device=device, dtype=dtype)
        lr1 = torch.rand(b, l, 1, device=device, dtype=dtype)
        momentum = torch.rand(b, l, 1, device=device, dtype=dtype)

        for _ in tqdm.trange(3000, disable=not benchmark):
            y_ref = block_prefix_causal_linear_attention_recurrent(s0, q, k, v, lr1, chunk, use_muon, momentum)
        for _ in tqdm.trange(30000, disable=not benchmark):
            y_par = block_prefix_causal_linear_attention_parallel(s0, q, k, v, lr1, chunk, use_muon, momentum)

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
