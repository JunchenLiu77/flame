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

        for _ in tqdm.trange(3000, disable=not benchmark): # 1381 it/s; use moun: 193 it/s
            y_ref = block_prefix_causal_linear_attention_recurrent(s0, q, k, v, chunk, use_muon)
        for _ in tqdm.trange(30000, disable=not benchmark): # 11305 it/s; use muon: 2568 it/s
            y_par = block_prefix_causal_linear_attention_parallel(s0, q, k, v, chunk, use_muon)
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
