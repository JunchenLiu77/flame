import torch

from custom_models.lact_swiglu.ttt_operation import block_causal_lact_swiglu
from custom_models.lact_swiglu._ttt_operation_impl import block_causal_lact_swiglu as _block_causal_lact_swiglu


def create_test_data_dict():
    device = "cuda:0"
    dtype = torch.bfloat16
    
    b = 2
    l = 32
    dk = 16
    dh = 128
    dv = 48

    w0 = torch.randn(b, dh, dk, device=device, dtype=dtype)
    w1 = torch.randn(b, dv, dh, device=device, dtype=dtype)
    w2 = torch.randn(b, dh, dk, device=device, dtype=dtype)
    k = torch.randn(b, l, dk, device=device, dtype=dtype)
    v = torch.randn(b, l, dv, device=device, dtype=dtype)
    q = torch.randn(b, l, dk, device=device, dtype=dtype)
    lr0 = torch.randn(b, l, 1, device=device, dtype=torch.float32)
    lr1 = torch.randn(b, l, 1, device=device, dtype=torch.float32)
    lr2 = torch.randn(b, l, 1, device=device, dtype=torch.float32)
    chunk_size = 8
    use_muon = False
    momentum = torch.randn(b, l, 1, device=device, dtype=torch.float32)

    data_dict = {
        "w0": w0,
        "w1": w1,
        "w2": w2,
        "q": q,
        "k": k,
        "v": v,
        "lr0": lr0,
        "lr1": lr1,
        "lr2": lr2,
        "chunk_size": chunk_size,
        "use_muon": use_muon,
        "momentum": momentum,
        "loss_type": "dot_product"
    }
    return data_dict


if __name__ == "__main__":
    data_dict = create_test_data_dict()

    for loss_type in [
        "dot_product", 
        "ga_dot_product", 
        "mse", 
        "no_query_dot_product", 
        "only_w1"
    ]:
        data_dict["loss_type"] = loss_type
        o = block_causal_lact_swiglu(**data_dict)
        _o = _block_causal_lact_swiglu(**data_dict)
        torch.testing.assert_close(o, _o)
