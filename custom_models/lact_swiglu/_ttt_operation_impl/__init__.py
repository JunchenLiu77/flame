import torch

from .original import block_causal_lact_swiglu as _impl_original
from .mse import block_causal_lact_swiglu as _impl_mse
from .no_query import block_causal_lact_swiglu as _impl_no_query
from .only_w1 import block_causal_lact_swiglu as _impl_only_w1
from .ga import block_causal_lact_swiglu as _impl_ga
from .only_w1_momentum_one import block_causal_lact_swiglu as _impl_only_w1_momentum_one
from .only_w1_momentum_one_no_norm import block_causal_lact_swiglu as _impl_only_w1_momentum_one_no_norm
from .only_w1_straight_qk import block_causal_lact_swiglu as _impl_only_w1_straight_qk


def block_causal_lact_swiglu(
    *args,
    **kwargs,
):
    assert "loss_type" in kwargs, "loss_type is required"
    loss_type = kwargs.pop("loss_type")
    
    if loss_type == "dot_product":
        return _impl_original(*args, **kwargs)
    elif loss_type == "mse":
        return _impl_mse(*args, **kwargs)
    elif loss_type == "no_query_dot_product":
        return _impl_no_query(*args, **kwargs)
    elif loss_type == "only_w1":
        return _impl_only_w1(*args, **kwargs)
    elif loss_type == "ga_dot_product":
        return _impl_ga(*args, **kwargs)
    elif loss_type == "only_w1_momentum_one":
        return _impl_only_w1_momentum_one(*args, **kwargs)
    elif loss_type == "only_w1_momentum_one_no_norm":
        return _impl_only_w1_momentum_one_no_norm(*args, **kwargs)
    elif loss_type == "only_w1_straight_qk":
        return _impl_only_w1_straight_qk(*args, **kwargs)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")