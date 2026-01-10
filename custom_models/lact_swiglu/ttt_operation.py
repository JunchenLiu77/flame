import torch.nn.functional as F
import torch

@torch.compile()
def silu_backprop(dy: torch.Tensor, x: torch.Tensor):
    """
    Args:
        dy: [b, d, l], gradient of the outer loss wrt the y
        x: [b, d, l], input of the silu activation
    outs:
        dx: [b, d, l], gradient of the outer loss wrt the x
        dx = dy * sigma * (1 + x * (1 - sigma))
    """
    sigma = torch.sigmoid(x)
    dx = dy * sigma * (1 + x * (1 - sigma))
    return dx

@torch.compile
def silu_backprop_(x: torch.Tensor):
    """
    Similar to silu_backprop, but don't take the upstream gradient
    Args:
        x: [b, d, l], input of the silu activation
    outs:
        dx: [b, d, l], gradient of the outer loss wrt the x
        dx = sigma * (1 + x * (1 - sigma))
    """
    sigma = torch.sigmoid(x)
    dx = sigma * (1 + x * (1 - sigma))
    return dx


@torch.compile()
def l2_norm(x: torch.Tensor):
    """
    x: [b, l, d]
    """
    x_type = x.dtype
    ret = x / (x.norm(dim=-1, keepdim=True) + 1e-5)  # norm will upcast to float32
    return ret.type(x_type)


@torch.compile()
def zeropower_via_newtonschulz5(G):
    """
    This is an updated version of the zeropower_via_newtonschulz5 function in here:
    https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt_medium.py#L26
    The code is modified from https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py#L49, which contains the original muon implementation.
    Major change: G is [b, d, d] rather than [d, d]
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    Args:
        G: [b, d, d']
    Returns:
        X: [b, d, d']
    FLOPS:  When d=d', Total FLOPS=30 * b * d^3
    """
    assert len(G.shape) == 3
    X = G.bfloat16()
    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for a, b, c in [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]:
        A = X @ X.transpose(1, 2)
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    return X



@torch.compile()
def block_causal_lact_swiglu(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    chunk_size: int=2048,  # test-time training chunk size
    use_muon: bool = False,
    momentum: torch.Tensor = None, # [b, s, 1]
    loss_type: str = "dot_product",
):
    """
    Block causal LaCT with SwiGLU fast weight function.
        Apply then Update => Shifted Block Causal LaCT
    w0, w1, w2 are the fast weights. f(x) =  w1 @ (silu(w0 @ x) * (w2 @ x))
    
    About precision:
        w0, w1, w2 are mostly likely fp32. 
        q, k, v are fp16.
        lr0, lr1, lr2 are fp32.
        The forward, backward produce bf16 gradients, updated fast weights are fp32.
        The final output are bf16.
    
    FLOPS:
        (assume dk=dv denoted as D, hidden dimension of swiglu-mlp is H, ignore muon, ignore last chunk)
        Forward pass with key: 4 * D * H * L * B
        Backward pass: 8 * D * H * L * B
        Forward with Query: 6 * D * H * L * B
        Total: 18 * D * H * L * B
    Outputs:
        o: [b, l, dv]
    """
    
    # adding detach here sometimes improves stability.
    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)

    if momentum is not None:
        dw1_momentum = torch.zeros_like(w1)
        dw0_momentum = torch.zeros_like(w0)
        dw2_momentum = torch.zeros_like(w2)

    q = q.transpose(1, 2)  # [b, dk, l]
    v = v.transpose(1, 2)

    output = torch.zeros_like(v)

    e_index = 0
    seq_len = k.shape[1]
    for i in range(0, seq_len - chunk_size, chunk_size):
        s_index = i
        e_index = s_index + chunk_size

        # [b, l, dk]
        ki = k[:, s_index:e_index, :]  # bf16
        # [b, dv, l]
        vi = v[:, :, s_index:e_index]  # bf16
        # [b, dh, l]
        qi = q[:, :, s_index:e_index]
        # [b, l, d/1] fp32
        if lr1 is not None:
            lr1i = lr1[:, s_index:e_index, :]  # [b, l, d/1] fp32
        if lr2 is not None:
            lr2i = lr2[:, s_index:e_index, :]  # [b, l, d/1] fp32
        if lr0 is not None:
            lr0i = lr0[:, s_index:e_index, :]  # [b, l, d/1] fp32

        if loss_type == "only_w1_momentum_one":
            lr1i = 1.0

        if loss_type in ["design1", "design2"]:
            # apply: o = MLP(0.5 * q + 0.5 * k)
            mlp_input = 0.5 * qi + 0.5 * ki.transpose(1, 2)
            # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
            h = torch.bmm(w2, mlp_input)
            gate = F.silu(torch.bmm(w0, mlp_input), inplace=True)
            # [b, dv, dh] @ [b, dh, l] -> [b, dv, l] -> [b, l, dv]
            output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)
        elif loss_type in ["no_query_dot_product"]:
            # use previous w0 and w1 to get the final output
            # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
            h = torch.bmm(w2, ki.transpose(1, 2))
            gate = F.silu(torch.bmm(w0, ki.transpose(1, 2)), inplace=True)
            # [b, dv, dh] @ [b, dh, l] -> [b, dv, l] -> [b, l, dv]
            output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)
        else:
            # default case, follow the original design
            # use previous w0 and w1 to get the final output
            # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
            h = torch.bmm(w2, qi)
            gate = F.silu(torch.bmm(w0, qi), inplace=True)
            # [b, dv, dh] @ [b, dh, l] -> [b, dv, l] -> [b, l, dv]
            output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)

        if loss_type == "design1":
            # update: 0.5 * MLP(q) + 0.5 * MLP(k) -> v, dot product loss

            # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
            k_gate_before_act = torch.bmm(w0, ki.transpose(1, 2))
            k_hidden_before_mul = torch.bmm(w2, ki.transpose(1, 2))
            k_hidden = F.silu(k_gate_before_act, inplace=False) * k_hidden_before_mul
            # [b, dv, dh] @ [b, dh, l] -> [b, dv, l] -> [b, l, dv]
            k_vpi = torch.bmm(w1, k_hidden)
            
            # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
            q_gate_before_act = torch.bmm(w0, qi)
            q_hidden_before_mul = torch.bmm(w2, qi)
            q_hidden = F.silu(q_gate_before_act, inplace=False) * q_hidden_before_mul
            # [b, dv, dh] @ [b, dh, l] -> [b, dv, l] -> [b, l, dv]
            q_vpi = torch.bmm(w1, q_hidden)
            
            # vpi = 0.5 * q_vpi + 0.5 * k_vpi
            # dot product loss: -vpi * vi
            dvpi = -vi

            k_dhidden = torch.bmm(w1.transpose(1, 2), dvpi)
            k_dhidden_before_mul = k_dhidden * F.silu(k_gate_before_act, inplace=False)
            k_dgate = k_dhidden * k_hidden_before_mul
            k_dgate_before_act = silu_backprop(k_dgate, k_gate_before_act)

            q_dhidden = torch.bmm(w1.transpose(1, 2), dvpi)
            q_dhidden_before_mul = q_dhidden * F.silu(q_gate_before_act, inplace=False)
            q_dgate = q_dhidden * q_hidden_before_mul
            q_dgate_before_act = silu_backprop(q_dgate, q_gate_before_act)

            dw1 = torch.bmm(
                dvpi, (0.5 * (k_hidden.transpose(1, 2) * lr1i) + 0.5 * (q_hidden.transpose(1, 2) * lr1i)).type_as(dvpi)
            )  # [b, d, d]
            # [b, dh, l] @ [b, l, dk] -> [b, dh, dk]
            dw0 = 0.5 * torch.bmm(k_dgate_before_act, (ki * lr0i).type_as(k_dgate_before_act)) + 0.5 * torch.bmm(q_dgate_before_act, (qi.transpose(1, 2) * lr0i).type_as(q_dgate_before_act))
            dw2 = 0.5 * torch.bmm(k_dhidden_before_mul, (ki * lr2i).type_as(k_dhidden_before_mul)) + 0.5 * torch.bmm(q_dhidden_before_mul, (qi.transpose(1, 2) * lr2i).type_as(q_dhidden_before_mul))
        elif loss_type == "design2":
            # update: MLP(0.5 * q + 0.5 * k) -> v, dot product loss
            mlp_input = 0.5 * qi.transpose(1, 2) + 0.5 * ki
            # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
            gate_before_act = torch.bmm(w0, mlp_input.transpose(1, 2))
            hidden_before_mul = torch.bmm(w2, mlp_input.transpose(1, 2))
            hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul
            # [b, dv, dh] @ [b, dh, l] -> [b, dv, l]
            vpi = torch.bmm(w1, hidden)

            # update: MLP(k) -> v, dot product loss
            dvpi = -vi

            # [b, dh, dv] @ [b, dv, l] -> [b, dh, l]
            dhidden = torch.bmm(w1.transpose(1, 2), dvpi)
            dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
            dgate = dhidden * hidden_before_mul
            dgate_before_act = silu_backprop(dgate, gate_before_act)

            # [b, d_2, l] @ [b, l, d_1] -> [b, d_2, d_1]
            # in bmm two mat is fp32, but the result is bf16.
            # it's better to cast the mat to bf16 before bmm.
            # [b, dv, l] @ [b, l, dh] -> [b, dv, dh]
            # it's better to cast the mat to bf16 before bmm.
            dw1 = torch.bmm(
                dvpi, (hidden.transpose(1, 2) * lr1i).type_as(dvpi)
            )  # [b, d, d]
            # [b, dh, l] @ [b, l, dk] -> [b, dh, dk]
            dw0 = torch.bmm(dgate_before_act, (mlp_input * lr0i).type_as(dgate_before_act))
            dw2 = torch.bmm(dhidden_before_mul, (mlp_input * lr2i).type_as(dhidden_before_mul))
        elif loss_type in ["unroll1", "simplify8", "simplify9", "simplify10"]:
            # unroll the gradient calculation formula, assume using dot-product loss
            # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
            # gate_before_act = torch.bmm(w0, ki.transpose(1, 2))
            # hidden_before_mul = torch.bmm(w2, ki.transpose(1, 2))
            # hidden = (F.silu(torch.bmm(w0, ki.transpose(1, 2)), inplace=False) * torch.bmm(w2, ki.transpose(1, 2)))
            # [b, dv, dh] @ [b, dh, l] -> [b, dv, l]
            # vpi = torch.bmm(w1, (F.silu(torch.bmm(w0, ki.transpose(1, 2)), inplace=False) * torch.bmm(w2, ki.transpose(1, 2))))

            # update: MLP(k) -> v, loss type is dot-product
            # dvpi = -vi

            # [b, dh, dv] @ [b, dv, l] -> [b, dh, l]
            # dhidden = torch.bmm(w1.transpose(1, 2), -vi)
            # dhidden_before_mul = (torch.bmm(w1.transpose(1, 2), -vi) * F.silu(torch.bmm(w0, ki.transpose(1, 2)), inplace=False))
            # dgate = (torch.bmm(w1.transpose(1, 2), -vi) * torch.bmm(w2, ki.transpose(1, 2)))
            # dgate_before_act = silu_backprop((torch.bmm(w1.transpose(1, 2), -vi) * torch.bmm(w2, ki.transpose(1, 2))), torch.bmm(w0, ki.transpose(1, 2)))

            # [b, d_2, l] @ [b, l, d_1] -> [b, d_2, d_1]
            # in bmm two mat is fp32, but the result is bf16.
            # it's better to cast the mat to bf16 before bmm.
            # [b, dv, l] @ [b, l, dh] -> [b, dv, dh]
            # it's better to cast the mat to bf16 before bmm.
            dw1 = torch.bmm(
                -vi, ((F.silu(torch.bmm(w0, ki.transpose(1, 2)), inplace=False) * torch.bmm(w2, ki.transpose(1, 2))).transpose(1, 2) * lr1i).type_as(vi)
            )  # [b, d, d]
            # [b, dh, l] @ [b, l, dk] -> [b, dh, dk]
            dw0 = torch.bmm(silu_backprop((torch.bmm(w1.transpose(1, 2), -vi) * torch.bmm(w2, ki.transpose(1, 2))), torch.bmm(w0, ki.transpose(1, 2))), (ki * lr0i).type_as(vi))
            dw2 = torch.bmm((torch.bmm(w1.transpose(1, 2), -vi) * F.silu(torch.bmm(w0, ki.transpose(1, 2)), inplace=False)), (ki * lr2i).type_as(vi))
        elif loss_type == "simplify2":
            # remove activation function in dw1 and dw2
            dw1 = torch.bmm(
                -vi, ((torch.bmm(w0, ki.transpose(1, 2)) * torch.bmm(w2, ki.transpose(1, 2))).transpose(1, 2) * lr1i).type_as(vi)
            )  # [b, d, d]
            # [b, dh, l] @ [b, l, dk] -> [b, dh, dk]
            dw0 = torch.bmm(silu_backprop((torch.bmm(w1.transpose(1, 2), -vi) * torch.bmm(w2, ki.transpose(1, 2))), torch.bmm(w0, ki.transpose(1, 2))), (ki * lr0i).type_as(vi))
            dw2 = torch.bmm((torch.bmm(w1.transpose(1, 2), -vi) * torch.bmm(w0, ki.transpose(1, 2))), (ki * lr2i).type_as(vi))
        elif loss_type == "simplify5":
            # based on simplify2, replace silu_backprop with silu
            dw1 = torch.bmm(
                -vi, ((torch.bmm(w0, ki.transpose(1, 2)) * torch.bmm(w2, ki.transpose(1, 2))).transpose(1, 2) * lr1i).type_as(vi)
            )  # [b, d, d]
            # [b, dh, l] @ [b, l, dk] -> [b, dh, dk]
            dw0 = torch.bmm((torch.bmm(w1.transpose(1, 2), -vi) * torch.bmm(w2, ki.transpose(1, 2))) * F.silu(torch.bmm(w0, ki.transpose(1, 2)), inplace=False), (ki * lr0i).type_as(vi))
            dw2 = torch.bmm((torch.bmm(w1.transpose(1, 2), -vi) * torch.bmm(w0, ki.transpose(1, 2))), (ki * lr2i).type_as(vi))
        elif loss_type == "simplify6":
            # based on unroll1, remove the lr1
            dw1 = torch.bmm(
                -vi, ((F.silu(torch.bmm(w0, ki.transpose(1, 2)), inplace=False) * torch.bmm(w2, ki.transpose(1, 2))).transpose(1, 2)).type_as(vi)
            )  # [b, d, d]
            # [b, dh, l] @ [b, l, dk] -> [b, dh, dk]
            dw0 = torch.bmm(silu_backprop((torch.bmm(w1.transpose(1, 2), -vi) * torch.bmm(w2, ki.transpose(1, 2))), torch.bmm(w0, ki.transpose(1, 2))), (ki * lr0i).type_as(vi))
            dw2 = torch.bmm((torch.bmm(w1.transpose(1, 2), -vi) * F.silu(torch.bmm(w0, ki.transpose(1, 2)), inplace=False)), (ki * lr2i).type_as(vi))
        elif loss_type == "simplify7":
            # based on unroll1, remove all the lr
            dw1 = torch.bmm(
                -vi, ((F.silu(torch.bmm(w0, ki.transpose(1, 2)), inplace=False) * torch.bmm(w2, ki.transpose(1, 2))).transpose(1, 2)).type_as(vi)
            )  # [b, d, d]
            # [b, dh, l] @ [b, l, dk] -> [b, dh, dk]
            dw0 = torch.bmm(silu_backprop((torch.bmm(w1.transpose(1, 2), -vi) * torch.bmm(w2, ki.transpose(1, 2))), torch.bmm(w0, ki.transpose(1, 2))), (ki).type_as(vi))
            dw2 = torch.bmm((torch.bmm(w1.transpose(1, 2), -vi) * F.silu(torch.bmm(w0, ki.transpose(1, 2)), inplace=False)), (ki).type_as(vi))
        else:
            # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
            gate_before_act = torch.bmm(w0, ki.transpose(1, 2))
            hidden_before_mul = torch.bmm(w2, ki.transpose(1, 2))
            hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul
            # [b, dv, dh] @ [b, dh, l] -> [b, dv, l]
            vpi = torch.bmm(w1, hidden)

            # update: MLP(k) -> v, loss type can be arbitrary.
            if loss_type in ["dot_product", "no_query_dot_product", "ga_dot_product", "only_w1", "only_w1_momentum_one"]:
                dvpi = -vi
            elif loss_type == "vp**2":
                dvpi = 2*vpi
            elif loss_type == "mse":
                dvpi = vpi - vi
            else:
                raise ValueError(f"Invalid loss type: {loss_type}")

            # [b, dh, dv] @ [b, dv, l] -> [b, dh, l]
            dhidden = torch.bmm(w1.transpose(1, 2), dvpi)
            dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
            dgate = dhidden * hidden_before_mul
            dgate_before_act = silu_backprop(dgate, gate_before_act)

            # [b, d_2, l] @ [b, l, d_1] -> [b, d_2, d_1]
            # in bmm two mat is fp32, but the result is bf16.
            # it's better to cast the mat to bf16 before bmm.
            # [b, dv, l] @ [b, l, dh] -> [b, dv, dh]
            # it's better to cast the mat to bf16 before bmm.
            dw1 = torch.bmm(
                dvpi, (hidden.transpose(1, 2) * lr1i).type_as(dvpi)
            )  # [b, d, d]
            # [b, dh, l] @ [b, l, dk] -> [b, dh, dk]
            dw0 = torch.bmm(dgate_before_act, (ki * lr0i).type_as(dgate_before_act))
            dw2 = torch.bmm(dhidden_before_mul, (ki * lr2i).type_as(dhidden_before_mul))

            if "ga" in loss_type:
                dw1 = -dw1
                dw0 = -dw0
                dw2 = -dw2

        if momentum is not None:
            if loss_type in ["simplify8"]:
                # momentum coeff is set as 1 for simplify8
                m_i = momentum
            else:
                m_i = momentum[:, s_index:e_index, :]
                m_i = m_i.mean(dim=1, keepdim=True)
            
            if loss_type == "only_w1_momentum_one":
                m_i = 1.0

            dw0 = dw0 + dw0_momentum * m_i
            dw1 = dw1 + dw1_momentum * m_i
            dw2 = dw2 + dw2_momentum * m_i
            dw0_momentum = dw0
            dw1_momentum = dw1
            dw2_momentum = dw2


        if use_muon and loss_type not in ["simplify10"]:
            dw1 = zeropower_via_newtonschulz5(dw1)
            dw0 = zeropower_via_newtonschulz5(dw0)
            dw2 = zeropower_via_newtonschulz5(dw2)
            # legacy code for different global lr for muon. Conclusion: 1.0 is good
            # if muon_w0_lr is not None:
            #     # lr is fp32 (after softplus)
            #     # in future version, we can cast it before input. TODO
            #     dw1 = (dw1 * muon_w1_lr).type_as(w1)
            #     dw0 = (dw0 * muon_w0_lr).type_as(w0)
            #     dw2 = (dw2 * muon_w2_lr).type_as(w2)

        # w1 = w1 + dw1
        # w0 = w0 + dw0
        # w2 = w2 + dw2
        # print(f"using gradient ascent")
        w1 = w1 - dw1
        if loss_type not in ["only_w1", "only_w1_momentum_one"]:
            w0 = w0 - dw0
            w2 = w2 - dw2
    
        # Do channel-wise l2 norm.  conceptually like post-norm.
        if loss_type not in ["simplify9"]:
            # ablation the weight norm
            w1 = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
            if loss_type not in ["only_w1", "only_w1_momentum_one"]:
                # if we don't update w0 and w2, we don't need to apply weight norm on them.
                w0 = w0 / (w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
                w2 = w2 / (w2.norm(dim=2, keepdim=True) + 1e-5) * w2_norm
        
    # for the last chunk, don't update the fast weights, directly apply the fast weights to the query.
    s_index = e_index
    e_index = seq_len

    if loss_type in ["no_query_dot_product"]:
        # use k to update the fast weights
        ki = k[:, s_index:e_index, :].transpose(1, 2)
        h = torch.bmm(w2, ki)
        gate = F.silu(torch.bmm(w0, ki), inplace=True)
        output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)
    else:
        qi = q[:, :, s_index:e_index]
        # use the last w0 and w1 to get the final output
        # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
        h = torch.bmm(w2, qi)
        gate = F.silu(torch.bmm(w0, qi), inplace=True)
        # [b, dv, dh] @ [b, dh, l] -> [b, dv, l] -> [b, l, dv]
        output[:, :, s_index:e_index] = torch.bmm(w1, gate * h)

    return output.transpose(1, 2)


@torch.compile()
def prenorm_block_causal_lact_swiglu(
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    chunk_size: int=2048,  # test-time training chunk size
    use_muon: bool = False,
    momentum: torch.Tensor = None, # [b, s, 1]
):
    raise NotImplementedError("We only do exps on the non-prenorm version.")
    """
    Block causal LaCT with SwiGLU fast weight function.
        Apply then Update => Shifted Block Causal LaCT
    w0, w1, w2 are the fast weights. f(x) =  w1 @ (silu(w0 @ x) * (w2 @ x))
    
    About precision:
        w0, w1, w2 are mostly likely fp32. 
        q, k, v are fp16.
        lr0, lr1, lr2 are fp32.
        The forward, backward produce bf16 gradients, updated fast weights are fp32.
        The final output are bf16.
    
    FLOPS:
        (assume dk=dv denoted as D, hidden dimension of swiglu-mlp is H, ignore muon, ignore last chunk)
        Forward pass with key: 4 * D * H * L * B
        Backward pass: 8 * D * H * L * B
        Forward with Query: 6 * D * H * L * B
        Total: 18 * D * H * L * B
    Outputs:
        o: [b, l, dv]
    """
    
    # adding detach here sometimes improves stability.
    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)

    w0_main, w1_main, w2_main = w0, w1, w2

    if momentum is not None:
        dw1_momentum = torch.zeros_like(w1)
        dw0_momentum = torch.zeros_like(w0)
        dw2_momentum = torch.zeros_like(w2)

    q = q.transpose(1, 2)  # [b, dk, l]
    v = v.transpose(1, 2)

    output = torch.zeros_like(v)

    e_index = 0
    seq_len = k.shape[1]
    for i in range(0, seq_len - chunk_size, chunk_size):
        s_index = i
        e_index = s_index + chunk_size

        # [b, l, dk]
        ki = k[:, s_index:e_index, :]  # bf16
        # [b, dv, l]
        vi = v[:, :, s_index:e_index]  # bf16
        # [b, dh, l]
        qi = q[:, :, s_index:e_index]
        # [b, l, d/1] fp32
        lr1i = lr1[:, s_index:e_index, :]  # [b, l, d/1] fp32
        lr2i = lr2[:, s_index:e_index, :]  # [b, l, d/1] fp32
        lr0i = lr0[:, s_index:e_index, :]  # [b, l, d/1] fp32

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
        
        # [b, dh, dv] @ [b, dv, l] -> [b, dh, l]
        dhidden = torch.bmm(w1.transpose(1, 2), vi)

        dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)

        dgate = dhidden * hidden_before_mul
        dgate_before_act = silu_backprop(dgate, gate_before_act)

        # [b, d_2, l] @ [b, l, d_1] -> [b, d_2, d_1]
        # in bmm two mat is fp32, but the result is bf16.
        # it's better to cast the mat to bf16 before bmm.
        # [b, dv, l] @ [b, l, dh] -> [b, dv, dh]
        # it's better to cast the mat to bf16 before bmm.
        dw1 = torch.bmm(
            vi, (hidden.transpose(1, 2) * lr1i).type_as(vi)
        )  # [b, d, d]
        # [b, dh, l] @ [b, l, dk] -> [b, dh, dk]
        dw0 = torch.bmm(dgate_before_act, (ki * lr0i).type_as(dgate_before_act))
        dw2 = torch.bmm(dhidden_before_mul, (ki * lr2i).type_as(dhidden_before_mul))

        if momentum is not None:
            m_i = momentum[:, s_index:e_index, :] 
            m_i = m_i.mean(dim=1, keepdim=True)

            dw0 = dw0 + dw0_momentum * m_i
            dw1 = dw1 + dw1_momentum * m_i
            dw2 = dw2 + dw2_momentum * m_i
            dw0_momentum = dw0
            dw1_momentum = dw1
            dw2_momentum = dw2


        if use_muon:
            dw1 = zeropower_via_newtonschulz5(dw1)
            dw0 = zeropower_via_newtonschulz5(dw0)
            dw2 = zeropower_via_newtonschulz5(dw2)
            # legacy code for different global lr for muon. Conclusion: 1.0 is good
            # if muon_w0_lr is not None:
            #     # lr is fp32 (after softplus)
            #     # in future version, we can cast it before input. TODO
            #     dw1 = (dw1 * muon_w1_lr).type_as(w1)
            #     dw0 = (dw0 * muon_w0_lr).type_as(w0)
            #     dw2 = (dw2 * muon_w2_lr).type_as(w2)

        w1_main = w1_main + dw1
        w0_main = w0_main + dw0
        w2_main = w2_main + dw2
    
        # Do channel-wise l2 norm.  conceptually like post-norm.
        w0 = w0_main / (w0_main.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
        w1 = w1_main / (w1_main.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
        w2 = w2_main / (w2_main.norm(dim=2, keepdim=True) + 1e-5) * w2_norm
        
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