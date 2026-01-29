"""
Benchmark script for parallel TTT operations.
Calculates and reports tokens/sec for all four parallel implementation files.

Usage: python benchmark_parallel_ops.py
"""
import sys
import time
import torch
import torch.nn.functional as F
import importlib.util

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load modules
original = load_module("custom_models/lact_swiglu/_ttt_operation_impl/original.py", "original")
only_w1 = load_module("custom_models/lact_swiglu/_ttt_operation_impl/only_w1.py", "only_w1")
only_w1_no_wn = load_module("custom_models/lact_swiglu/_ttt_operation_impl/parallel/only_w1_no_wn.py", "only_w1_no_wn")
only_w1_no_lr1_no_wn = load_module("custom_models/lact_swiglu/_ttt_operation_impl/parallel/only_w1_no_lr1_no_wn.py", "only_w1_no_lr1_no_wn")
only_w1_no_lr1_no_wn_momen = load_module("custom_models/lact_swiglu/_ttt_operation_impl/parallel/only_w1_no_lr1_no_wn_momen.py", "only_w1_no_lr1_no_wn_momen")
only_w1_straight_qk_no_lr1_no_wn_momen = load_module("custom_models/lact_swiglu/_ttt_operation_impl/parallel/only_w1_straight_qk_no_lr1_no_wn_momen.py", "only_w1_straight_qk_no_lr1_no_wn_momen")
only_w1_straight_qk_no_wn = load_module("custom_models/lact_swiglu/_ttt_operation_impl/parallel/only_w1_straight_qk_no_wn.py", "only_w1_straight_qk_no_wn")
only_w1_straight_qk_no_lr1_no_wn = load_module("custom_models/lact_swiglu/_ttt_operation_impl/parallel/only_w1_straight_qk_no_lr1_no_wn.py", "only_w1_straight_qk_no_lr1_no_wn")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

print(f"Device: {device}")
print(f"Dtype: {dtype}")
print()

# Benchmark config
b, l, dk, dh, dv, chunk = 4, 32768, 384, 384, 384, 2048
n_warmup = 10
n_iters = 1000

torch.manual_seed(44)

@torch.no_grad()
def benchmark_original(use_muon):
    """Benchmarks original.py (baseline, updates all w0/w1/w2, has weight norm) - recurrent only"""
    w0 = torch.rand(b, dh, dk, device=device, dtype=dtype)
    w1 = torch.rand(b, dv, dh, device=device, dtype=dtype)
    w2 = torch.rand(b, dh, dk, device=device, dtype=dtype)
    q = torch.rand(b, l, dk, device=device, dtype=dtype)
    k = torch.rand(b, l, dk, device=device, dtype=dtype)
    v = torch.rand(b, l, dv, device=device, dtype=dtype)
    lr0 = torch.rand(b, l, 1, device=device, dtype=dtype)
    lr1 = torch.rand(b, l, 1, device=device, dtype=dtype)
    lr2 = torch.rand(b, l, 1, device=device, dtype=dtype)
    momentum = torch.rand(b, l, 1, device=device, dtype=dtype)
    
    fn = original.block_causal_lact_swiglu
    
    # Warmup (more warmup for torch.compile)
    for _ in range(min(n_warmup, 20)):
        fn(w0, w1, w2, q, k, v, lr0, lr1, lr2, chunk, use_muon, momentum)
    torch.cuda.synchronize()
    
    # Benchmark recurrent
    n_rec = n_iters // 10
    start = time.perf_counter()
    for _ in range(n_rec):
        fn(w0, w1, w2, q, k, v, lr0, lr1, lr2, chunk, use_muon, momentum)
    torch.cuda.synchronize()
    elapsed_rec = time.perf_counter() - start
    
    tokens_per_sec_rec = (n_rec * b * l) / elapsed_rec
    
    return tokens_per_sec_rec

@torch.no_grad()
def benchmark_only_w1(use_muon):
    """Benchmarks only_w1.py (variant 1, only updates w1, has weight norm) - recurrent only"""
    w0 = torch.rand(b, dh, dk, device=device, dtype=dtype)
    w1 = torch.rand(b, dv, dh, device=device, dtype=dtype)
    w2 = torch.rand(b, dh, dk, device=device, dtype=dtype)
    q = torch.rand(b, l, dk, device=device, dtype=dtype)
    k = torch.rand(b, l, dk, device=device, dtype=dtype)
    v = torch.rand(b, l, dv, device=device, dtype=dtype)
    lr0 = torch.rand(b, l, 1, device=device, dtype=dtype)  # unused but required by API
    lr1 = torch.rand(b, l, 1, device=device, dtype=dtype)
    lr2 = torch.rand(b, l, 1, device=device, dtype=dtype)  # unused but required by API
    momentum = torch.rand(b, l, 1, device=device, dtype=dtype)
    
    fn = only_w1.block_causal_lact_swiglu
    
    # Warmup (more warmup for torch.compile)
    for _ in range(min(n_warmup, 20)):
        fn(w0, w1, w2, q, k, v, lr0, lr1, lr2, chunk, use_muon, momentum, weight_norm=True)
    torch.cuda.synchronize()
    
    # Benchmark recurrent
    n_rec = n_iters // 10
    start = time.perf_counter()
    for _ in range(n_rec):
        fn(w0, w1, w2, q, k, v, lr0, lr1, lr2, chunk, use_muon, momentum, weight_norm=True)
    torch.cuda.synchronize()
    elapsed_rec = time.perf_counter() - start
    
    tokens_per_sec_rec = (n_rec * b * l) / elapsed_rec
    
    return tokens_per_sec_rec

@torch.no_grad()
def benchmark_only_w1_no_wn(use_muon):
    """Benchmarks only_w1_no_wn (has lr1 and momentum)"""
    w0 = torch.rand(b, dh, dk, device=device, dtype=dtype)
    w1 = torch.rand(b, dv, dh, device=device, dtype=dtype)
    w2 = torch.rand(b, dh, dk, device=device, dtype=dtype)
    q = torch.rand(b, l, dk, device=device, dtype=dtype)
    k = torch.rand(b, l, dk, device=device, dtype=dtype)
    v = torch.rand(b, l, dv, device=device, dtype=dtype)
    lr1 = torch.rand(b, l, 1, device=device, dtype=dtype)
    momentum = torch.rand(b, l, 1, device=device, dtype=dtype)
    
    fn_rec = only_w1_no_wn.block_prefix_causal_linear_attention_recurrent
    fn_par = only_w1_no_wn.block_prefix_causal_linear_attention_parallel
    
    # Warmup
    for _ in range(n_warmup):
        fn_par(w0, w1, w2, q, k, v, lr1, chunk, use_muon, momentum)
    torch.cuda.synchronize()
    
    # Benchmark parallel
    start = time.perf_counter()
    for _ in range(n_iters):
        fn_par(w0, w1, w2, q, k, v, lr1, chunk, use_muon, momentum)
    torch.cuda.synchronize()
    elapsed_par = time.perf_counter() - start
    
    # Warmup recurrent
    for _ in range(min(n_warmup, 20)):
        fn_rec(w0, w1, w2, q, k, v, lr1, chunk, use_muon, momentum)
    torch.cuda.synchronize()
    
    # Benchmark recurrent (fewer iters since slower)
    n_rec = n_iters // 10
    start = time.perf_counter()
    for _ in range(n_rec):
        fn_rec(w0, w1, w2, q, k, v, lr1, chunk, use_muon, momentum)
    torch.cuda.synchronize()
    elapsed_rec = time.perf_counter() - start
    
    tokens_per_sec_par = (n_iters * b * l) / elapsed_par
    tokens_per_sec_rec = (n_rec * b * l) / elapsed_rec
    
    return tokens_per_sec_rec, tokens_per_sec_par

@torch.no_grad()
def benchmark_only_w1_no_lr1_no_wn(use_muon):
    """Benchmarks only_w1_no_lr1_no_wn (has momentum, no lr1)"""
    w0 = torch.rand(b, dh, dk, device=device, dtype=dtype)
    w1 = torch.rand(b, dv, dh, device=device, dtype=dtype)
    w2 = torch.rand(b, dh, dk, device=device, dtype=dtype)
    q = torch.rand(b, l, dk, device=device, dtype=dtype)
    k = torch.rand(b, l, dk, device=device, dtype=dtype)
    v = torch.rand(b, l, dv, device=device, dtype=dtype)
    momentum = torch.rand(b, l, 1, device=device, dtype=dtype)
    
    fn_rec = only_w1_no_lr1_no_wn.block_prefix_causal_linear_attention_recurrent
    fn_par = only_w1_no_lr1_no_wn.block_prefix_causal_linear_attention_parallel
    
    # Warmup
    for _ in range(n_warmup):
        fn_par(w0, w1, w2, q, k, v, chunk, use_muon, momentum)
    torch.cuda.synchronize()
    
    # Benchmark parallel
    start = time.perf_counter()
    for _ in range(n_iters):
        fn_par(w0, w1, w2, q, k, v, chunk, use_muon, momentum)
    torch.cuda.synchronize()
    elapsed_par = time.perf_counter() - start
    
    # Warmup recurrent
    for _ in range(min(n_warmup, 20)):
        fn_rec(w0, w1, w2, q, k, v, chunk, use_muon, momentum)
    torch.cuda.synchronize()
    
    # Benchmark recurrent
    n_rec = n_iters // 10
    start = time.perf_counter()
    for _ in range(n_rec):
        fn_rec(w0, w1, w2, q, k, v, chunk, use_muon, momentum)
    torch.cuda.synchronize()
    elapsed_rec = time.perf_counter() - start
    
    tokens_per_sec_par = (n_iters * b * l) / elapsed_par
    tokens_per_sec_rec = (n_rec * b * l) / elapsed_rec
    
    return tokens_per_sec_rec, tokens_per_sec_par

@torch.no_grad()
def benchmark_only_w1_no_lr1_no_wn_momen(use_muon):
    """Benchmarks only_w1_no_lr1_no_wn_momen (no momentum, no lr1, w0/w2 gating)"""
    w0 = torch.rand(b, dh, dk, device=device, dtype=dtype)
    w1 = torch.rand(b, dv, dh, device=device, dtype=dtype)
    w2 = torch.rand(b, dh, dk, device=device, dtype=dtype)
    q = torch.rand(b, l, dk, device=device, dtype=dtype)
    k = torch.rand(b, l, dk, device=device, dtype=dtype)
    v = torch.rand(b, l, dv, device=device, dtype=dtype)
    
    fn_rec = only_w1_no_lr1_no_wn_momen.block_prefix_causal_linear_attention_recurrent
    fn_par = only_w1_no_lr1_no_wn_momen.block_prefix_causal_linear_attention_parallel
    
    # Warmup
    for _ in range(n_warmup):
        fn_par(w0, w1, w2, q, k, v, chunk, use_muon)
    torch.cuda.synchronize()
    
    # Benchmark parallel
    start = time.perf_counter()
    for _ in range(n_iters):
        fn_par(w0, w1, w2, q, k, v, chunk, use_muon)
    torch.cuda.synchronize()
    elapsed_par = time.perf_counter() - start
    
    # Warmup recurrent
    for _ in range(min(n_warmup, 20)):
        fn_rec(w0, w1, w2, q, k, v, chunk, use_muon)
    torch.cuda.synchronize()
    
    # Benchmark recurrent
    n_rec = n_iters // 10
    start = time.perf_counter()
    for _ in range(n_rec):
        fn_rec(w0, w1, w2, q, k, v, chunk, use_muon)
    torch.cuda.synchronize()
    elapsed_rec = time.perf_counter() - start
    
    tokens_per_sec_par = (n_iters * b * l) / elapsed_par
    tokens_per_sec_rec = (n_rec * b * l) / elapsed_rec
    
    return tokens_per_sec_rec, tokens_per_sec_par

@torch.no_grad()
def benchmark_only_w1_straight_qk_no_lr1_no_wn_momen(use_muon):
    """Benchmarks only_w1_straight_qk_no_lr1_no_wn_momen (simple s0 @ q state)"""
    s0 = torch.rand(b, dv, dk, device=device, dtype=dtype)
    q = torch.rand(b, l, dk, device=device, dtype=dtype)
    k = torch.rand(b, l, dk, device=device, dtype=dtype)
    v = torch.rand(b, l, dv, device=device, dtype=dtype)
    
    fn_rec = only_w1_straight_qk_no_lr1_no_wn_momen.block_prefix_causal_linear_attention_recurrent
    fn_par = only_w1_straight_qk_no_lr1_no_wn_momen.block_prefix_causal_linear_attention_parallel
    
    # Warmup
    for _ in range(n_warmup):
        fn_par(s0, q, k, v, chunk, use_muon)
    torch.cuda.synchronize()
    
    # Benchmark parallel
    start = time.perf_counter()
    for _ in range(n_iters):
        fn_par(s0, q, k, v, chunk, use_muon)
    torch.cuda.synchronize()
    elapsed_par = time.perf_counter() - start
    
    # Warmup recurrent
    for _ in range(min(n_warmup, 20)):
        fn_rec(s0, q, k, v, chunk, use_muon)
    torch.cuda.synchronize()
    
    # Benchmark recurrent
    n_rec = n_iters // 10
    start = time.perf_counter()
    for _ in range(n_rec):
        fn_rec(s0, q, k, v, chunk, use_muon)
    torch.cuda.synchronize()
    elapsed_rec = time.perf_counter() - start
    
    tokens_per_sec_par = (n_iters * b * l) / elapsed_par
    tokens_per_sec_rec = (n_rec * b * l) / elapsed_rec
    
    return tokens_per_sec_rec, tokens_per_sec_par

@torch.no_grad()
def benchmark_only_w1_straight_qk_no_wn(use_muon):
    """Benchmarks only_w1_straight_qk_no_wn (straight qk, has lr1, momentum)"""
    s0 = torch.rand(b, dv, dk, device=device, dtype=dtype)
    q = torch.rand(b, l, dk, device=device, dtype=dtype)
    k = torch.rand(b, l, dk, device=device, dtype=dtype)
    v = torch.rand(b, l, dv, device=device, dtype=dtype)
    lr1 = torch.rand(b, l, 1, device=device, dtype=dtype)
    momentum = torch.rand(b, l, 1, device=device, dtype=dtype)
    
    fn_rec = only_w1_straight_qk_no_wn.block_prefix_causal_linear_attention_recurrent
    fn_par = only_w1_straight_qk_no_wn.block_prefix_causal_linear_attention_parallel
    
    # Warmup
    for _ in range(n_warmup):
        fn_par(s0, q, k, v, lr1, chunk, use_muon, momentum)
    torch.cuda.synchronize()
    
    # Benchmark parallel
    start = time.perf_counter()
    for _ in range(n_iters):
        fn_par(s0, q, k, v, lr1, chunk, use_muon, momentum)
    torch.cuda.synchronize()
    elapsed_par = time.perf_counter() - start
    
    # Warmup recurrent
    for _ in range(min(n_warmup, 20)):
        fn_rec(s0, q, k, v, lr1, chunk, use_muon, momentum)
    torch.cuda.synchronize()
    
    # Benchmark recurrent
    n_rec = n_iters // 10
    start = time.perf_counter()
    for _ in range(n_rec):
        fn_rec(s0, q, k, v, lr1, chunk, use_muon, momentum)
    torch.cuda.synchronize()
    elapsed_rec = time.perf_counter() - start
    
    tokens_per_sec_par = (n_iters * b * l) / elapsed_par
    tokens_per_sec_rec = (n_rec * b * l) / elapsed_rec
    
    return tokens_per_sec_rec, tokens_per_sec_par

@torch.no_grad()
def benchmark_only_w1_straight_qk_no_lr1_no_wn(use_muon):
    """Benchmarks only_w1_straight_qk_no_lr1_no_wn (straight qk, has momentum, no lr1)"""
    s0 = torch.rand(b, dv, dk, device=device, dtype=dtype)
    q = torch.rand(b, l, dk, device=device, dtype=dtype)
    k = torch.rand(b, l, dk, device=device, dtype=dtype)
    v = torch.rand(b, l, dv, device=device, dtype=dtype)
    momentum = torch.rand(b, l, 1, device=device, dtype=dtype)
    
    fn_rec = only_w1_straight_qk_no_lr1_no_wn.block_prefix_causal_linear_attention_recurrent
    fn_par = only_w1_straight_qk_no_lr1_no_wn.block_prefix_causal_linear_attention_parallel
    
    # Warmup
    for _ in range(n_warmup):
        fn_par(s0, q, k, v, chunk, use_muon, momentum)
    torch.cuda.synchronize()
    
    # Benchmark parallel
    start = time.perf_counter()
    for _ in range(n_iters):
        fn_par(s0, q, k, v, chunk, use_muon, momentum)
    torch.cuda.synchronize()
    elapsed_par = time.perf_counter() - start
    
    # Warmup recurrent
    for _ in range(min(n_warmup, 20)):
        fn_rec(s0, q, k, v, chunk, use_muon, momentum)
    torch.cuda.synchronize()
    
    # Benchmark recurrent
    n_rec = n_iters // 10
    start = time.perf_counter()
    for _ in range(n_rec):
        fn_rec(s0, q, k, v, chunk, use_muon, momentum)
    torch.cuda.synchronize()
    elapsed_rec = time.perf_counter() - start
    
    tokens_per_sec_par = (n_iters * b * l) / elapsed_par
    tokens_per_sec_rec = (n_rec * b * l) / elapsed_rec
    
    return tokens_per_sec_rec, tokens_per_sec_par


if __name__ == "__main__":
    print(f"Config: b={b}, l={l}, dk={dk}, dh={dh}, dv={dv}, chunk={chunk}")
    print(f"Warmup: {n_warmup}, Iterations: {n_iters} (parallel), {n_iters//10} (recurrent)")
    print("="*80)

    print("\nBaseline: original.py (updates all w0/w1/w2, has weight norm) - recurrent only")
    rec = benchmark_original(True)
    print(f"   Recurrent: {rec/1e6:.2f}M tokens/sec")
    
    print("\nVariant1: only_w1.py (only updates w1, has weight norm) - recurrent only")
    rec = benchmark_only_w1(True)
    print(f"   Recurrent: {rec/1e6:.2f}M tokens/sec")
    
    print("\nVariant2: only_w1_no_wn.py (has lr1, momentum, w0/w2 gating, no weight norm)")
    rec, par = benchmark_only_w1_no_wn(True)
    print(f"   Recurrent: {rec/1e6:.2f}M tokens/sec")
    print(f"   Parallel:  {par/1e6:.2f}M tokens/sec")
    print(f"   Speedup:   {par/rec:.2f}x")
    
    print("\nVariant3: only_w1_straight_qk_no_wn.py (straight qk, has lr1, momentum)")
    rec, par = benchmark_only_w1_straight_qk_no_wn(True)
    print(f"   Recurrent: {rec/1e6:.2f}M tokens/sec")
    print(f"   Parallel:  {par/1e6:.2f}M tokens/sec")
    print(f"   Speedup:   {par/rec:.2f}x")
    
    print("\nVariant4: only_w1_straight_qk_no_lr1_no_wn.py (straight qk, has momentum, no lr1)")
    rec, par = benchmark_only_w1_straight_qk_no_lr1_no_wn(True)
    print(f"   Recurrent: {rec/1e6:.2f}M tokens/sec")
    print(f"   Parallel:  {par/1e6:.2f}M tokens/sec")
    print(f"   Speedup:   {par/rec:.2f}x")
    
    print("\nVariant5: only_w1_straight_qk_no_lr1_no_wn_momen.py (simple state, no momentum)")
    rec, par = benchmark_only_w1_straight_qk_no_lr1_no_wn_momen(True)
    print(f"   Recurrent: {rec/1e6:.2f}M tokens/sec")
    print(f"   Parallel:  {par/1e6:.2f}M tokens/sec")
    print(f"   Speedup:   {par/rec:.2f}x")
    
    print("\nVariant6. only_w1_straight_qk_no_lr1_no_wn_muon_momen.py (simple state, no muon, no momentum)")
    rec, par = benchmark_only_w1_straight_qk_no_lr1_no_wn_momen(False)
    print(f"   Recurrent: {rec/1e6:.2f}M tokens/sec")
    print(f"   Parallel:  {par/1e6:.2f}M tokens/sec")
    print(f"   Speedup:   {par/rec:.2f}x")

    print("\n" + "="*80)
    print("Benchmark complete!")
