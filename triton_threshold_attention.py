"""
Triton implementation of Threshold Rectified Attention.

This implements a fused kernel for Threshold Rectified Attention that avoids materializing
the full attention matrix, similar to FlashAttention.

Threshold Rectified Attention formula:
    out = (ReLU(Q@K^T - tau))^p @ V

Where tau(i) = beta * sqrt(2 * log(i+1) / d) is a position-dependent threshold.
"""

import torch
import triton
import triton.language as tl
import math


@triton.jit
def _threshold_rela_fwd_kernel(
    Q, K, V, Out,
    tau_scale,  # beta parameter
    seq_len,
    relu_power,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Forward kernel for Threshold Rectified Attention.
    
    Each block computes a BLOCK_M x head_dim output tile.
    """
    # Program ID
    pid_bh = tl.program_id(0)  # batch * num_heads (flattened)
    pid_m = tl.program_id(1)  # which M block
    
    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    
    # Initialize pointers - use stride_*h for flattened batch*heads dimension
    q_ptrs = Q + pid_bh * stride_qh + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
    k_ptrs = K + pid_bh * stride_kh + offs_n[None, :] * stride_kt + offs_d[:, None] * stride_kd
    v_ptrs = V + pid_bh * stride_vh + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd
    
    # Load Q block (BLOCK_M x BLOCK_D)
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < seq_len), other=0.0)
    
    # Accumulator for output - use float32 for numerical stability
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Compute tau for each query position: tau(i) = beta * sqrt(2 * log(i+1) / d)
    # offs_m is the query position (0-indexed), so we use offs_m + 1
    # Compute tau in fp32 for numerical accuracy
    row_indices = (offs_m + 1.0).to(tl.float32)  # 1-indexed positions
    tau = tau_scale * tl.sqrt((2.0 * tl.log(row_indices)) / HEAD_DIM)
    
    # Loop over K, V in BLOCK_N chunks
    for start_n in range(0, seq_len, BLOCK_N):
        # Current key/value positions
        curr_n = start_n + offs_n
        
        # Load K block (BLOCK_D x BLOCK_N) - recompute pointers each time
        k_ptrs_curr = K + pid_bh * stride_kh + curr_n[None, :] * stride_kt + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs_curr, mask=(curr_n[None, :] < seq_len), other=0.0)
        
        # Compute attention scores: Q @ K^T (BLOCK_M x BLOCK_N)
        qk = tl.dot(q, k)
        
        # Apply causal mask: only attend to positions <= current position
        # For query at position i, can only attend to keys at positions j <= i
        mask = curr_n[None, :] <= offs_m[:, None]
        qk = tl.where(mask, qk, 0.0)
        
        # Apply threshold: ReLU(qk - tau)
        # Convert tau to match qk dtype for compatibility with bf16/fp16
        # tau has shape (BLOCK_M,), need to broadcast
        tau_cast = tau[:, None].to(qk.dtype)
        qk_thresh = qk - tau_cast
        qk_relu = tl.where(qk_thresh > 0.0, qk_thresh, 0.0)
        
        # Apply power: ReLU^p
        if relu_power == 2.0:
            qk_powered = qk_relu * qk_relu
        elif relu_power == 1.0:
            qk_powered = qk_relu
        else:
            # General power using exp and log
            # x^p = exp(p * log(x)) for x > 0
            qk_powered = tl.where(
                qk_relu > 0.0,
                tl.exp(relu_power * tl.log(qk_relu + 1e-10)),
                0.0
            )
        
        # Load V block (BLOCK_N x BLOCK_D)
        v_ptrs_curr = V + pid_bh * stride_vh + curr_n[:, None] * stride_vt + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs_curr, mask=(curr_n[:, None] < seq_len), other=0.0)
        
        # Accumulate: attn @ V
        # Convert both to fp32 for stable accumulation
        acc += tl.dot(qk_powered.to(tl.float32), v.to(tl.float32))
    
    # Write output
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    out_ptrs = Out + pid_bh * stride_oh + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=(offs_m[:, None] < seq_len))


@triton.jit
def _threshold_rela_bwd_kernel_dq(
    Q, K, V, dOut, dQ,
    tau_scale,
    seq_len,
    relu_power,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_dob, stride_doh, stride_dot, stride_dod,
    stride_dqb, stride_dqh, stride_dqt, stride_dqd,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Backward kernel for computing dQ.
    
    dQ = (dOut @ V^T) * d(ReLU(qk - tau)^p) / dQ
       = (dOut @ V^T) * p * ReLU(qk - tau)^(p-1) * 1{qk > tau} @ K
    """
    pid_bh = tl.program_id(0)  # Flattened batch*num_heads index
    pid_m = tl.program_id(1)
    
    # Split pid_bh into batch and head indices
    # pid_bh = batch_idx * num_heads + head_idx
    # But we don't have num_heads here, so we use strides correctly
    # Actually, stride_qb is for the batch dimension, stride_qh is for num_heads
    # pid_bh indexes into the flattened BH dimension
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    
    # Load Q and dOut - use stride_qb + stride_qh to get correct offset
    # For a (B, NH, T, D) tensor, element [b, h, t, d] is at: b*stride_qb + h*stride_qh + t*stride_qt + d*stride_qd
    # We have pid_bh = b*NH + h, so we need to reconstruct b and h
    # BUT: stride_qb is typically NH*T*D and stride_qh is T*D
    # So pid_bh * stride_qh gives us the correct offset for flattened (b,h)
    q_ptrs = Q + pid_bh * stride_qh + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < seq_len), other=0.0)
    
    do_ptrs = dOut + pid_bh * stride_doh + offs_m[:, None] * stride_dot + offs_d[None, :] * stride_dod
    do = tl.load(do_ptrs, mask=(offs_m[:, None] < seq_len), other=0.0)
    
    # Compute tau
    row_indices = offs_m + 1.0
    tau = tau_scale * tl.sqrt((2.0 * tl.log(row_indices)) / HEAD_DIM)
    
    # Accumulator for dQ
    dq_acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Loop over K, V
    for start_n in range(0, seq_len, BLOCK_N):
        curr_n = start_n + offs_n
        
        # Load K as (BLOCK_D x BLOCK_N) for Q @ K^T computation
        # K is stored as (batch, heads, seq, dim)
        k_ptrs = K + pid_bh * stride_kh + curr_n[None, :] * stride_kt + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=(curr_n[None, :] < seq_len), other=0.0)
        
        # Load V (BLOCK_N x BLOCK_D)
        v_ptrs = V + pid_bh * stride_vh + curr_n[:, None] * stride_vt + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=(curr_n[:, None] < seq_len), other=0.0)
        
        # Recompute attention scores: Q @ K
        # q: (M, D), k: (D, N) → qk: (M, N)
        qk = tl.dot(q, k)
        mask = curr_n[None, :] <= offs_m[:, None]
        qk = tl.where(mask, qk, 0.0)
        
        # Compute gradient of ReLU^p
        # Convert tau to match qk dtype
        tau_cast = tau[:, None].to(qk.dtype)
        qk_thresh = qk - tau_cast
        qk_relu = tl.where(qk_thresh > 0.0, qk_thresh, 0.0)
        
        # d(ReLU(x)^p)/dx = p * ReLU(x)^(p-1) * 1{x > 0}
        if relu_power == 2.0:
            grad_relu_p = 2.0 * qk_relu  # 2 * x^1
        elif relu_power == 1.0:
            grad_relu_p = tl.where(qk_thresh > 0.0, 1.0, 0.0)
        else:
            grad_relu_p = tl.where(
                qk_relu > 0.0,
                relu_power * tl.exp((relu_power - 1.0) * tl.log(qk_relu + 1e-10)),
                0.0
            )
        
        # dOut @ V^T gives gradient w.r.t. attention weights
        # do: (BLOCK_M x BLOCK_D), v: (N, D), v^T: (D, N)
        # Convert to fp32 for stable accumulation
        dAttn = tl.dot(do.to(tl.float32), tl.trans(v.to(tl.float32)))  # (BLOCK_M x BLOCK_N)
        
        # Apply gradient of ReLU^p (convert grad_relu_p to fp32)
        dAttn = dAttn * grad_relu_p.to(tl.float32)
        
        # dQ += dAttn @ K^T where K is (D, N), so K^T is (N, D)
        # dAttn: (M, N), K^T: (N, D) → dQ: (M, D) ✓
        # Convert k transpose to fp32 for stable accumulation
        dq_acc += tl.dot(dAttn, tl.trans(k.to(tl.float32)))
    
    # Store dQ
    dq_ptrs = dQ + pid_bh * stride_dqh + offs_m[:, None] * stride_dqt + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq_acc.to(dQ.dtype.element_ty), mask=(offs_m[:, None] < seq_len))


@triton.jit
def _threshold_rela_bwd_kernel_dkv(
    Q, K, V, dOut, dK, dV,
    tau_scale,
    seq_len,
    relu_power,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_dob, stride_doh, stride_dot, stride_dod,
    stride_dkb, stride_dkh, stride_dkt, stride_dkd,
    stride_dvb, stride_dvh, stride_dvt, stride_dvd,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Backward kernel for computing dK and dV.
    """
    pid_bh = tl.program_id(0)  # Flattened batch*num_heads index
    pid_n = tl.program_id(1)  # Key/Value position block
    
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    offs_m = tl.arange(0, BLOCK_M)
    
    # Load K and V for this block
    k_ptrs = K + pid_bh * stride_kh + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd
    k = tl.load(k_ptrs, mask=(offs_n[:, None] < seq_len), other=0.0)
    
    v_ptrs = V + pid_bh * stride_vh + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd
    v = tl.load(v_ptrs, mask=(offs_n[:, None] < seq_len), other=0.0)
    
    # Accumulators
    dk_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dv_acc = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    
    # Loop over Q, dOut
    for start_m in range(0, seq_len, BLOCK_M):
        curr_m = start_m + offs_m
        
        # Load Q (BLOCK_M x BLOCK_D)
        q_ptrs = Q + pid_bh * stride_qh + curr_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=(curr_m[:, None] < seq_len), other=0.0)
        
        # Load dOut (BLOCK_M x BLOCK_D)
        do_ptrs = dOut + pid_bh * stride_doh + curr_m[:, None] * stride_dot + offs_d[None, :] * stride_dod
        do = tl.load(do_ptrs, mask=(curr_m[:, None] < seq_len), other=0.0)
        
        # Compute tau for these query positions
        row_indices = curr_m + 1.0
        tau = tau_scale * tl.sqrt((2.0 * tl.log(row_indices)) / HEAD_DIM)
        
        # Recompute attention: Q @ K^T
        # q: (BLOCK_M x BLOCK_D), k^T: (BLOCK_D x BLOCK_N)
        qk = tl.dot(q, tl.trans(k))  # (BLOCK_M x BLOCK_N)
        
        # Causal mask
        mask = offs_n[None, :] <= curr_m[:, None]
        qk = tl.where(mask, qk, 0.0)
        
        # Apply threshold and power
        # Convert tau to match qk dtype
        tau_cast = tau[:, None].to(qk.dtype)
        qk_thresh = qk - tau_cast
        qk_relu = tl.where(qk_thresh > 0.0, qk_thresh, 0.0)
        
        if relu_power == 2.0:
            qk_powered = qk_relu * qk_relu
            grad_relu_p = 2.0 * qk_relu
        elif relu_power == 1.0:
            qk_powered = qk_relu
            grad_relu_p = tl.where(qk_thresh > 0.0, 1.0, 0.0)
        else:
            qk_powered = tl.where(qk_relu > 0.0, tl.exp(relu_power * tl.log(qk_relu + 1e-10)), 0.0)
            grad_relu_p = tl.where(qk_relu > 0.0, relu_power * tl.exp((relu_power - 1.0) * tl.log(qk_relu + 1e-10)), 0.0)
        
        # dV: attn^T @ dOut
        # qk_powered^T: (BLOCK_N x BLOCK_M), do: (BLOCK_M x BLOCK_D)
        # Convert to fp32 for stable accumulation
        dv_acc += tl.dot(tl.trans(qk_powered.to(tl.float32)), do.to(tl.float32))
        
        # dK: dAttn^T @ Q
        # First compute dAttn = dOut @ V^T * grad(ReLU^p)
        # Convert to fp32 for stable computation
        dAttn = tl.dot(do.to(tl.float32), tl.trans(v.to(tl.float32)))  # (BLOCK_M x BLOCK_N)
        dAttn = dAttn * grad_relu_p.to(tl.float32)
        
        # dK += dAttn^T @ Q (convert q to fp32)
        # dAttn^T: (BLOCK_N x BLOCK_M), Q: (BLOCK_M x BLOCK_D)
        dk_acc += tl.dot(tl.trans(dAttn), q.to(tl.float32))
    
    # Store dK and dV
    dk_ptrs = dK + pid_bh * stride_dkh + offs_n[:, None] * stride_dkt + offs_d[None, :] * stride_dkd
    tl.store(dk_ptrs, dk_acc.to(dK.dtype.element_ty), mask=(offs_n[:, None] < seq_len))
    
    dv_ptrs = dV + pid_bh * stride_dvh + offs_n[:, None] * stride_dvt + offs_d[None, :] * stride_dvd
    tl.store(dv_ptrs, dv_acc.to(dV.dtype.element_ty), mask=(offs_n[:, None] < seq_len))


class ThresholdRelaFunction(torch.autograd.Function):
    """
    Threshold Rectified Attention implemented with Triton.
    
    Forward: out = (ReLU(Q@K^T - tau))^p @ V
    Where tau(i) = beta * sqrt(2 * log(i+1) / d)
    """
    
    @staticmethod
    def forward(ctx, q, k, v, beta, relu_power):
        """
        Args:
            q: (batch, num_heads, seq_len, head_dim)
            k: (batch, num_heads, seq_len, head_dim)
            v: (batch, num_heads, seq_len, head_dim)
            beta: scalar, threshold scaling parameter
            relu_power: scalar, power for ReLU (1.0, 2.0, 15.0, etc.)
        Returns:
            out: (batch, num_heads, seq_len, head_dim)
        """
        # Check shapes
        assert q.shape == k.shape == v.shape
        batch, num_heads, seq_len, head_dim = q.shape
        
        # Convert beta and relu_power to Python scalars if they're tensors
        beta_val = beta.item() if isinstance(beta, torch.Tensor) else float(beta)
        relu_power_val = relu_power.item() if isinstance(relu_power, torch.Tensor) else float(relu_power)
        
        # Save original dtype and convert to FP32 for numerical stability
        original_dtype = q.dtype
        if q.dtype != torch.float32:
            q = q.float()
            k = k.float()
            v = v.float()
        
        # Allocate output
        out = torch.empty_like(q)
        
        # Determine block sizes
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_D = head_dim  # Must match head_dim exactly
        
        # Merge batch and head dimensions for launching
        BH = batch * num_heads
        
        # Launch kernel
        grid = (BH, triton.cdiv(seq_len, BLOCK_M))
        
        _threshold_rela_fwd_kernel[grid](
            q, k, v, out,
            beta_val,
            seq_len,
            relu_power_val,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            HEAD_DIM=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
        )
        
        # Convert output back to original dtype
        if original_dtype != torch.float32:
            out = out.to(original_dtype)
        
        # Save for backward (save FP32 versions for numerical stability)
        ctx.save_for_backward(q, k, v)
        ctx.beta = beta_val
        ctx.relu_power = relu_power_val
        ctx.head_dim = head_dim
        ctx.seq_len = seq_len
        ctx.original_dtype = original_dtype
        
        return out
    
    @staticmethod
    def backward(ctx, dout):
        """
        Compute gradients w.r.t. Q, K, V.
        """
        q, k, v = ctx.saved_tensors
        beta = ctx.beta
        relu_power = ctx.relu_power
        head_dim = ctx.head_dim
        seq_len = ctx.seq_len
        original_dtype = ctx.original_dtype
        
        batch, num_heads = q.shape[:2]
        BH = batch * num_heads
        
        # Ensure dout is FP32 to match saved tensors
        if dout.dtype != torch.float32:
            dout = dout.float()
        
        # Allocate gradient tensors (FP32 for numerical stability)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_D = head_dim
        
        # Compute dQ
        grid_dq = (BH, triton.cdiv(seq_len, BLOCK_M))
        _threshold_rela_bwd_kernel_dq[grid_dq](
            q, k, v, dout, dq,
            beta, seq_len, relu_power,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            HEAD_DIM=head_dim,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        )
        
        # Compute dK and dV
        grid_dkv = (BH, triton.cdiv(seq_len, BLOCK_N))
        _threshold_rela_bwd_kernel_dkv[grid_dkv](
            q, k, v, dout, dk, dv,
            beta, seq_len, relu_power,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            HEAD_DIM=head_dim,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        )
        
        # Convert gradients to match input dtype for PyTorch's gradient engine
        # Gradient accumulation in master weights (FP32) happens inside the optimizer  
        if original_dtype != torch.float32:
            dq = dq.to(original_dtype)
            dk = dk.to(original_dtype)
            dv = dv.to(original_dtype)
        
        return dq, dk, dv, None, None


def threshold_rela_triton(q, k, v, beta, relu_power=2.0):
    """
    Triton implementation of Threshold Rectified Attention.
    
    Args:
        q: (batch, num_heads, seq_len, head_dim)
        k: (batch, num_heads, seq_len, head_dim)
        v: (batch, num_heads, seq_len, head_dim)
        beta: scalar tensor, threshold scaling parameter
        relu_power: scalar, power for ReLU (default: 2.0)
    
    Returns:
        out: (batch, num_heads, seq_len, head_dim)
    """
    return ThresholdRelaFunction.apply(q, k, v, beta, relu_power)


def differential_threshold_rela_triton(q1, q2, k1, k2, v, beta, lambda_param, relu_power=2.0, normalize=True):
    """
    Threshold Differential Attention using two Threshold Rectified Attention operations.
    
    Args:
        q1, q2, k1, k2: (batch, num_heads, seq_len, head_dim)
        v: (batch, num_heads, seq_len, head_dim)
        beta: scalar tensor, threshold scaling parameter
        lambda_param: scalar tensor, differential weighting (0-1)
        relu_power: scalar, power for ReLU (default: 2.0)
        normalize: bool, whether to normalize Q and K (for cosine similarity, default: True)
    
    Returns:
        out: (batch, num_heads, seq_len, head_dim)
    """
    # Optionally normalize for cosine similarity
    if normalize:
        q1 = torch.nn.functional.normalize(q1, p=2, dim=-1)
        k1 = torch.nn.functional.normalize(k1, p=2, dim=-1)
        q2 = torch.nn.functional.normalize(q2, p=2, dim=-1)
        k2 = torch.nn.functional.normalize(k2, p=2, dim=-1)
    
    # Compute two Threshold Rectified Attention paths
    out1 = threshold_rela_triton(q1, k1, v, beta, relu_power)
    out2 = threshold_rela_triton(q2, k2, v, beta, relu_power)
    
    # Differential combination
    lambda_clamped = torch.clamp(lambda_param, min=0.0, max=1.0)
    out = out1 - lambda_clamped * out2
    
    return out

