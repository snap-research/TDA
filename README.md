# Threshold Differential Attention (TDA)

[![ACL 2026](https://img.shields.io/badge/ACL-2026-4b7bec.svg)](https://2026.aclweb.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2601.12145-b31b1b.svg)](https://arxiv.org/abs/2601.12145)
[![PyTorch](https://img.shields.io/badge/PyTorch-required-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Triton](https://img.shields.io/badge/Triton-required-1a237e?logo=nvidia&logoColor=white)](https://triton-lang.org/)

Triton implementation of Threshold Rectified Attention and Threshold Differential Attention with fused kernels for efficient computation.

This repository accompanies the paper **[Threshold Differential Attention for Sink-Free, Ultra-Sparse, and Non-Dispersive Language Modeling](https://arxiv.org/abs/2601.12145)**, accepted to **ACL 2026 (Oral)**.

**Formula:** `out = (ReLU(Q@K^T - tau))^p @ V`

Where `tau(i) = beta * sqrt(2 * log(i+1) / d)` is a position-dependent threshold.

## Usage

### Threshold Rectified Attention

```python
from triton_threshold_attention import threshold_rela_triton
import torch

# Input tensors: (batch, num_heads, seq_len, head_dim)
q = torch.randn(1, 8, 128, 64, device="cuda")
k = torch.randn(1, 8, 128, 64, device="cuda")
v = torch.randn(1, 8, 128, 64, device="cuda")
beta = torch.tensor(1.0)  # threshold scaling parameter

# Compute threshold rectified attention
output = threshold_rela_triton(q, k, v, beta, relu_power=2.0)
```

### Threshold Differential Attention

```python
from triton_threshold_attention import differential_threshold_rela_triton

q1 = torch.randn(1, 8, 128, 64, device="cuda")
q2 = torch.randn(1, 8, 128, 64, device="cuda")
k1 = torch.randn(1, 8, 128, 64, device="cuda")
k2 = torch.randn(1, 8, 128, 64, device="cuda")
v = torch.randn(1, 8, 128, 64, device="cuda")
beta = torch.tensor(1.0)
lambda_param = torch.tensor(0.5)  # differential weighting (0-1)

output = differential_threshold_rela_triton(
    q1, q2, k1, k2, v, 
    beta, lambda_param, 
    relu_power=2.0
)
```

## Parameters

- **q, k, v**: Query, key, value tensors of shape `(batch, num_heads, seq_len, head_dim)`
- **beta**: Threshold scaling parameter (scalar tensor)
- **relu_power**: Power for ReLU activation (default: 2.0)
- **lambda_param**: Differential weighting parameter for threshold differential attention (0-1)
- **normalize**: Whether to normalize Q and K for cosine similarity (default: True)

## Requirements

- PyTorch
- Triton

## Citation

If you use this implementation, please cite:

```bibtex
@inproceedings{huang2026threshold,
  title={Threshold Differential Attention for Sink-Free, Ultra-Sparse, and Non-Dispersive Language Modeling},
  author={Huang, Xingyue and Ding, Xueying and Ju, Mingxuan and Liu, Yozen and Shah, Neil and Zhao, Tong},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics},
  year={2026},
  url={https://arxiv.org/abs/2601.12145}
}
```
