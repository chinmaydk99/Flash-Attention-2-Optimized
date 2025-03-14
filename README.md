# Flash Attention V2

This repository implements Flash Attention V2 using Triton for fast, memory‐efficient attention computations in Transformer models. The implementation features both the forward and backward passes, a numerically stable online softmax, and hierarchical block tiling to support long sequences and efficient GPU usage.

---

## Table of Contents

- [Overview](#overview)
- [Mathematical Background](#mathematical-background)
  - [Standard Attention](#standard-attention)
  - [Stable Softmax via Online Aggregation](#stable-softmax-via-online-aggregation)
  - [Hierarchical Tiling](#hierarchical-tiling)
- [Triton Kernels and Implementation](#triton-kernels-and-implementation)
  - [Forward Pass Kernel: `flash_attn_v2_forward`](#forward-pass-kernel-flash_attn_v2_forward)
  - [Backward Preprocess Kernel: `flash_attn_v2_backward_preprocess`](#backward-preprocess-kernel-flash_attn_v2_backward_preprocess)
  - [Backward dK/dV Kernel: `flash_attn_v2_backward_dkv`](#backward-dkdv-kernel-flash_attn_v2_backward_dkv)
  - [Backward dQ Kernel (Not Shown Here)](#backward-dq-kernel)
- [Autograd Function Integration](#autograd-function-integration)
- [Reference PyTorch Implementation](#reference-pytorch-implementation)
- [Design Quirks and Challenges](#design-quirks-and-challenges)
- [Running and Benchmarking](#running-and-benchmarking)
- [Future Work](#future-work)
- [Conclusion](#conclusion)

---

## Overview

Flash Attention V2 is designed to:
- **Reduce memory usage:** By processing data in blocks.
- **Improve numerical stability:** Using a stable online softmax algorithm.
- **Support causal attention:** For autoregressive tasks.
- **Leverage GPU performance:** Via custom Triton kernels that perform block-wise tiling and coalesced memory accesses.

Both the forward and backward passes are implemented using Triton. The kernels are designed to work with 4D tensors of shape \([B, H, S, d]\) (Batch, Heads, Sequence Length, Head Dimension).

---

## Mathematical Background

### Standard Attention

Standard attention computes:
\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]
where:
- \( Q \), \( K \), \( V \) are the query, key, and value matrices.
- \( d_k \) is the dimensionality per head.
- The softmax is applied row-wise on the score matrix \( \frac{QK^\top}{\sqrt{d_k}} \).

### Stable Softmax via Online Aggregation

To prevent numerical overflow/underflow in the softmax, Flash Attention V2 uses an online stable softmax. For each query row, the kernel:
1. **Computes a running maximum \( m_i \):**
   \[
   m_i = \max(S) \quad \text{(over the current block)}
   \]
2. **Shifts the scores:**
   \[
   p = \exp(S - m_i)
   \]
3. **Accumulates the normalization term \( l_i \):**
   \[
   l_i = \sum p
   \]
4. **Accumulates the weighted sum:**
   \[
   \text{acc} = \sum p \times V
   \]
5. **Final output is given by:**
   \[
   \text{Output} = \frac{\text{acc}}{l_i}
   \]
This “online” update is computed across blocks of keys/values to support very long sequences without materializing the full score matrix.

### Hierarchical Tiling

The kernels partition the sequence into blocks:
- **BLOCK_SIZE_M:** Number of query rows processed per kernel call.
- **BLOCK_SIZE_N:** Number of key/value rows processed in each iteration.
- **BLOCK_SIZE_K:** The tiling size along the head dimension.  
For efficiency, `BLOCK_SIZE_K` is rounded up to the next power of 2 if needed.

Hierarchical tiling:
- Maximizes data reuse (e.g., the query block is loaded once and reused across key/value blocks).
- Improves memory coalescing by operating on contiguous memory blocks.
- Enables handling of sequences that are not multiples of the block size by using masks.

---

## Triton Kernels and Implementation

### Forward Pass Kernel: `flash_attn_v2_forward`

This kernel computes the forward pass of Flash Attention V2.

#### Key Steps:
1. **Grid Indexing:**
   - **`batch_id`** selects the batch.
   - **`head_id`** selects the attention head.
   - **`m_id`** selects the query block (each block contains `BLOCK_SIZE_M` rows).

2. **Memory Offsets:**
   The kernel computes offsets for each tensor using strides:
   ```python
   q_batch_offset = batch_id * q_batch_stride
   q_head_offset = head_id * q_head_stride
   ...
