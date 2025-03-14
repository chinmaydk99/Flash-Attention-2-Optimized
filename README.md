# Flash-Attention-2-Optimized

### Overview
This repository contains an implementation of a Triton-based Flash Attention operation for efficient sequence processing in deep learning models. The implementation leverages Triton's JIT compilation and optimized memory access patterns to accelerate attention computation while maintaining precision.

### Features
Efficient attention computation using block-wise processing and softmax scaling.
Causal and non-causal attention support.
Optimized GPU memory access to reduce latency.
Custom Triton kernel to speed up matrix multiplications for QK^T and softmax.

### Dependencies
To run this implementation, install the following:

Python 3.8+
PyTorch (CUDA-enabled)
Triton (pip install triton)



### How It Works
1. Triton Kernel (_attn_fwd)
This function computes the scaled dot-product attention using block-wise matrix multiplications and a custom softmax computation. The logic:

Load Query, Key, and Value blocks into SRAM.
Compute QK^T (query-key dot product).
Apply causal mask (if needed).
Perform numerically stable softmax (using logsumexp trick).
Multiply attention scores with Value matrix (QK^T * V).
Inner Loop Optimization (_attn_fwd_inner)
The kernel efficiently processes attention by:

Splitting into stages to handle causal attention masks.
Using Triton’s matrix tiling and block-wise operations to optimize memory movement.
Keeping frequently used data in shared memory (SRAM).

2. Autograd Function (TritonAttention)
This class provides:

Forward Pass: Calls the Triton kernel to compute attention efficiently.
Backward Pass: (To be implemented) Uses saved values (Q, K, V, O, M) for efficient gradient computation.
Usage
Running a Test
You can test the Triton implementation against PyTorch’s reference attention computation:

python
Copy
Edit
from triton_attention import TritonAttention, test_op

BATCH_SIZE = 4
NUM_HEADS = 8
SEQ_LEN = 128
HEAD_DIM = 64
causal = True

test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal)
The test:

Generates random Q, K, V tensors.
Computes attention using both PyTorch and Triton.
Validates output similarity using .backward() to compare gradients.
Performance Gains
Triton-based attention is significantly faster than naive PyTorch implementations due to:

Efficient memory access (blocked processing)
Reduced redundant computation (logsumexp trick)
Parallelized computation across heads & sequences
For large sequence lengths (≥512), this optimization is substantially faster than standard attention.

### Next Steps
Implement Backward Pass for full gradient support.
Optimize Triton tiling further for larger sequence lengths.
Extend to multi-query attention variants.
