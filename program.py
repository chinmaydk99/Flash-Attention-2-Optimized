import torch
import triton
import triton.language as tl

# Coding the inner loop
@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    block_index_q,
    softmax_scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    SEQ_LEN: tl.constexpr,
):  
    # We are essentially splitting the for loop into stage to process elements below, on and above the diagonal
    
    # The reason we dont fuse the below diagonal and on diagonal elements is to optimize the pipelining that triton does, which will be seen later
    # range of values handled by this stage
    if STAGE == 1:
        # From 0 to the left of the diagonal for causal attention
        # The elements above the diagonal need to be masked out
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q
    
    elif STAGE == 2:
        # Used only for the block in which there is transition between non-masked and masked keys
        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
        # This is for compiler level optimisation that Triton does
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    else:
        # For non causal attention we don't need all this, just process the entire sequence
        lo, hi = 0, SEQ_LEN
    
    K_block_ptr = tl.advance(K_block_ptr, (0, lo)) # Note the indexing. This has been transposed
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0)) 

    # loop over k, v and update accumulator
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        # Just let the compiler know that start_n is a multiple of BLOCK_N, so the compiler can do optimizations
        # Functionality wise this changes nothing, this is just for pipeline optimization
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        # Computing QK. Again K has already been transposed
        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block)


        if STAGE == 2: # This is when we are exactly on the diagonal
            # For causal attention apply mask on indices greater than the diagonal value
            mask = offs_q[:, None] >= (start_kv + offs_kv[None, :])
            QK_block = QK_block * softmax_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
            QK_block -= m_ij[:, None] # This is part of the softmax * operation (S - max element in row)
        else:
            # Compute the maximum value of qk or keep the old max value
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_scale)
            QK_block = QK_block * softmax_scale - m_ij[:, None]
        
        # Compute the exponential of each dot product, so now we are computing exp(qk_ij - m_ij)
        # Again this is a aprt of softmax star operation. We have already subtracted the max value in each row  and this just needs to be exponentiated
        P_block = tl.math.exp(QK_block)

        # Compute the sum by rows of the attention scores. This is for the normalization score. Dividing the softmax star output by this at the end of the loop will yield us the softmax output
        # This is for the current KV block
        l_ij = tl.sum(P_block, 1)

        # Correction factor for previous block
        alpha = tl.math.exp(m_i - m_ij)

        # Apply the correction factor to the previous l_i and add the new l_ij
        l_i = l_i * alpha + l_ij

        V_block = tl.load(V_block_ptr)
        P_block = P_block.to(tl.float16)

         # This computes the following: O_new = P x V + O_old * alpha
        # We start with applying the correction to the output block
        O_block = O_block * alpha[:, None]
        # Next, we perform the multiplication between the P and V bloc. O_block here is the accumulator
        O_block = tl.dot(P_block, V_block, O_block)

        # New estimation of max value. Will be used in correction factor calculation in next block
        m_i = m_ij

        # Move to the next block of K and V
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0)) # V(SEQ_LEN, HEAD_DIM)
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV)) # K[HEAD_DIM, SEQ_LEN]

    return O_block, l_i, m_i

# Triton Kernel
@triton.jit
def _attn_fwd(
    Q,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    K,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    V,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    softmax_scale,
    M,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN
    O,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)

    # This indicate which block in the sequence length to process
    block_index_q = tl.program_id(0)

    # This indicates which head and batch to process. Each program is associated with a single head of a single batch
    # tl.program_id(1) is assigned with BATCH_SIZE* NUM_HEADS
    index_batch_head = tl.program_id(1)
    # This indicate which batch this program is associated with (each batch has NUM_HEADS heads)
    index_batch = index_batch_head // NUM_HEADS
    # This indicate the position of the head in the batch
    index_head = index_batch_head % NUM_HEADS

    # We need an offset to move in batch_size and head_dim dimension to access the element/s that we need
    # This allows to get the (N_CTX, HEAD_DIM) block in the Q, K, V by selecting indexing it by batch and head
    qvk_offset = (
        index_batch.to(tl.int64) * stride_Q_batch
        + index_head.to(tl.int64) * stride_Q_head
    )

    # Each of these is a tensor of pointers of size `block_shape``
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset, # Q[index_batch, index_head, block_index_q*BLOCK_SIZE_q:, :]
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_Q_seq, stride_Q_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset, # V[index_batch, index_head,:, :]
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(0, 0), # We are parallelizing across Q blocks, therefore we need access to all V values. Therefore no skipping
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr( # Since we need transpose of K, we invert the order in which we feed in the strides
        base=K + qvk_offset,  # K[index_batch, index_head,:, :]
        shape=(HEAD_DIM, SEQ_LEN),
        strides=(
            stride_K_dim,
            stride_K_seq,
        ),  # We invert the strides w.r.t Q, so we transpose the matrix
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        order=(0, 1),
    )

    O_block_ptr = tl.make_block_ptr( # Each output block is same dimension as the query block being processed
        base=O + qvk_offset, # O[index_batch, index_head, block_index_q*BLOCK_SIZE_q:, :]
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_O_seq, stride_O_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    # offs_q: the offsets for the tokens in the Q to process. Loads these many queries for current processing
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    
    # Similarly we have offsets for the tokens in the K and V sequence to process
    # unlike in query offset we aren't skipping anything since the program needs to iteratively go through the keys and values
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)

    # Next up initliazing the running maximum array(mi) and normalization (Li) so that we can calculate softmax(QK^T)
    # m_i: the running maximum. We have one for each query
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")
    
    # l_i: Normalization factor for each row. We have one for each query (as we sum the attention scores by rows)
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0

    # This is one row of blocks in the output matrix (Softmax(Q1K1.T/ dk**0.5)* V)
    # acc: the accumulator for the output, which is a group of rows of the O matrix
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    # Next up is the inner loop in the flash attention 2 algorithm
    # For causal attention we don't want Q blocks to attend to key values that come after it

    # load the blocks of Q: it will stay in SRAM throughout
    Q_block = tl.load(Q_block_ptr)

    # Stage: 3 if causal, else 1

    if STAGE == 1 or STAGE == 3:
        # This step runs for non-causal attention or for the blocks to the left of the diagonal in the causal attention
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i, 
            m_i, # Max for each query
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            4 - STAGE,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )

    if STAGE == 3:
        # This step runs for the blocks to the right of the diagonal in the causal attention
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            2,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )


# Operators in torch are defined using autograd.function. Must provide two functionalities. Backward pass and forward pass
class TritonAttention(torch.autograd.Function):
    @staticmethod
    # Forward method takes context as input which saves activations necessary during backward pass
    def forward(ctx, Q, K, V, causal, softmax_scale):
        HEAD_DIM_Q, HEAD_DIM_K = Q.shape[-1], K.shape[-1]
        HEAD_DIM_V = V.shape[-1]

        BATCH_SIZE, NUM_HEADS, SEQ_LEN< HEAD_DIM = Q.shape

        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V

        O = torch.empty_like(Q)
        # In cross attention, Query comes from one sequence while key and value come from a differeny sequence

        stage = 3 if causal else 1

        # Specify how many grids need to be launched
        # We are parallelizing across batches. Within batches we want parallelization by head
        # Within batches we are going to group queries and parallelize
        grid = lambda args: (
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]), # Tells how many many blocks we have. The group of queries we work with within the given head
            BATCH_SIZE * NUM_HEADS, # Which head from which batch are we working with 
            1,
        )

        # Number of parallel programs = BATCH_SIZE * NUM_HEADS * NUMBER_OF_BLOCKS_Q
        # To prevent calculating max value per row and normalization factor during backprop from scratch, we can save logsumexp instead
        M = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32
        )

        _attn_fwd[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=softmax_scale,
            M=M, # Info saved to be used during backprop
            O=O,
            # Strides help us access any elements we want. What we have intially is just pointer to the starting element
            stride_Q_batch=Q.stride(0),
            stride_Q_head=Q.stride(1),
            stride_Q_seq=Q.stride(2),
            stride_Q_dim=Q.stride(3),
            stride_K_batch=K.stride(0),
            stride_K_head=K.stride(1),
            stride_K_seq=K.stride(2),
            stride_K_dim=K.stride(3),
            stride_V_batch=V.stride(0),
            stride_V_head=V.stride(1),
            stride_V_seq=V.stride(2),
            stride_V_dim=V.stride(3),
            stride_O_batch=O.stride(0),
            stride_O_head=O.stride(1),
            stride_O_seq=O.stride(2),
            stride_O_dim=O.stride(3),
            BATCH_SIZE=Q.shape[0],
            NUM_HEADS=Q.shape[1],
            SEQ_LEN=Q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            STAGE=stage, # specifying if we are implementing causal or non causal attention
        )

        # Saving information needed for backward pass
        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return O



def test_op(BATCH_SIZE,NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype = torch.float16):
    Q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype= dtype, device = "cuda"
        )
        .normal_(mean = 0.0, std = 0.5)
    )

    K = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype= dtype, device = "cuda"
        )
        .normal_(mean = 0.0, std = 0.5)
    )

    V = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype= dtype, device = "cuda"
        )
        .normal_(mean = 0.0, std = 0.5)
    )


softmax_scale = 1/(HEAD_DIM**0.5) #square root of dk
dO = torch.randn_like(Q) # Needed for backward pass

#Reference implementation

# Create causal mask. tril creates a lower triangular mask
MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN)), device = "cuda")
P = torch.matmul(Q, K.transpose(2,3)) * softmax_scale  #(batch_size, num_heads, seq_len, head_dim) -> (batch_size, num_heads, head_dim, seq_len)
# Result will be (batch_size, num_heads, seq_len, seq_len)

if causal:
    P[:, :, MASK == 0] = float("-inf") # After passing through softmax this becomes zero

P = torch.softmax(P.float(), dim = -1).half() # Converting to fp16 precision

# reference_output
    
ref_O = torch.matmul(P, V)
ref_O.backward(dO)
ref_dV, V.grad = V.grad.clone(), None
ref_dK, K.grad = K.grad.clone(), None
ref_dQ, Q.grad = Q.grad.clone(), None

# triton implementation
tri_out = TritonAttention.apply(Q, K, V, causal, softmax_scale).half()
tri_out.backward(dO)
tri_dV, V.grad = V.grad.clone(), None
tri_dK, K.grad = K.grad.clone(), None
tri_dQ, Q.grad = Q.grad.clone(), None

