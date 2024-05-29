import torch
import math
from flash_attn import flash_attn_with_kvcache

from typing import Optional

import torch
import triton
import triton.language as tl


def cdiv(a, b):
    return (a + b - 1) // b


def ref_tree_attention(query, key_cache, value_cache, block_tables,
                       context_lens, attention_mask):
    output = torch.zeros_like(query)
    bs, seq_len, num_heads, head_size = query.shape
    num_blocks, block_size, num_kv_heads, k_head_size = key_cache.shape
    assert head_size == k_head_size
    assert key_cache.shape == value_cache.shape
    assert num_heads % num_kv_heads == 0
    num_query = num_heads // num_kv_heads
    scale = 1 / math.sqrt(head_size)

    def gather_kv(cache, block_table, ctx_len):
        output = torch.empty(ctx_len,
                             num_kv_heads,
                             head_size,
                             dtype=cache.dtype,
                             device=query.device)
        for i in range(ctx_len):
            block_id = block_table[i // block_size]
            offset = i % block_size
            output[i] = cache[block_id, offset]
        return output

    bias = torch.where(attention_mask > 0, 0, -torch.inf)

    for i in range(bs):
        ctx_len = context_lens[i]
        k = gather_kv(key_cache, block_tables[i], ctx_len)
        v = gather_kv(value_cache, block_tables[i], ctx_len)
        tmp = torch.empty(num_heads,
                          seq_len,
                          ctx_len,
                          dtype=query.dtype,
                          device=query.device)
        for j in range(num_heads):
            # shape [seq_len, ctx_len]
            qk = torch.matmul(query[i, :, j, :],
                              k[:, j // num_query, :].t()) * scale
            qk[:, -seq_len:] += bias
            tmp[j] = torch.softmax(qk, dim=-1)
        for j in range(num_heads):
            # shape [seq_len, ctx_len]
            qk = tmp[j]
            # [seq_len, head_size]
            qkv = torch.matmul(qk, v[:, j // num_query, :])
            output[i, :, j, :] = qkv
    return output


@triton.jit
def _fwd_kernel(
    query,  # [bsz, num_query_tokens, num_heads, head_size]
    output,  # [bsz, num_query_tokens, num_heads, head_size]
    key_cache,  # [num_blocks, num_kv_heads, head_size/x, block_size, x]
    value_cache,  # [num_blocks, num_kv_heads, head_size, block_size]
    block_tables,  # [bsz, max_num_blocks]
    context_lens,  # [bsz]
    causal_mask,
    stride_query_0,
    stride_query_1,
    stride_query_2,
    stride_output_0,
    stride_output_1,
    stride_output_2,
    stride_key_0,
    stride_key_1,
    stride_key_2,
    stride_value_0,
    stride_value_1,
    stride_value_2,
    stride_block_tables_0,
    stride_causal_mask_0,
    head_size: tl.constexpr,
    num_query_tokens: tl.constexpr,
    block_size: tl.constexpr,
    sm_scale: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    block_m_idx = tl.program_id(2)

    kv_head_idx = head_idx // num_queries_per_kv
    prefix_len = tl.load(context_lens + batch_idx) - num_query_tokens
    ctx_len = prefix_len + (block_m_idx + 1) * block_size

    offs_m = block_m_idx * block_size + tl.arange(0, block_size)
    offs_block_size = tl.arange(0, block_size)
    offs_head_size = tl.arange(0, head_size)

    # load q from block ptr of shape [block_size, head_size]
    offs_q = batch_idx * stride_query_0 + \
        offs_m[:, None] * stride_query_1 + head_idx * \
        stride_query_2 + offs_head_size[None, :]
    q = tl.load(query + offs_q,
                mask=offs_m[:, None] < num_query_tokens,
                other=0.0)

    m_i = tl.zeros([block_size], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([block_size], dtype=tl.float32)
    acc = tl.zeros([block_size, head_size], dtype=tl.float32)

    for block_idx in range(0, tl.cdiv(ctx_len, block_size)):
        token_offs = (block_idx * block_size + offs_block_size)
        block_offs = tl.load(block_tables + batch_idx * stride_block_tables_0 +
                             token_offs // block_size,
                             mask=token_offs < ctx_len)
        # load k from block ptr of shape [head_size, block_size]
        # [num_blocks, block_size, num_heads, head_size]
        offs_k = block_offs[None, :] * stride_key_0 + offs_block_size[None, :] * stride_key_1 + \
            kv_head_idx * stride_key_2 + offs_head_size[:, None]
        k = tl.load(
            key_cache + offs_k,
            mask=block_idx * block_size + offs_block_size[None, :] < ctx_len,
            other=0.0)
        # calculate qk
        qk = tl.zeros([block_size, block_size], dtype=tl.float32)
        qk += tl.dot(q, k)
        # apply causal mask
        offs_a = prefix_len + offs_m[:, None]
        offs_b = token_offs[None, :]
        auto_mask = offs_a >= offs_b
        offs_mask = offs_m[:, None] * stride_causal_mask_0 + (
            token_offs[None, :] - prefix_len)
        load_mask = (offs_m[:, None] < num_query_tokens) & (
            (token_offs[None, :] < ctx_len) &
            (token_offs[None, :] >= prefix_len))
        mask = tl.load(causal_mask + offs_mask, mask=load_mask, other=auto_mask)
        qk = qk * sm_scale + tl.where(mask, 0, -1.0e6)
        # calculate m_ij and l_ij
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.exp(qk)
        l_ij = tl.sum(p, 1)
        # update m_i and l_i
        alpha = tl.exp(m_i - m_ij)
        m_i = m_ij
        l_i = l_i * alpha + l_ij
        # rescale previous acc
        acc = acc * alpha[:, None]
        # load v from block ptr of shape [block_size, head_size]
        # [num_blocks, block_size, num_heads, head_size]
        offs_v = block_offs[:, None] * stride_value_0 + offs_block_size[:, None] * stride_value_1 + \
              kv_head_idx * stride_value_2 + offs_head_size[None, :]
        v = tl.load(value_cache + offs_v)
        # update acc
        acc += tl.dot(p.to(v.dtype), v)
    # divide acc by row sum
    acc = acc / l_i[:, None]
    # update output
    offs_output = batch_idx * stride_output_0 + \
        offs_m[:, None] * stride_output_1 + head_idx * \
        stride_output_2 + offs_head_size[None, :]
    tl.store(output + offs_output,
             acc.to(output.type.element_ty),
             mask=offs_m[:, None] < num_query_tokens)
    return


def tree_attention_triton(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    causal_mask: torch.Tensor,
    alibi_slopes: Optional[torch.Tensor] = None,
) -> None:
    assert alibi_slopes is None, "Alibi is currently not supported."
    # Note that triton requires A.shape(0) to be at least 16 in tl.dot(A, B).
    bs, seq_len, num_heads, head_size = query.shape
    num_blocks, block_size, num_kv_heads, k_head_size = key_cache.shape
    assert head_size == k_head_size
    assert key_cache.shape == value_cache.shape
    assert num_heads % num_kv_heads == 0
    output = torch.zeros_like(query)
    scale = 1 / math.sqrt(head_size)

    num_queries_per_kv = num_heads // num_kv_heads
    assert key_cache.shape[1:] == value_cache.shape[1:] == (
        block_size, num_kv_heads, head_size)
    grid = (bs, num_heads, cdiv(seq_len, block_size))

    _fwd_kernel[grid](
        query,  # [bsz, num_query_tokens, num_heads, head_size]
        output,  # [bsz, num_query_tokens, num_heads, head_size]
        key_cache,  # [num_blocks, block_size, num_heads, head_size]
        value_cache,  # [num_blocks, block_size, num_heads, head_size]
        block_tables,  # [bsz, max_num_blocks]
        context_lens,  # [bsz]
        causal_mask,  # [seq_len, seq_len]
        query.stride(0),
        query.stride(1),
        query.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        block_tables.stride(0),
        causal_mask.stride(0),
        head_size,
        seq_len,
        block_size,
        scale,
        num_queries_per_kv,
    )
    return output


if __name__ == "__main__":
    block_size = 16
    num_blocks = 1000
    max_seq_len = 1024
    device = "cuda:0"

    bs = 16
    seq_len = 32
    num_heads = 8
    num_kv_heads = 1
    head_size = 128
    dtype = torch.float16
    query = torch.rand(bs,
                       seq_len,
                       num_heads,
                       head_size,
                       device=device,
                       dtype=dtype)
    key_cache = torch.rand(num_blocks,
                           block_size,
                           num_kv_heads,
                           head_size,
                           device=device,
                           dtype=dtype)
    value_cache = torch.rand(num_blocks,
                             block_size,
                             num_kv_heads,
                             head_size,
                             device=device,
                             dtype=dtype)

    block_tables = torch.zeros(bs,
                               cdiv(max_seq_len, block_size),
                               dtype=torch.int,
                               device=device)
    context_lens = torch.randint(100,
                                 1000,
                                 size=(bs, ),
                                 dtype=torch.int,
                                 device=device)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))

    ref_naive = ref_tree_attention(query, key_cache, value_cache, block_tables,
                                   context_lens, causal_mask)
    ref_flash = flash_attn_with_kvcache(query,
                                        key_cache,
                                        value_cache,
                                        cache_seqlens=context_lens,
                                        block_table=block_tables,
                                        causal=True)
    torch.testing.assert_close(ref_naive, ref_flash, rtol=1e-3, atol=1e-3)

    for _ in range(10):
        random_mask = (torch.rand(seq_len, seq_len, device=device) <
                       0.5) * causal_mask

        ref_naive = ref_tree_attention(query, key_cache, value_cache,
                                       block_tables, context_lens, random_mask)
        out = tree_attention_triton(query, key_cache, value_cache, block_tables,
                                    context_lens, random_mask, None)

        torch.testing.assert_close(ref_naive, out, rtol=1e-3, atol=1e-3)
