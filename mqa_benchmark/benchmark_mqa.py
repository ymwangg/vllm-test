import torch
import triton
import triton.language as tl

from vllm._C import cache_ops
from mqa_vllm_layout import mqa_vllm_layout
from mqa_flash_layout import mqa_flash_layout
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from dataclasses import dataclass, fields
import argparse
import math
import time

# We need to understand how the following variables affect the performance
# block_size
# kv cache layout


def cdiv(a: int, b: int):
    return (a + b - 1) // b


def prepare_kv_cache(num_blocks: int,
                     block_size: int,
                     num_kv_heads: int,
                     head_size: int,
                     key_style: str = "vllm",
                     value_style: str = "vllm",
                     dtype: torch.dtype = torch.float16):
    # key cache
    if key_style == "vllm":
        x = 8
        key_shape = [num_blocks, num_kv_heads, head_size // x, block_size, x]
    elif key_style == "head_first":
        key_shape = [num_blocks, num_kv_heads, block_size, head_size]
    elif key_style == "block_first":
        key_shape = [num_blocks, block_size, num_kv_heads, head_size]
    else:
        raise ValueError("Unknown layout")
    # value cache
    if value_style == "vllm":
        value_shape = [num_blocks, num_kv_heads, head_size, block_size]
    elif value_style == "head_first":
        value_shape = [num_blocks, num_kv_heads, block_size, head_size]
    elif value_style == "block_first":
        value_shape = [num_blocks, block_size, num_kv_heads, head_size]
    else:
        raise ValueError("Unknown layout")
    key_cache = torch.randn(*key_shape, dtype=dtype).cuda()
    value_cache = torch.randn(*value_shape, dtype=dtype).cuda()
    return key_cache, value_cache


def prepare_inputs(batch_size: int = 16,
                   seq_len: int = 6,
                   num_query_heads: int = 8,
                   head_size: int = 128,
                   max_ctx_len: int = 2048,
                   ctx_len_style: str = "uniform",
                   block_table_style: str = "random",
                   block_size: int = 16,
                   num_blocks: int = 1024,
                   dtype: torch.dtype = torch.float16):
    query_shape = [batch_size, seq_len, num_query_heads, head_size]
    query = torch.rand(query_shape, dtype=dtype).cuda()
    assert max_ctx_len >= seq_len
    if ctx_len_style == "uniform":
        context_lens = torch.full(size=(batch_size, ), fill_value=max_ctx_len)
    elif ctx_len_style == "random":
        context_lens = torch.randint(seq_len + 1,
                                     max_ctx_len,
                                     size=(batch_size, ))
    else:
        raise ValueError("Unknown ctx_len_style")
    num_blocks_used = cdiv(max_ctx_len, block_size)
    if block_table_style == "random":
        block_tables = torch.randint(0,
                                     num_blocks_used,
                                     size=(batch_size, num_blocks_used))
    elif block_table_style == "sequential":
        block_tables = []
        for idx, ctx_len in enumerate(context_lens):
            block_tables.append([
                offset // block_size +
                (idx * num_blocks_used) % num_blocks if offset < ctx_len else -1
                for offset in range(max_ctx_len)
            ])
        block_tables = torch.tensor(block_tables, dtype=torch.int64)
    else:
        raise ValueError("Unknown block_table_style")
    context_lens = context_lens.cuda()
    block_tables = block_tables.cuda()
    return query, context_lens, block_tables


@dataclass
class BenchmarkConfig:
    batch_size: int
    ctx_len: int
    num_kv_heads: int
    num_query_heads: int
    head_size: int

    @classmethod
    def from_cmd_args(cls):
        parser = argparse.ArgumentParser(description="Benchmark Configuration")

        for field in fields(cls):
            parser.add_argument(f'--{field.name}',
                                type=field.type,
                                help=f'{field.name} ({field.type.__name__})')

        args = parser.parse_args()
        return cls(**vars(args))


def ref_flash_attn_with_kvcache(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
) -> None:
    out = flash_attn_with_kvcache(
        query,
        key_cache,
        value_cache,
        cache_seqlens=context_lens,
        block_table=block_tables,
        causal=True,
    )
    return out


def ref_flash_attn_varlen_func(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
) -> None:

    num_tokens, num_heads, head_size = query.shape
    bs = context_lens.shape[0]
    seq_len = num_tokens // bs
    assert num_heads % num_kv_heads == 0
    x = key_cache.shape[-1]
    block_size = value_cache.shape[3]
    max_context_len = 2048
    # Allocate key and value entries
    kv_slot_mapping = []
    for i, context_len in enumerate(context_lens):
        block_table = block_tables[i]
        for position in range(context_len):
            block_number = block_table[position // block_size]
            block_offset = position % block_size
            slot = block_number * block_size + block_offset
            kv_slot_mapping.append(slot)
    kv_slot_mapping = torch.tensor(kv_slot_mapping,
                                   dtype=torch.int,
                                   device="cuda")
    num_kv_tokens = kv_slot_mapping.shape[0]
    key = torch.zeros(num_kv_tokens,
                      num_kv_heads,
                      head_size,
                      dtype=query.dtype,
                      device=query.device)
    value = torch.zeros(num_kv_tokens,
                        num_kv_heads,
                        head_size,
                        dtype=query.dtype,
                        device=query.device)

    # Call vLLM kernel to collect key and value entries
    cache_ops.gather_cached_kv(key, value, key_cache, value_cache,
                               kv_slot_mapping)
    cu_seqlens_q = torch.arange(0, (bs + 1) * seq_len,
                                step=seq_len,
                                dtype=torch.int32,
                                device="cuda")
    cu_seqlens_k = torch.cumsum(torch.tensor([0] + context_lens.tolist(),
                                             dtype=torch.int,
                                             device="cuda"),
                                dim=0,
                                dtype=torch.int)
    # Call flashattention v2
    output = flash_attn_varlen_func(
        query.view(-1, num_heads, head_size),
        key,
        value,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q=seq_len,
        max_seqlen_k=max_context_len,
        dropout_p=0.0,
        softmax_scale=scale,
        causal=
        True,  # checked that flashattention uses our expected causal masking from v2.1
    )
    return output


def test_cached_mqa(
    batch_size,
    seq_len,
    head_size,
    num_query_heads,
    num_kv_heads,
):
    block_size = 16
    max_ctx_len = 1024
    num_blocks = cdiv(max_ctx_len * batch_size, block_size)
    key_cache, value_cache = prepare_kv_cache(num_blocks, block_size,
                                              num_kv_heads, head_size)
    query, context_lens, block_tables = prepare_inputs(
        batch_size=batch_size,
        seq_len=seq_len,
        num_query_heads=num_query_heads,
        head_size=head_size,
        max_ctx_len=max_ctx_len,
        ctx_len_style="random",
        block_table_style="sequential",
        block_size=block_size,
        num_blocks=num_blocks)
    query = query.view(-1, num_query_heads, head_size)
    scale = 1.0 / math.sqrt(head_size)
    # block_tables = torch.randint(0, num_blocks,
    #                              size=(bs, max_num_blocks)).cuda()
    # context_lens = torch.randint(100, 250, size=(bs, )).cuda()
    res = torch.empty_like(query)

    mqa_vllm_layout(
        res,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        context_lens,
    )

    ref_res = ref_flash_attn_varlen_func(
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        context_lens,
    )
    torch.testing.assert_close(res, ref_res, rtol=1e-3, atol=1e-3)


def benchmark(
    batch_size: int = 16,
    seq_len: int = 6,
    head_size: int = 128,
    num_query_heads: int = 8,
    num_kv_heads: int = 8,
    block_size: int = 16,
    max_ctx_len: int = 1024,
):
    num_blocks = cdiv(max_ctx_len * batch_size, block_size)
    key_cache, value_cache = prepare_kv_cache(num_blocks, block_size,
                                              num_kv_heads, head_size)
    query, context_lens, block_tables = prepare_inputs(
        batch_size=batch_size,
        seq_len=seq_len,
        num_query_heads=num_query_heads,
        head_size=head_size,
        max_ctx_len=max_ctx_len,
        ctx_len_style="random",
        block_table_style="sequential",
        block_size=block_size,
        num_blocks=num_blocks)
    query = query.view(-1, num_query_heads, head_size)
    scale = 1.0 / math.sqrt(head_size)
    output = torch.empty_like(query)

    def fn1():
        mqa_vllm_layout(
            output,
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            block_tables,
            context_lens,
        )

    def fn2():
        ref_flash_attn_varlen_func(
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            block_tables,
            context_lens,
        )

    def benchmark(fn, n_warm=2, n_steps=10):
        for _ in range(n_warm):
            fn()
        torch.cuda.synchronize()
        t0 = time.monotonic()
        for _ in range(n_steps):
            fn()
        torch.cuda.synchronize()
        t1 = time.monotonic()
        return (t1 - t0) / n_steps

    print(benchmark(fn1), benchmark(fn2))


def validate(
    batch_size: int = 16,
    seq_len: int = 6,
    head_size: int = 128,
    num_query_heads: int = 8,
    num_kv_heads: int = 8,
    block_size: int = 16,
    max_ctx_len: int = 1024,
):
    num_blocks = cdiv(max_ctx_len * batch_size, block_size)
    key_cache, value_cache = prepare_kv_cache(num_blocks, block_size,
                                              num_kv_heads, head_size)
    query, context_lens, block_tables = prepare_inputs(
        batch_size=batch_size,
        seq_len=seq_len,
        num_query_heads=num_query_heads,
        head_size=head_size,
        max_ctx_len=max_ctx_len,
        ctx_len_style="random",
        block_table_style="sequential",
        block_size=block_size,
        num_blocks=num_blocks)
    scale = 1.0 / math.sqrt(head_size)
    output = torch.empty_like(query)
    mqa_vllm_layout(
        output,
        query.view(-1, num_query_heads, head_size),
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        context_lens,
    )
    # [num_blocks, num_kv_heads, head_size // x, block_size, x] to
    # (num_blocks, page_block_size, nheads_k, headdim)
    flash_key_cache = key_cache.permute(0, 3, 1, 2,
                                        4).reshape(num_blocks, block_size,
                                                   num_kv_heads,
                                                   head_size).contiguous()
    # [num_blocks, num_kv_heads, head_size, block_size] to
    # (num_blocks, page_block_size, nheads_k, headdim)
    flash_value_cache = value_cache.permute(0, 3, 1, 2).contiguous()
    assert flash_key_cache.shape == flash_value_cache.shape == (
        num_blocks, block_size, num_kv_heads, head_size)
    ref_output = ref_flash_attn_with_kvcache(query, flash_key_cache,
                                             flash_value_cache,
                                             block_tables.to(torch.int32),
                                             context_lens.to(torch.int32))
    torch.testing.assert_close(output, ref_output, atol=1e-3, rtol=1e-3)
    mqa_flash_layout(
        output,
        query.view(-1, num_query_heads, head_size),
        flash_key_cache,
        flash_value_cache,
        num_kv_heads,
        scale,
        block_tables,
        context_lens,
    )
    torch.testing.assert_close(output, ref_output, atol=1e-3, rtol=1e-3)

    def fn1():
        mqa_vllm_layout(
            output,
            query.view(-1, num_query_heads, head_size),
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            block_tables,
            context_lens,
        )
    def fn2():
        mqa_flash_layout(
            output,
            query.view(-1, num_query_heads, head_size),
            flash_key_cache,
            flash_value_cache,
            num_kv_heads,
            scale,
            block_tables,
            context_lens,
        )

    def fn3():
        ref_output = ref_flash_attn_with_kvcache(query, flash_key_cache,
                                                 flash_value_cache,
                                                 block_tables.to(torch.int32),
                                                 context_lens.to(torch.int32))

    def benchmark(fn, n_warm=2, n_steps=100):
        for _ in range(n_warm):
            fn()
        torch.cuda.synchronize()
        t0 = time.monotonic()
        for _ in range(n_steps):
            fn()
        torch.cuda.synchronize()
        t1 = time.monotonic()
        return (t1 - t0) / n_steps

    print(benchmark(fn1), benchmark(fn2), benchmark(fn3))


if __name__ == "__main__":
    # test_cached_mqa(1, 5, 64, 12, 12)
    # test_cached_mqa(1, 5, 64, 12, 4)
    # test_cached_mqa(24, 5, 128, 12, 4)
    # test_cached_mqa(24, 5, 128, 12, 12)
    # test_cached_mqa(24, 16, 128, 4, 1)
    # benchmark()
    validate(batch_size=4, num_query_heads=8, num_kv_heads=1)
