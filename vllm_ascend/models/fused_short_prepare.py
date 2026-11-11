"""Single-launch metadata and causal indices for a rectangular BS1 Q8 target."""
from vllm.triton_utils import tl, triton


@triton.jit
def _prepare_q8(
    ids, positions, slots, blocks, seq_len,
    cos_table, sin_table,
    stable_ids, stable_positions, stable_slots, stable_blocks, stable_kv, stable_q,
    stable_cos, stable_sin, short_indices,
    num_blocks: tl.constexpr, block_chunk: tl.constexpr,
):
    task = tl.program_id(0)
    if task < 8:
        position = tl.load(positions + task)
        tl.store(stable_ids + task, tl.load(ids + task))
        tl.store(stable_positions + task, position)
        tl.store(stable_slots + task, tl.load(slots + task))
        cols = tl.arange(0, 64)
        source = position.to(tl.int32) * 64 + cols
        tl.store(stable_cos + task * 64 + cols, tl.load(cos_table + source))
        tl.store(stable_sin + task * 64 + cols, tl.load(sin_table + source))
        visible = tl.load(seq_len).to(tl.int32) - 7 + task
        keys = tl.arange(0, 2048)
        tl.store(short_indices + task * 2048 + keys, tl.where(keys < visible, keys, -1))
    else:
        if task == 8:
            tl.store(stable_kv, tl.load(seq_len))
            tl.store(stable_q, 8)
        block_offsets = (task - 8) * block_chunk + tl.arange(0, block_chunk)
        values = tl.load(blocks + block_offsets, mask=block_offsets < num_blocks, other=0)
        tl.store(stable_blocks + block_offsets, values, mask=block_offsets < num_blocks)


def prepare_q8(ids, positions, slots, blocks, seq_len, cos_table, sin_table, outputs, num_blocks):
    chunk = 256
    _prepare_q8[(8 + max(1, triton.cdiv(num_blocks, chunk)),)](
        ids, positions, slots, blocks, seq_len, cos_table, sin_table, *outputs, num_blocks, chunk,
        num_warps=1,
    )
