"""Causal short-context indices for an unpadded single-request verify."""
import triton
import triton.language as tl


@triton.jit
def _write_short_context_indices(kv_seq_len, indices, rows: tl.constexpr,
                                 width: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.program_id(1) * block + tl.arange(0, block)
    valid = tl.load(kv_seq_len) - rows + row + 1
    tl.store(indices + row * width + offsets, tl.where(offsets < valid, offsets, -1),
             offsets < width)


def write_short_context_indices(kv_seq_len, indices):
    _, rows, _, width = indices.shape
    block = 512
    _write_short_context_indices[(rows, triton.cdiv(width, block))](
        kv_seq_len, indices, rows, width, block
    )
