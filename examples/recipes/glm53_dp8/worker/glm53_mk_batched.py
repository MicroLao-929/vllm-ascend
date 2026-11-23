"""Exact batching of the MK bridge's many small device-to-device writebacks.

The v0.30 ordinary bridge commits one row/state slice per cache layer on every
decode token: 34 KDA conv + 34 KDA recurrent tensors plus 11 MLA rows and the
11 Indexer pool/tail rows.  Each ``Tensor.copy_`` is a separate dispatcher
entry and device launch on the host critical path.  ``torch._foreach_copy_``
performs the same element-wise copies in one call; this module validates every
pair first and falls back to the original per-pair loop whenever the fused op
cannot be used.

Safety rules (deliberately enforced here):

* A pair is only batched when both sides are tensors with the *same shape* and
  the *same device*.  ``copy_`` and ``_foreach_copy_`` both cast dtype, so a
  differing element type is fine (the bridge relies on BF16->FP32 casts).
* Sources are bridge/MK-owned state that the commit never mutates, and
  destinations are the VA-owned cache.  Re-running the plain loop after a
  partial fused failure therefore writes exactly the same values again; no
  write is ever skipped and no value is invented.
* No tensor is read on the host and no metadata is synthesised here.
"""

import torch


def _pair_ok(dst, src):
    return (
        isinstance(dst, torch.Tensor)
        and isinstance(src, torch.Tensor)
        and dst.shape == src.shape
        and dst.device == src.device
        and dst.device.type != "meta"
        and src.device.type != "meta"
    )


def copy_pairs(pairs, *, min_pairs=2):
    """Copy ``(dst, src)`` pairs, returning True when the fused op ran.

    A return value of False means the caller must run its original loop.  The
    check is conservative: any unusual tensor, or any exception from the fused
    op, selects the loop.  The fused op is never attempted for fewer than
    ``min_pairs`` pairs because per-call dispatch would dominate.
    """
    items = tuple((dst, src) for dst, src in pairs)
    if len(items) < min_pairs or any(not _pair_ok(dst, src) for dst, src in items):
        return False
    destinations = [dst for dst, _ in items]
    sources = [src for _, src in items]
    try:
        torch._foreach_copy_(destinations, sources)
    except Exception:
        # The fused kernel may have written a prefix of the pairs before it
        # failed.  Every source is immutable during a commit, so the caller's
        # plain loop is idempotent and restores the intended values.
        return False
    return True
