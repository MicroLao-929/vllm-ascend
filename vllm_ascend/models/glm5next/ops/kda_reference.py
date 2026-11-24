# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plain-PyTorch KDA recurrence.

Numerical reference for the GLM-5.3-Flash linear attention.  The Triton KDA
kernels in this tree misbehave on the current CANN/Triton stack (the chunked
prefill kernel traps on deep models; the recurrent kernel traps for more than
two tokens), so the model can be driven end to end with this module
(``GLM53_KDA_TORCH=1``).  It mirrors the kernel arithmetic exactly:

    S_t = exp(g_t) * S_{t-1}
    v'_t = v_t - S_t k_t
    S_t += beta_t * k_t v'_t^T
    o_t = scale * S_t q_t

with per-head L2 normalization of q/k and the bounded GLM gate
``g = lower_bound * sigmoid(exp(A) * (raw_g + bias))``.
"""

import torch


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / torch.sqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def safe_gate(
    raw_g: torch.Tensor,
    a_log: torch.Tensor,
    g_bias: torch.Tensor | None,
    lower_bound: float,
) -> torch.Tensor:
    """``lower_bound * sigmoid(exp(A) * (raw_g + bias))`` in fp32."""
    g = raw_g.float()
    heads = a_log.numel()
    a = a_log.reshape(1, 1, heads, 1).float()
    if g_bias is not None:
        g = g + g_bias.reshape(1, 1, heads, -1).float()
    return lower_bound * torch.sigmoid(torch.exp(a) * g)


@torch.no_grad()
def kda_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor | list[int],
    *,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the KDA recurrence over flattened variable-length sequences.

    Args:
        q, k, v: ``[1, T, H, D]`` (bf16/fp32) concatenated tokens.
        g: ``[1, T, H, D]`` fp32 decay (already passed through ``safe_gate``).
        beta: ``[1, T, H, D]`` fp32 beta (already sigmoided).
        initial_state: ``[N, H, D, D]`` fp32, updated in place per sequence.
        cu_seqlens: per-sequence boundaries (list or ``[N + 1]`` tensor).

    Returns:
        ``(output [1, T, H, D] in q dtype, final_state [N, H, D, D] fp32)``.
    """
    bounds = cu_seqlens.tolist() if isinstance(cu_seqlens, torch.Tensor) else list(cu_seqlens)
    num_tokens, heads, dim = q.shape[1], q.shape[2], q.shape[3]
    scale = dim**-0.5 if scale is None else scale
    out = torch.empty_like(q)
    qf, kf, vf = q.float(), k.float(), v.float()
    for seq in range(len(bounds) - 1):
        bos, eos = bounds[seq], bounds[seq + 1]
        state = initial_state[seq].float()
        for t in range(bos, eos):
            qt = _l2norm(qf[0, t]) * scale
            kt = _l2norm(kf[0, t])
            vt = vf[0, t]
            state = state * torch.exp(g[0, t]).unsqueeze(1)
            vt = vt - (state * kt.unsqueeze(1)).sum(dim=-1)
            bt = beta[0, t]
            vt = vt * (bt.unsqueeze(-1) if bt.dim() == 1 else bt)
            state = state + vt.unsqueeze(-1) * kt.unsqueeze(1)
            out[0, t] = (state * qt.unsqueeze(1)).sum(dim=-1).to(out.dtype)
        initial_state[seq] = state.to(initial_state.dtype)
    return out, initial_state


@torch.no_grad()
def kda_decode_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Single-token decode for ``N`` sequences writing ``state_cache`` rows."""
    heads, dim = q.shape[-2], q.shape[-1]
    scale = dim**-0.5 if scale is None else scale
    out = torch.empty_like(q)
    qf = _l2norm(q.float()) * scale
    kf = _l2norm(k.float())
    vf = v.float()
    rows = state_indices.tolist()
    for seq, row in enumerate(rows):
        if row < 0:
            continue
        state = state_cache[row].float()
        state = state * torch.exp(g[0, seq]).unsqueeze(1)
        vt = vf[0, seq] - (state * kf[0, seq].unsqueeze(1)).sum(dim=-1)
        bt = beta[0, seq]
        vt = vt * (bt.unsqueeze(-1) if bt.dim() == 1 else bt)
        state = state + vt.unsqueeze(-1) * kf[0, seq].unsqueeze(1)
        out[0, seq] = (state * qf[0, seq].unsqueeze(1)).sum(dim=-1).to(out.dtype)
        state_cache[row] = state.to(state_cache.dtype)
    return out
