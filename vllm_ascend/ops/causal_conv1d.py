import torch
import torch.nn.functional as F
from vllm.v1.attention.backends.utils import PAD_SLOT_ID


def causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    return_final_states: bool = False,
    final_states_out: torch.Tensor | None = None,
    activation: str | None = "silu",
):
    """
    PyTorch reference implementation of causal_conv1d.

    Args:
        x: (batch, dim, seqlen)
        weight: (dim, width)
        bias: (dim,)
        initial_states: (batch, dim, width - 1)
        final_states_out: (batch, dim, width - 1)
        return_final_states: bool
        activation: str

    Returns:
        out: (batch, dim, seqlen)
        final_states_out: (batch, dim, width - 1) if return_final_states
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape

    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]

    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(dtype_in)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return (out, None) if not return_final_states else (out, final_states_out)


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = "silu",
    conv_states: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    cache_indices: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    pad_slot_id: int = PAD_SLOT_ID,
):
    """
    PyTorch implementation of causal_conv1d_fn.

    Args:
        x: (dim, cu_seq_len) for varlen
        weight: (dim, width)
        bias: (dim,)
        activation: str
        conv_states: (..., dim, width - 1)
        has_initial_state: (batch) bool
        cache_indices: (batch) int32
        query_start_loc: (batch + 1) int32
        pad_slot_id: int

    Returns:
        out: (batch, dim, seqlen)
    """

    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")

    if query_start_loc is None:
        raise RuntimeError("causal_conv1d_fn requires query_start_loc for varlen inputs.")
    if cache_indices is None:
        raise RuntimeError("causal_conv1d_fn requires cache_indices.")
    if has_initial_state is None:
        raise RuntimeError("causal_conv1d_fn requires has_initial_state.")
    if conv_states is None:
        raise RuntimeError("causal_conv1d_fn requires conv_states.")

    if x.stride(-1) != 1:
        x = x.contiguous()
    bias = bias.contiguous() if bias is not None else None

    # Normalize x to [dim, total_tokens]
    if x.dim() == 3:
        if x.shape[0] == 1:
            x = x.squeeze(0)
        elif x.shape[1] == 1:
            x = x.squeeze(1).transpose(0, 1)
        else:
            raise RuntimeError(f"Unsupported x shape for causal_conv1d_fn: {tuple(x.shape)}")
    if x.dim() != 2:
        raise RuntimeError(f"Unsupported x ndim for causal_conv1d_fn: {x.dim()}")

    feature_dim = x.shape[0]
    if weight.shape[0] != feature_dim and weight.shape[1] == feature_dim:
        weight = weight.transpose(0, 1)
    weight = weight.contiguous()
    dim, width = weight.shape
    if dim != feature_dim:
        raise RuntimeError(
            f"causal_conv1d_fn: weight dim mismatch, x dim={feature_dim}, weight.shape={tuple(weight.shape)}"
        )

    state_len = width - 1
    if conv_states.shape[-2] != dim and conv_states.shape[-1] == dim:
        conv_states = conv_states.transpose(-1, -2)
    if conv_states.shape[-2] != dim:
        raise RuntimeError(
            f"causal_conv1d_fn: conv_states dim mismatch, "
            f"expected dim={dim}, conv_states.shape={tuple(conv_states.shape)}"
        )
    if conv_states.shape[-1] < state_len:
        raise RuntimeError(f"causal_conv1d_fn: conv_states too short, need >= {state_len}, got {conv_states.shape[-1]}")

    if __import__("os").environ.get("GLM53_DEBUG_TRACE") == "1":
        import sys
        print(f"[glm53] cc x={tuple(x.shape)} w={tuple(weight.shape)} st={tuple(conv_states.shape)}"
              f" his={has_initial_state.tolist() if has_initial_state.numel() < 8 else has_initial_state.numel()}",
              file=sys.stderr, flush=True)
    seqlens = (query_start_loc[1:] - query_start_loc[:-1]).tolist()
    if __import__("os").environ.get("GLM53_DEBUG_TRACE") == "1":
        import sys
        print(f"[glm53] cc seqlens={seqlens}", file=sys.stderr, flush=True)
    splits = torch.split(x, seqlens, dim=-1)

    out_chunks = []
    import os as _os, sys as _sys
    _dbg = _os.environ.get("GLM53_DEBUG_TRACE") == "1"
    for i, x_s in enumerate(splits):
        if _dbg:
            print(f"[glm53] cc chunk{i} x={tuple(x_s.shape)}", file=_sys.stderr, flush=True)
        cache_idx = int(cache_indices[i].item())
        if cache_idx == pad_slot_id:
            continue

        state = conv_states[cache_idx]
        init_state = state[..., :state_len].unsqueeze(0) if bool(has_initial_state[i].item()) else None
        out_ref, final_state = causal_conv1d_ref(
            x_s.unsqueeze(0),
            weight,
            bias,
            activation=activation,
            return_final_states=True,
            initial_states=init_state,
        )
        state[..., :state_len].copy_(final_state.squeeze(0))
        out_chunks.append(out_ref.squeeze(0))

    if not out_chunks:
        return x.new_zeros((dim, 0))
    return torch.cat(out_chunks, dim=-1)


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    pad_slot_id: int = PAD_SLOT_ID,
):
    """
    PyTorch implementation of causal_conv1d_update.

    Args:
        x: Input tensor
        conv_state: (..., dim, state_len)
        weight: (dim, width)
        bias: (dim,)
        activation: str
        conv_state_indices: (batch,) int32
        num_accepted_tokens: (batch,) int32
        query_start_loc: (batch + 1,) int32
        pad_slot_id: int

    Returns:
        out: same shape as x
    """
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]

    original_x_dtype = x.dtype
    x = x.to(conv_state.dtype)

    feature_dim = x.shape[-1] if query_start_loc is None else x.shape[1]
    if weight.shape[0] != feature_dim and weight.shape[1] == feature_dim:
        weight = weight.transpose(0, 1)
    weight = weight.contiguous()
    dim, width = weight.shape
    if dim != feature_dim:
        raise RuntimeError(
            f"causal_conv1d_update: weight dim mismatch, feature_dim={feature_dim}, weight.shape={tuple(weight.shape)}"
        )

    if conv_state.shape[-2] != dim and conv_state.shape[-1] == dim:
        # Accept both (..., dim, state_len) and (..., state_len, dim) inputs.
        conv_state = conv_state.transpose(-1, -2)
    if conv_state.shape[-2] != dim:
        raise RuntimeError(
            f"causal_conv1d_update: conv_state dim mismatch, "
            f"expected dim={dim}, conv_state.shape={tuple(conv_state.shape)}"
        )

    state_len = width - 1
    if conv_state.shape[-1] < state_len:
        raise RuntimeError(
            f"causal_conv1d_update: conv_state too short, need >= {state_len}, got {conv_state.shape[-1]}"
        )

    out = x.clone()

    def _device_update(x2d: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Host-sync-free update of one token per sequence.

        Plain decode reads one token per request, so the state row can be
        gathered, convolved and scattered with tensor indices only.  PAD rows
        are redirected to slot 0 and then written back unchanged, which keeps
        the operation branch-free and therefore capturable by ACL graphs (a
        ``.item()`` here is what previously forced the whole model to run
        eager).
        """
        valid = idx != pad_slot_id
        safe = torch.where(valid, idx, torch.zeros_like(idx))
        states = conv_state.index_select(0, safe)  # [B, dim, state_len_total]
        history = states[..., :state_len]
        window = torch.cat(
            [history.to(weight.dtype), x2d.to(weight.dtype).unsqueeze(-1)], dim=-1
        )
        conv = F.conv1d(window, weight.unsqueeze(1), bias, groups=dim)
        step = torch.where(
            valid.view(-1, 1, 1), window[..., -state_len:].to(conv_state.dtype), history
        )
        if states.shape[-1] > state_len:
            step = torch.cat([step, states[..., state_len:]], dim=-1)
        conv_state.index_copy_(0, safe, step)
        return conv if activation is None else F.silu(conv)

    if (
        query_start_loc is None
        and num_accepted_tokens is None
        and conv_state_indices is not None
        and x.dim() == 2
    ):
        indices = conv_state_indices[: x.shape[0]].to(torch.int64)
        return _device_update(x, indices).squeeze(-1).to(original_x_dtype)

    def _select_state(i: int) -> torch.Tensor | None:
        if conv_state_indices is not None:
            idx = int(conv_state_indices[i].item())
            if idx == pad_slot_id:
                return None
            state = conv_state[idx]
        else:
            state = conv_state[i]
        return state

    def _run_one(seq_tokens: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # seq_tokens: [L, dim] -> [1, dim, L]
        x_ref = seq_tokens.transpose(0, 1).unsqueeze(0)
        init_state = state[..., :state_len].unsqueeze(0)
        out_ref, final_state = causal_conv1d_ref(
            x_ref,
            weight,
            bias,
            initial_states=init_state,
            return_final_states=True,
            activation=activation,
        )
        state[..., :state_len].copy_(final_state.squeeze(0))
        # [1, dim, L] -> [L, dim]
        return out_ref.squeeze(0).transpose(0, 1)

    def _run_spec_block(seq_tokens: torch.Tensor, state: torch.Tensor, accepted: int) -> torch.Tensor:
        """Slide the convolution window over a draft-verify block.

        Mirrors upstream's spec-decoding rollout: the state row holds
        ``state_len + num_spec`` entries.  The previous step stored
        ``[history, draft tokens]``, so this step reads its ``state_len``
        history entries starting at ``accepted - 1``, runs the *whole* block,
        and stores ``[history after the block, draft tail]`` back.  The next
        step can then rewind to any accepted length inside that window.
        """
        width_entries = state.shape[-1]
        num_spec = width_entries - state_len
        offset = accepted - 1
        if offset < 0 or offset + state_len > width_entries:
            raise RuntimeError(
                f"causal_conv1d_update: accepted={accepted} outside the "
                f"speculative state window (num_spec={num_spec})"
            )
        history = state[..., offset:offset + state_len].unsqueeze(0)
        x_ref = seq_tokens.transpose(0, 1).unsqueeze(0)
        out_ref, final_state = causal_conv1d_ref(
            x_ref,
            weight,
            bias,
            initial_states=history,
            return_final_states=True,
            activation=activation,
        )
        tail = seq_tokens[-num_spec:] if num_spec else seq_tokens[:0]
        rolled = torch.cat([final_state.squeeze(0), tail.transpose(0, 1)], dim=-1)
        state[..., :width_entries].copy_(rolled[..., -width_entries:])
        # [1, dim, L] -> [L, dim]
        return out_ref.squeeze(0).transpose(0, 1)

    def _run_block(seq_tokens: torch.Tensor, state: torch.Tensor, i: int) -> torch.Tensor | None:
        if num_accepted_tokens is None:
            return _run_one(seq_tokens, state)
        accepted = int(num_accepted_tokens[i].item())
        if accepted <= 0:
            return None
        return _run_spec_block(seq_tokens, state, accepted)

    if query_start_loc is None:
        if x.dim() == 2:
            batch = x.shape[0]
            for i in range(batch):
                state = _select_state(i)
                if state is None:
                    continue
                out_i = _run_block(x[i : i + 1], state, i)
                if out_i is not None:
                    out[i : i + out_i.shape[0]] = out_i
        else:
            batch = x.shape[0]
            for i in range(batch):
                state = _select_state(i)
                if state is None:
                    continue
                out_i = _run_block(x[i], state, i)
                if out_i is not None:
                    out[i, : out_i.shape[0]] = out_i
    else:
        assert conv_state_indices is not None
        batch = conv_state_indices.size(0)
        for i in range(batch):
            start = int(query_start_loc[i].item())
            end = int(query_start_loc[i + 1].item())
            if end <= start:
                continue
            state = _select_state(i)
            if state is None:
                continue
            out_i = _run_block(x[start:end], state, i)
            if out_i is not None:
                out[start : start + out_i.shape[0]] = out_i

    return out.to(original_x_dtype)
