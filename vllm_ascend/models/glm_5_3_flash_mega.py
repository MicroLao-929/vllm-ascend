"""GLM-5.3-Flash MegaKernel integration for vLLM-Ascend.

``ENABLE_GLM53_MEGAKERNEL=1`` routes ``Glm5NextForConditionalGeneration``
through the MegaKernel GLM-5.3 W4A4 decode graph
(``blockrt.models.glm_5_3_flash.Glm53FlashMegaKernel``) instead of the
op-by-op Ascend path, mirroring the GLM-5.2 integration in
``vllm_ascend/models/glm_5_2_mega.py``.

Scope of this bridge (deliberately explicit — see ``docs`` in the repo):

* **MK owns all runtime state.**  The MegaKernel request pools (MLA latent
  pages, compressed indexer pages plus tails, KDA conv/temporal state) are
  allocated and advanced by ``blockrt``'s ``MKDecodeState``.  vLLM's own KV /
  mamba caches are allocated as usual but are *not* consulted by the decode
  path, so prefill and decode cannot disagree about cache layout.
* **Every token is one MK graph step.**  ``MKDecodeState`` supports a single
  64-token page, i.e. positions ``0..63``; the prompt is therefore advanced
  through the same decode graph the generated tokens use.  With
  ``--max-model-len 64`` this is the whole serving window.
* **Weights come from the checkpoint, not from vLLM's module tree.**  The
  MegaKernel loader (``WeightLoader``) understands the mixed
  W4A4-MXFP4-expert / W8A8-MXFP8-{KDA,Dense,MLA} layout; re-deriving that
  from vLLM's already-quantized parameters is not equivalent.  The vLLM
  wrapper therefore skips loading its own (unused) parameters.
* **Layer count is fixed by the MegaKernel graph**: 4, 8 or 45.  The weight
  budget of one 96 GB card only allows the full 45-layer model in EP2
  (~85.4 GiB/rank, see ``megakernel/docs/GLM53-W4A4-target-validation.md``),
  which needs an expert-parallel process group vLLM's TP/DP cannot currently
  express.  This bridge therefore runs the 8-layer reduced graph.

Usage::

    ENABLE_GLM53_MEGAKERNEL=1 vllm serve /path/to/GLM-5.3-Flash-FP8-w4a4-kda8 \
        --max-model-len 64 --max-num-seqs 1 --max-num-batched-tokens 64 \
        --enforce-eager --tensor-parallel-size 1 \
        --hf-overrides '{"num_hidden_layers": 8}'

``--hf-overrides`` must keep the layer count in {4, 8, 45}; the MegaKernel
graph structure is built from it.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, Optional

import torch

from vllm.config import VllmConfig
from vllm.logger import logger

from vllm_ascend import envs
from vllm_ascend.models.glm5next.model import (
    Glm5NextForCausalLM,
    Glm5NextForConditionalGeneration,
)

__all__ = [
    "AscendGlm53FlashMegaForCausalLM",
    "register_megakernel_model",
]

# The MegaKernel decode graph captures a single 64-token page; ``MKDecodeState``
# rejects larger RoPE tables because the page table is one page wide.
_MAX_MK_POSITIONS = 64
_SUPPORTED_LAYERS = (4, 8, 45)


def _megakernel_enabled() -> bool:
    return envs.VLLM_ASCEND_GLM53_MEGAKERNEL


def _borrow_index_rope(vllm_model: Any, max_positions: int) -> torch.Tensor:
    """Return vLLM's full indexer RoPE table, sliced to the MK page width.

    The indexer RoPE kernel indexes this table with absolute positions, so a
    complete ``[max_positions, 64]`` table is required (a per-step table would
    be indexed twice and read out of bounds after prefill).  Slicing the first
    ``max_positions`` rows keeps the buffer contiguous.
    """
    layers = getattr(getattr(vllm_model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError(
            "GLM-5.3 MegaKernel: cannot locate vLLM decoder layers for the "
            "indexer RoPE table"
        )
    for layer_id, layer in enumerate(layers):
        self_attn = getattr(layer, "self_attn", None)
        rope = getattr(self_attn, "indexer_rope_emb", None)
        cache = getattr(rope, "cos_sin_cache", None)
        if isinstance(cache, torch.Tensor):
            if (
                cache.ndim != 2
                or cache.shape[1] != 64
                or cache.dtype != torch.bfloat16
                or not cache.is_npu
                or not cache.is_contiguous()
            ):
                raise RuntimeError(
                    "GLM-5.3 MegaKernel indexer RoPE cache must be contiguous "
                    f"NPU BF16 [max_positions, 64]; layer {layer_id} has "
                    f"shape={tuple(cache.shape)} dtype={cache.dtype} "
                    f"contiguous={cache.is_contiguous()}"
                )
            return cache[:max_positions].contiguous()

    # NoPE checkpoints (``qk_rope_head_dim == 0``) build no indexer RoPE at all
    # and the MegaKernel indexer runs with ``indexer_skip_rope``.  The decode
    # state still requires a contiguous BF16 ``[positions, 64]`` placeholder.
    table = torch.zeros(
        (max_positions, 64), dtype=torch.bfloat16, device="npu"
    )
    table[:, :32] = 1.0
    return table


class _Glm53MegaState:
    """Lazily built MegaKernel runtime for one GLM-5.3-Flash model."""

    def __init__(self, vllm_config: VllmConfig, vllm_model: Any):
        from blockrt.models.glm_5_3_flash.glm5_3_flash_megakernel import (
            Glm53FlashMegaKernel,
        )
        from blockrt.models.glm_5_3_flash.model_args import ModelArgsGLM5_3Flash
        from blockrt.models.glm_5_3_flash.weight_loader import WeightLoader

        hf_config = vllm_config.model_config.hf_config
        layers = int(getattr(hf_config, "num_hidden_layers", 0))
        if layers not in _SUPPORTED_LAYERS:
            raise RuntimeError(
                "GLM-5.3 MegaKernel graphs exist for layer counts "
                f"{_SUPPORTED_LAYERS}, got num_hidden_layers={layers}. "
                "Pass --hf-overrides '{\"num_hidden_layers\": 8}'."
            )

        args = ModelArgsGLM5_3Flash()
        args.num_hidden_layers = layers
        args.attention_impl = "mk_decode"
        # W4A4 routed experts with MXFP8 KDA / Dense / shared / MLA projections
        # is the only mixed mode the migrated graph supports (TP1).
        args.mxfp8_decode = True
        args.kda_projection_format = "W8A8_MXFP8"
        args.routed_expert_format = "W4A4_MXFP4"
        args.ep_size = 1
        args.ep_rank = 0
        # NoPE checkpoints (``qk_rope_head_dim == 0``) have no rotary component
        # in the sparse indexer either; the MegaKernel indexer must skip the
        # rotation to match the checkpoint's own configuration.
        args.indexer_skip_rope = int(getattr(hf_config, "qk_rope_head_dim", 0) or 0) == 0
        self.model_args = args

        self.max_positions = min(
            int(getattr(vllm_config.model_config, "max_model_len", _MAX_MK_POSITIONS)),
            _MAX_MK_POSITIONS,
        )
        self.model_dir = os.environ.get(
            "MEGA_WEIGHTS_DIR", vllm_config.model_config.model
        )
        rope = _borrow_index_rope(vllm_model, self.max_positions)

        logger.info(
            "[GLM53MegaKernel] loading checkpoint weights: %s (%d layers)",
            self.model_dir,
            layers,
        )
        weights = WeightLoader(args, self.model_dir).load()
        self.mega = Glm53FlashMegaKernel(
            args,
            weight_dict=weights,
            tp_size=1,
            tp_rank=0,
            index_rope_cache=rope,
        )
        del weights
        self.position = -1
        # The captured graph binds the token tensor by address (the Python
        # Session has no updateTensorAddress), so every step must write into
        # this one persistent 1-token buffer instead of passing a fresh view.
        self.token_buffer = torch.zeros(1, dtype=torch.int64, device="npu")

    def step(self, token_id: torch.Tensor):
        """Advance one token through the MK graph and return its hidden state."""
        position = self.position + 1
        if position == 0:
            batch, positions = self.mega.prepare_initial_decode()
        else:
            batch, positions = self.mega.prepare_next_decode()
        self.token_buffer.copy_(token_id.reshape(-1)[:1].to(torch.int64))
        self.mega.forward(batch, self.token_buffer, positions, None)
        self.position = position
        return self.mega.last_hidden_states

    def reset(self) -> None:
        self.position = -1


class AscendGlm53FlashMegaForCausalLM(Glm5NextForCausalLM):
    """GLM-5.3-Flash model whose token loop runs on the MegaKernel graph.

    The text-only ``Glm5NextForCausalLM`` entry point is used deliberately:
    the multimodal wrapper makes the model runner drive every forward with
    ``inputs_embeds`` and ``input_ids=None``, while the MegaKernel decode graph
    needs token ids (``MKDecodeState.validate`` rejects embeddings).  Serving
    text checkpoints through the text architecture keeps the plain
    ``input_ids`` contract; ``--hf-overrides`` selects it.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.vllm_config = vllm_config
        self._glm53_mega: Optional[_Glm53MegaState] = None
        logger.info(
            "[GLM53MegaKernel] AscendGlm53FlashMegaForConditionalGeneration "
            "initialized (ENABLE_GLM53_MEGAKERNEL=%s)",
            _megakernel_enabled(),
        )

    # ── MegaKernel runtime ────────────────────────────────────────────────
    def _ensure_mega(self) -> _Glm53MegaState:
        if self._glm53_mega is None:
            # vLLM runs forward under ``torch.inference_mode()`` (thread-local).
            # Tensors created there are inference tensors, which the MegaKernel
            # weight loader cannot mutate (it fuses experts on a thread pool).
            # Build the runtime in a fresh thread so every tensor is ordinary.
            worker_device = (
                torch.npu.current_device() if torch.npu.is_available() else None
            )
            result: Dict[str, Any] = {}
            errors: list[BaseException] = []

            def _init() -> None:
                try:
                    if worker_device is not None:
                        torch.npu.set_device(worker_device)
                    result["state"] = _Glm53MegaState(self.vllm_config, self)
                except BaseException as exc:  # noqa: BLE001
                    import traceback

                    traceback.print_exc()
                    errors.append(exc)

            thread = threading.Thread(target=_init)
            thread.start()
            thread.join()
            if errors:
                raise errors[0]
            self._glm53_mega = result["state"]
            if torch.npu.is_available():
                torch.npu.synchronize()
            logger.info("[GLM53MegaKernel] GLM-5.3 mega kernel ready")
        return self._glm53_mega

    # ── vLLM model interface ──────────────────────────────────────────────
    @torch.inference_mode(False)
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if not _megakernel_enabled() or input_ids is None:
            return super().forward(
                input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
            )

        # ``MKDecodeState`` requires view tracking, hence inference_mode(False)
        # plus no_grad for this path (the same contract as GLM-5.2).
        state = self._ensure_mega()
        total = int(input_ids.shape[0])
        # The memory-profiling dummy run feeds ``max_num_batched_tokens`` dummy
        # tokens whose ``positions`` are placeholders (all zeros); the real
        # request path always carries contiguous positions.
        in_profile_run = False
        try:
            from vllm_ascend.ascend_forward_context import _EXTRA_CTX

            in_profile_run = bool(getattr(_EXTRA_CTX, "in_profile_run", False))
        except Exception:  # noqa: BLE001 - context is optional during tests
            in_profile_run = False
        position_list = [] if in_profile_run else positions[:total].tolist()
        if in_profile_run or (total and int(position_list[0]) == 0):
            # A fresh request (or the profiling dummy run): restart the page.
            state.reset()

        hidden_states = []
        for index in range(total):
            if not in_profile_run:
                actual = int(position_list[index])
                if actual != state.position + 1:
                    raise RuntimeError(
                        "GLM-5.3 MegaKernel expects contiguous single-token "
                        f"steps: position {actual} does not follow "
                        f"{state.position} (chunked prefill is not supported; "
                        "keep --max-num-batched-tokens within the 64-token "
                        "page)."
                    )
            token = input_ids[index : index + 1].to(torch.int64).contiguous()
            # The MK graph writes every step into the same output buffer.
            hidden_states.append(state.step(token).clone())
        return torch.cat(hidden_states, dim=0).to(torch.bfloat16)

    @torch.inference_mode(False)
    @torch.no_grad()
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        state = self._glm53_mega
        if state is None or not _megakernel_enabled():
            return super().compute_logits(hidden_states)
        # Apply the MegaKernel lm_head weight so the vLLM wrapper does not need
        # its own (deliberately unloaded) parameter copy.  A plain matmul is
        # used instead of ``MKLMHead.forward``: the eager ``matmul_colmajor``
        # registration only accepts (A, B) while the module passes a third
        # attribute argument, and eager mode never goes through capture.
        weight = state.mega.model.lm_head.weight
        logits = torch.matmul(
            hidden_states.to(torch.bfloat16), weight.t()
        )
        return logits.to(torch.float32)

    def load_weights(self, weights) -> set[str]:
        if envs.VLLM_ASCEND_GLM53_NATIVE_WEIGHTS:
            return super().load_weights(weights)
        # The MegaKernel loads the mixed-precision checkpoint itself; drain the
        # iterator so the loader does not keep shards mapped.
        for _ in weights:
            pass
        logger.info(
            "[GLM53MegaKernel] skipped loading vLLM-native weights "
            "(VLLM_ASCEND_GLM53_NATIVE_WEIGHTS=0)"
        )
        return set()


def register_megakernel_model() -> None:
    """Point the GLM-5.3 text registry entry at the MegaKernel wrapper."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "Glm5NextForCausalLM",
        "vllm_ascend.models.glm_5_3_flash_mega:"
        "AscendGlm53FlashMegaForCausalLM",
    )
