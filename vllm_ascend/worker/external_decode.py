#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""Generic hook for an externally supplied decode implementation.

vllm-ascend does not host model-specific external decoders.  When
``VLLM_ASCEND_EXTERNAL_DECODE_ADAPTOR`` names an importable factory
(``package.module:factory``), the runner asks that adaptor whether it can
execute the scheduled single-token decode step.  Returning ``None`` keeps the
native vLLM model path, so the hook is a pure optimisation/fast path.

The adaptor owns everything model-specific: checkpoint layout, captured graph,
expert-parallel topology and cache import/commit.  The contract below is the
only coupling; it is intentionally small so any backend (MegaKernel, another
runtime) can implement it without touching vllm-ascend.

Adaptor protocol::

    update_requests(runner, scheduler_output) -> None
        Observe scheduling decisions (finished/new/preempted requests).

    try_decode(runner, scheduler_output, forward_context, positions, input_ids)
        -> step | None
        ``step`` exposes ``hidden_states`` (``[num_tokens, hidden]``) and an
        optional ``logits`` tensor.  ``None`` means "use the native model".

    process_logits(runner, step, logits_indices) -> torch.Tensor
        Produce sampling logits for ``logits_indices`` when the step carried
        its own logits.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Optional, Protocol

import torch

from vllm.logger import logger

from vllm_ascend import envs


class ExternalDecodeStep(Protocol):
    """One scheduled decode step produced outside the vLLM model."""

    hidden_states: torch.Tensor
    logits: Optional[torch.Tensor]


class ExternalDecodeAdaptor(Protocol):
    """Minimal surface the model runner calls into."""

    def update_requests(self, runner: Any, scheduler_output: Any) -> None: ...

    def try_decode(
        self,
        runner: Any,
        scheduler_output: Any,
        forward_context: Any,
        positions: torch.Tensor,
        input_ids: Optional[torch.Tensor],
    ) -> Optional[ExternalDecodeStep]: ...

    def process_logits(
        self, runner: Any, step: ExternalDecodeStep, logits_indices: torch.Tensor
    ) -> torch.Tensor: ...


@dataclass
class NativeLogitsStep:
    """Step that only supplies hidden states and defers to the native head."""

    hidden_states: torch.Tensor
    logits: Optional[torch.Tensor] = None


def load_external_decode(runner: Any) -> Optional[ExternalDecodeAdaptor]:
    """Import and build the configured adaptor, or return ``None``.

    A failure to import or build is raised: a deployment that asked for an
    external decoder must not silently fall back to the native path.
    """
    spec = envs.VLLM_ASCEND_EXTERNAL_DECODE_ADAPTOR
    if not spec:
        return None
    module_name, _, factory_name = spec.partition(":")
    factory_name = factory_name or "build_adaptor"
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name, None)
    if factory is None:
        raise ValueError(
            f"external decode adaptor {spec!r}: {module_name} has no attribute "
            f"{factory_name!r}"
        )
    adaptor = factory(runner)
    for method in ("update_requests", "try_decode", "process_logits"):
        if not callable(getattr(adaptor, method, None)):
            raise ValueError(
                f"external decode adaptor {spec!r} does not implement {method}()"
            )
    logger.info("[external-decode] adaptor %s loaded", spec)
    return adaptor
