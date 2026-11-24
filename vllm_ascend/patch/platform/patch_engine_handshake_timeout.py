#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Make the front-end <-> EngineCore handshake timeout configurable.
#
# vLLM hard-codes ``HANDSHAKE_TIMEOUT_MINS = 5``.  Every data-parallel engine
# core imports torch/NPU/CANN and loads its weight shard before it answers, so
# on a shared network filesystem a DP-8 start routinely exceeds five minutes
# and the front end aborts with "Did not receive response from front-end
# process within 5 minutes".  Reading the value from the environment keeps the
# upstream default and lets slow deployments (and benchmarks) opt in without
# deriving a model- or hardware-specific special case.

import os

from vllm.logger import logger

_ENV = "VLLM_ASCEND_ENGINE_HANDSHAKE_TIMEOUT_MINS"


def _apply() -> None:
    raw = os.getenv(_ENV)
    if not raw:
        return
    try:
        minutes = float(raw)
    except ValueError as exc:
        raise ValueError(f"{_ENV} must be a number of minutes, got {raw!r}") from exc
    if minutes <= 0:
        raise ValueError(f"{_ENV} must be positive, got {minutes}")
    import vllm.v1.engine.core as engine_core

    engine_core.HANDSHAKE_TIMEOUT_MINS = minutes
    logger.info("Engine handshake timeout set to %s minutes via %s", minutes, _ENV)


_apply()
