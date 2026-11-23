"""Publish native GPU block tables only if a committed MK continuation needs them."""


def eligible(runner, hook, output, deferred_corrections):
    """The existing address plan and eight-rank vote still decide whether MK runs."""
    try:
        batch, parallel = runner.input_batch, runner.parallel_config
        early, bridge = hook.early_runner, hook.bridge
        if (deferred_corrections is not None or early is None or hook.native_builder_deferral != "on"
                or early.rollout.mode != "on" or not early.ledger.shadow_clean
                or early.ledger.shadow_steps < max(1, early.rollout.min_shadow)
                or hook.tp_size != 1 or hook.plan.size != 8 or hook.plan.depth != 45
                or (parallel.tensor_parallel_size, parallel.data_parallel_size,
                    parallel.pipeline_parallel_size, parallel.prefill_context_parallel_size,
                    parallel.decode_context_parallel_size) != (1, 8, 1, 1, 1)
                or runner.max_num_reqs != 1 or batch.num_reqs != 1
                or not runner.model_config.enforce_eager or runner.model_config.is_encoder_decoder
                or runner.use_async_scheduling or runner.use_async_spec_decode or runner.prepare_inputs_event is not None
                or runner.speculative_config is not None or runner.use_dcp or runner.use_compress or runner.dynamic_eplb
                or runner.cache_config.enable_prefix_caching or runner.cache_config.mamba_cache_mode != "none"
                or runner.device_metadata_executor is not None
                or runner.kvpp.scheduler is not None or runner.long_seq_metadata is not None
                or runner.sparse_kv_offload_enabled or runner.is_mm_prefix_lm or runner.supports_mm_inputs
                or hook.invalid or hook.active is not None or hook.verified_graph is None
                or bridge._poisoned or bridge._graph is not hook.verified_graph):
            return False
        rid = batch.req_ids[0]
        position = int(batch.num_computed_tokens_cpu[0])
        return (0 < int(batch.num_prompt_tokens[0]) < position < bridge.max_context_tokens
                and (hook.request_id, hook.next_position) == (rid, position)
                and hook._last_completed_round == (rid, position - 1)
                and (bridge._request, bridge._position) == (rid, position - 1)
                and output.total_num_scheduled_tokens == 1 and output.num_scheduled_tokens == {rid: 1}
                and not output.scheduled_spec_decode_tokens and not output.scheduled_encoder_inputs
                and not output.scheduled_new_reqs and not output.finished_req_ids
                and not output.scheduled_cached_reqs.resumed_req_ids
                and not output.preempted_req_ids and not output.block_table_updates
                and not output.kv_cache_block_copies and not output.new_block_ids_to_zero
                and not output.has_sync_kv_loads)
    except (AttributeError, KeyError, IndexError, TypeError, ValueError):
        return False


class DeferredBlockTables:
    """Single-step native replay. The runner always builds normal token and position buffers."""

    def __init__(self, tables):
        self.tables = tables
        self.mapping = None
        self.attempted = False
        self.restored = False

    def save_mapping(self, num_reqs, query, positions):
        if num_reqs != 1 or self.mapping is not None:
            raise RuntimeError("unexpected DP8 lazy slot mapping")
        self.mapping = (num_reqs, query, positions)

    def materialize(self):
        if self.restored:
            return
        if self.attempted or self.mapping is None:
            raise RuntimeError("incomplete DP8 lazy slot mapping")
        self.attempted = True
        self.tables.commit_block_table(self.mapping[0])
        self.tables.compute_slot_mapping(*self.mapping)
        self.restored = True


def bind_native_result(context, result):
    """Make the real attention builders' output visible to late native execution."""
    context.attn_metadata = result[0]
    return result
