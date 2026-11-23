"""Single-request GLM-5.3-Flash VA prefill to MK decode cache handoff."""

import inspect
import os
import logging
from dataclasses import dataclass, field
from threading import Lock

import torch


_LOG = logging.getLogger(__name__)
_PAGE = 64
_SCRATCH_PAGE = 1
_RATIO = 4
_CONV = (3, 24576)
_RECURRENT = (64, 128, 128)
_LATENT = 512
_INDEX = 128


@dataclass(frozen=True)
class KDACache:
    prefix: str
    conv: torch.Tensor
    recurrent: torch.Tensor
    conv_dim_first: bool


@dataclass(frozen=True)
class MLACache:
    prefix: str
    latent: torch.Tensor


@dataclass(frozen=True)
class IndexerCache:
    pool_prefix: str
    tail_prefix: str
    pool: torch.Tensor
    tail: torch.Tensor


@dataclass(frozen=True)
class _NativePages:
    ids: tuple[int, ...]
    rows_per_id: int
    capacity: int | None = None

    def slot(self, row):
        page, offset = divmod(row, self.rows_per_id)
        return self.ids[page] * self.rows_per_id + offset

    def extends(self, previous):
        return self is previous or (self.rows_per_id == previous.rows_per_id
                                   and self.ids[:len(previous.ids)] == previous.ids)


@dataclass(frozen=True)
class _IndexerAddress:
    pool_page: int | None
    pool_row: int | None
    tail_page: int
    history_rows: tuple[int, ...]
    tail_row: int | None
    pages: _NativePages = field(repr=False)


@dataclass(frozen=True)
class DecodeResult:
    """The caller must consume these MK-owned tensors before the next step."""

    logits: torch.Tensor | None
    hidden_states: torch.Tensor
    imported: bool
    graph_reused: bool


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class DecodePageLayout:
    """Disjoint latent and KPool table rows indexed by persistent request slot."""

    def __init__(self, max_context_tokens, request_slots=1):
        _require(type(max_context_tokens) is int and max_context_tokens > 0
                 and type(request_slots) is int and 1 <= request_slots <= 128, "invalid decode page layout")
        width = (max_context_tokens + _PAGE - 1) // _PAGE
        self.num_pages = _SCRATCH_PAGE + request_slots * width
        _require(self.num_pages <= 4096, "MK bridge exceeds the 4096 physical page capacity including padding")
        self.tables = tuple(tuple(range(start, start + width))
                            for start in range(_SCRATCH_PAGE, self.num_pages, width))


def _metadata_values(tensor, length, field):
    _require(isinstance(tensor, torch.Tensor) and tensor.ndim == 1 and tensor.shape[0] == length
             and tensor.dtype in (torch.int32, torch.int64), f"invalid {field}")
    return tuple(tensor.tolist())


def _value(tensor, field):
    return _metadata_values(tensor, 1, field)[0]


def _active_pages(table, count, rows_per_id, total_rows):
    _require(type(rows_per_id) is int and rows_per_id > 0 and total_rows % rows_per_id == 0,
             "invalid VA block granularity")
    columns = (count + rows_per_id - 1) // rows_per_id
    _require(table.ndim == 2 and table.shape[0] == 1 and table.shape[1] >= columns
             and table.dtype in (torch.int32, torch.int64), "invalid VA block table")
    ids = tuple(table[0, :columns].tolist())
    _require(all(0 <= page < total_rows // rows_per_id for page in ids), "VA page is out of range")
    _require(len(set(ids)) == len(ids), "duplicate active VA page in one request")
    return _NativePages(ids, rows_per_id)


class _DecodeMetadata:
    """Read and validate shared VA metadata within one decode step."""

    def __init__(self):
        self._reads = {}
        self._pages = {}

    def values(self, tensor, length, field):
        _require(isinstance(tensor, torch.Tensor) and tensor.ndim == 1 and tensor.shape[0] == length
                 and tensor.dtype in (torch.int32, torch.int64), f"invalid {field}")
        return self._read(tensor, None)

    def value(self, tensor, field):
        return self.values(tensor, 1, field)[0]

    def row(self, tensor, length, field):
        _require(isinstance(tensor, torch.Tensor) and tensor.ndim == 2 and tensor.shape[0] == 1
                 and tensor.shape[1] >= length and tensor.dtype in (torch.int32, torch.int64),
                 f"invalid {field}")
        return self._read(tensor, length)

    def active_pages(self, table, count, rows_per_id, total_rows, *, previous=None):
        _require(type(rows_per_id) is int and rows_per_id > 0 and total_rows % rows_per_id == 0,
                 "invalid VA block granularity")
        columns = (count + rows_per_id - 1) // rows_per_id
        key = (id(table), columns, rows_per_id, total_rows)
        pages = self._pages.get(key)
        if pages is None:
            # row() retains the table for the lifetime of this identity key.
            ids = self.row(table, columns, "VA block table")
            capacity = total_rows // rows_per_id
            if (previous is not None and previous.rows_per_id == rows_per_id
                    and previous.capacity == capacity and ids == previous.ids):
                pages = previous
            else:
                _require(all(0 <= page < capacity for page in ids), "VA page is out of range")
                _require(len(set(ids)) == len(ids), "duplicate active VA page in one request")
                pages = _NativePages(ids, rows_per_id, capacity)
            self._pages[key] = pages
        return pages

    def _read(self, tensor, columns):
        # Keep the source alive so its identity cannot be reused within this step.
        key = (id(tensor), columns)
        entry = self._reads.get(key)
        if entry is None:
            view = tensor if columns is None else tensor[0, :columns]
            entry = (tensor, tuple(view.tolist()))
            self._reads[key] = entry
        return entry[1]

    def prefetch(self, entries):
        """Read many native metadata tensors with as few host transfers as possible.

        ``entries`` is an iterable of ``(tensor, columns)`` matching what
        :meth:`value`/`values`/``row`` will ask for in this step (``columns``
        is None for a one-dimensional field).  Only entries whose shape and
        dtype match the lazy reader are staged; anything else keeps the
        original per-tensor ``tolist`` path.  Values are the exact device
        contents: the same tensors are concatenated and copied to the host
        once per dtype instead of once per field.
        """
        pending = []
        seen = set()
        for tensor, columns in entries:
            if not isinstance(tensor, torch.Tensor) or tensor.device.type == "meta":
                continue
            key = (id(tensor), columns)
            if key in self._reads or key in seen:
                continue
            if columns is None:
                if tensor.ndim != 1 or tensor.dtype not in (torch.int32, torch.int64):
                    continue
                view = tensor
            else:
                if (tensor.ndim != 2 or tensor.shape[0] != 1 or tensor.shape[1] < columns
                        or tensor.dtype not in (torch.int32, torch.int64)):
                    continue
                view = tensor[0, :columns]
            seen.add(key)
            pending.append((key, tensor, view.reshape(-1)))
        if not pending:
            return
        groups = {}
        for item in pending:
            groups.setdefault((item[2].device, item[2].dtype), []).append(item)
        staged = {}
        try:
            for items in groups.values():
                host = torch.cat([flat for _, _, flat in items]).cpu().tolist()
                offset = 0
                for key, tensor, flat in items:
                    width = flat.numel()
                    staged[key] = (tensor, tuple(host[offset:offset + width]))
                    offset += width
        except Exception:
            # A partial batch is useless to the caller; leave the lazy path.
            return
        self._reads.update(staged)


def _copy_prefix(source, native, target, count, mk_pages, mk_page_stride):
    row = 0
    while row < count:
        src_page, src_offset = divmod(native.slot(row), source.shape[1])
        mk_page, mk_offset = divmod(row, _PAGE)
        n = min(count - row, native.rows_per_id - row % native.rows_per_id,
                source.shape[1] - src_offset, _PAGE - mk_offset)
        target[mk_pages[mk_page * mk_page_stride], mk_offset:mk_offset + n].copy_(
            source[src_page, src_offset:src_offset + n])
        row += n


def _copy_prefix_runs(source, native, target, count, mk_pages, mk_page_stride):
    if not source.is_contiguous() or any(p != mk_pages[0] + i for i, p in enumerate(mk_pages)):
        return _copy_prefix(source, native, target, count, mk_pages, mk_page_stride)
    if not count:
        return

    # DecodePageLayout gives MK a contiguous range of physical pages. Pool
    # rows use every fourth MK page within that range.
    rows = source.view(-1, *source.shape[2:])
    first_mk_page = mk_pages[0]
    rows_per_id = native.rows_per_id
    run_start = 0
    for run_end in range(1, len(native.ids) + 1):
        if run_end < len(native.ids) and native.ids[run_end] == native.ids[run_end - 1] + 1:
            continue
        row = run_start * rows_per_id
        end = min(count, run_end * rows_per_id)
        source_row = native.ids[run_start] * rows_per_id
        while row < end:
            logical_page, offset = divmod(row, _PAGE)
            page = first_mk_page + logical_page * mk_page_stride
            whole_pages = (end - row) // _PAGE if not offset else 0
            if whole_pages:
                n = whole_pages * _PAGE
                payload = rows[source_row:source_row + n].view(whole_pages, _PAGE, *source.shape[2:])
                target[page:page + whole_pages * mk_page_stride:mk_page_stride].copy_(payload)
            else:
                n = min(end - row, _PAGE - offset)
                target[page, offset:offset + n].copy_(rows[source_row:source_row + n])
            row += n
            source_row += n
        run_start = run_end


def _row_cache(tensor, inner_shape, dtype, device):
    _require(tensor.ndim == len(inner_shape) + 2 and tensor.shape[2:] == inner_shape
             and tensor.dtype == dtype and tensor.device == device, "invalid VA row cache layout")
    width = inner_shape[-1]
    _require(tensor.stride(-1) == 1 and tensor.stride(1) == width
             and tensor.stride(0) >= tensor.shape[1] * width, "invalid VA cache strides")


def _layout(tensor):
    return (tensor.data_ptr(), tuple(tensor.shape), tensor.stride())


class VAMKBridge:
    """Decode a single non-speculative request on the VA model stream.

    Pass the real per-layer tensors from VA's cache owners and the active
    ForwardContext before the native model has executed the decode token.
    The worker must serialize prefill, this call and sampling on the same
    stream, and select this path only after native prefill for the request.
    Device-produced metadata must be ready on that stream before the call.
    """

    def __init__(self, model, *, ep2=False, ep_size=None, max_context_tokens=_PAGE,
                 batch_native_metadata=False, reuse_cache_views=False,
                 batch_writeback=False, reuse_indexer_tail=False):
        args = model.model_args
        if ep_size is None:
            ep_size = 2 if ep2 else 1
        depths = {1: 4, 2: 8, 6: 45, 8: 45}
        _require(ep_size in depths and (not ep2 or ep_size == 2), "unsupported VA/MK bridge profile")
        tp_size, tp_rank = model.tp_size, model.tp_rank
        depth = args.num_hidden_layers
        shared_experts = getattr(model, "native_routed_layers", ())
        valid_profile = ((tp_size, tp_rank) == (1, 0) and depth == depths[ep_size]
                         or tp_size == ep_size and tp_size in (2, 8) and tp_rank == args.ep_rank
                         and (depth in (4, 8) or depth == 45 and shared_experts == tuple(range(3, 45))))
        _require(valid_profile and args.ep_size == ep_size
                 and 0 <= args.ep_rank < ep_size and args.attention_impl == "mk_decode"
                 and args.num_hidden_layers == depth
                 and tuple(args.linear_layer_ids) == tuple(i for i in range(depth) if i % 4 != 3)
                 and tuple(args.full_attention_layer_ids) == tuple(range(3, depth, 4))
                 and args.indexer_skip_rope and args.index_kpool == _RATIO,
                 "VA bridge requires a supported TP/EP profile with NoPE and KPool=4")
        self._conv_shape = (_CONV[0], _CONV[1] // tp_size)
        self._recurrent_shape = (_RECURRENT[0] // tp_size, *_RECURRENT[1:])
        self._model = model
        self._state = model._mk_decode_state
        _require(self._state is not None and self._state._shared_pools is None
                 and self._state.batch.index_cos_sin_cache.shape[0] == _PAGE,
                 "VA bridge requires unbound MK state with a 64-row NoPE table")
        self._device = self._state.positions.device
        layout = DecodePageLayout(max_context_tokens)
        if max_context_tokens > _PAGE:
            _require((tp_size, ep_size) in ((1, 6), (1, 8), (2, 2), (8, 8)),
                 "paged VA bridge requires TP1/EP6 or EP8, TP2/EP2 or TP8/EP8")
        self.max_context_tokens = max_context_tokens
        num_pages = layout.num_pages
        self._page_table = layout.tables[0]
        with torch.inference_mode(False), torch.no_grad():
            self._token = torch.empty(1, dtype=torch.int64, device=self._device)
            self._mla = {layer: torch.zeros((num_pages, _PAGE, 1, _LATENT), dtype=torch.bfloat16, device=self._device)
                         for layer in args.full_attention_layer_ids}
            self._pool = {layer: torch.zeros((num_pages, _PAGE, 1, _INDEX), dtype=torch.bfloat16, device=self._device)
                          for layer in args.full_attention_layer_ids}
            self._keys = {layer: torch.empty((_RATIO, _INDEX), dtype=torch.bfloat16, device=self._device)
                          for layer in args.full_attention_layer_ids}
            self._gates = {layer: torch.empty((_RATIO, _INDEX), dtype=torch.float32, device=self._device)
                           for layer in args.full_attention_layer_ids}
        self._request = None
        self._position = None
        self._locations = None
        self._native_pages = None
        self._graph = None
        self._poisoned = False
        self._lock = Lock()
        # Opt-in steady-state fast path.  Every option defaults off and any
        # miss simply executes the original per-field/per-copy code below.
        self._batch_native_metadata = bool(batch_native_metadata)
        self._reuse_cache_views = bool(reuse_cache_views)
        self._batch_writeback = bool(batch_writeback)
        self._copy_pairs = None
        if self._batch_writeback:
            try:
                from vllm_ascend.worker.glm53_mk_batched import copy_pairs
            except Exception:
                _LOG.warning("MK batched writeback helper is unavailable; keeping per-tensor copies")
            else:
                self._copy_pairs = copy_pairs
        self._tail_reuse_supported = False
        if reuse_indexer_tail:
            try:
                self._tail_reuse_supported = "reuse_indexer_tail" in inspect.signature(
                    self._state.prepare_shared).parameters
            except (TypeError, ValueError):
                self._tail_reuse_supported = False
        self._continuation_tails = (
            {layer: (pool, None, None) for layer, pool in self._pool.items()}
            if self._tail_reuse_supported else None
        )
        self._kda_views = None

    def step(self, forward_context, *, request_id, position, input_ids, kda, mla, indexer):
        """Consume one scheduled token and commit it to the VA-owned cache.

        ``kda``, ``mla`` and ``indexer`` are mappings keyed by MK layer id of
        the public cache descriptors. Their prefixes resolve in
        ``forward_context.attn_metadata``. Position is zero-based and equals
        the number of already committed tokens. A new request imports VA's
        native cache, even when it reuses the previous request's pages.
        """
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("VA bridge only supports serial decode")
        try:
            with torch.inference_mode(False), torch.no_grad():
                return self._step(forward_context, request_id, position, input_ids, kda, mla, indexer)
        finally:
            self._lock.release()

    def release(self, request_id):
        """Forget a finished or interrupted request while retaining graph storage."""
        with self._lock:
            if self._request == request_id:
                self._request = self._position = self._locations = self._native_pages = None
                self._kda_views = None

    def _step(self, context, request_id, position, input_ids, kda, mla, indexer):
        if self._poisoned:
            raise RuntimeError("VA bridge stopped after an MK execution or writeback failure")
        runtime_mode = getattr(context, "cudagraph_runtime_mode", None)
        _require(not getattr(context, "capturing", False)
                 and (runtime_mode is None or getattr(runtime_mode, "name", None) == "NONE"),
                 "VA bridge requires an eager VA ForwardContext")
        _require(isinstance(request_id, str) and request_id and type(position) is int
                 and 0 < position < self.max_context_tokens,
                 "VA bridge requires a request id and a nonzero position within its context capacity")
        _require(input_ids.shape == (1,) and input_ids.dtype in (torch.int32, torch.int64)
                 and input_ids.device == self._device, "VA bridge requires one device token")
        _require(kda.keys() == self._state.kda.conv.keys() and mla.keys() == indexer.keys() == self._mla.keys(),
                 "VA caches must cover the MK layers")
        metadata = context.attn_metadata
        _require(isinstance(metadata, dict), "VA bridge requires a per-layer ForwardContext metadata dict")
        imported = request_id != self._request
        reads = _DecodeMetadata()
        # Steady-state fast path: batch this step's native metadata reads into
        # one host transfer per dtype instead of one blocking transfer per
        # field.  Only the first token of a request keeps the original path.
        if self._batch_native_metadata and not imported:
            try:
                reads.prefetch(self._prefetch_plan(metadata, kda, mla, indexer, position))
            except Exception:
                # A field the plan cannot classify keeps the lazy native read.
                pass
        kda_slots = {layer: self._kda_slot(cache, metadata[cache.prefix], reads)
                     for layer, cache in kda.items()}
        previous_pages = (self._native_pages if self._reuse_cache_views and not imported
                          and self._native_pages is not None else None)
        mla_pages = {
            layer: self._mla_target(cache, metadata[cache.prefix], position, reads,
                                    previous=(previous_pages[layer][0]
                                              if previous_pages and layer in previous_pages else None))
            for layer, cache in mla.items()
        }
        idx_addresses = {
            layer: self._indexer_target(cache, metadata[cache.pool_prefix], metadata[cache.tail_prefix], position, reads,
                                        previous=(previous_pages[layer][1]
                                                  if previous_pages and layer in previous_pages else None))
            for layer, cache in indexer.items()
        }
        locations = (
            {layer: (slot, _layout(kda[layer].conv), _layout(kda[layer].recurrent), kda[layer].conv_dim_first)
             for layer, slot in kda_slots.items()},
            {layer: _layout(mla[layer].latent) for layer in mla_pages},
            {layer: (address.tail_page, _layout(indexer[layer].pool), _layout(indexer[layer].tail))
             for layer, address in idx_addresses.items()},
        )
        native_pages = {layer: (pages, idx_addresses[layer].pages) for layer, pages in mla_pages.items()}
        if not imported:
            pages_extend = all(new.extends(old) for layer, pair in native_pages.items()
                               for new, old in zip(pair, self._native_pages[layer]))
            _require(position == self._position + 1 and locations == self._locations and pages_extend,
                     "VA request must advance by one token with unchanged cache locations")
        # The committed layouts prove the cache tensors, slots and strides are
        # unchanged, so their per-layer views (34 slices and transposes) need
        # not be rebuilt on a continuation.
        if (self._reuse_cache_views and not imported and self._kda_views is not None
                and locations == self._locations):
            conv_states = self._kda_views
        else:
            conv_states = {layer: self._kda_views_for_slot(cache, kda_slots[layer])
                           for layer, cache in kda.items()}

        # Validate every destination before touching any data. MK execution
        # uses bridge-owned buffers, then writeback commits to VA in stream order.
        try:
            initial = None
            reuse_tails = False
            if imported:
                initial = self._import_prefix(position, conv_states, mla, mla_pages, indexer, idx_addresses)
                tails = self._import_tail(position, indexer, idx_addresses)
            elif self._continuation_tails is not None:
                # MK's own Indexer tail buffers were written by the previous
                # completed graph for this exact request/slot/position.  The
                # blockrt opt-in keeps them instead of re-zeroing and copying
                # VA's identical history back in.
                tails = self._continuation_tails
                reuse_tails = True
            else:
                tails = self._import_tail(position, indexer, idx_addresses)
            if self._graph is None:
                self._model.clear_graph_cache()
            paging = ({"page": _SCRATCH_PAGE} if self.max_context_tokens <= _PAGE
                      else {"page_table": self._page_table})
            if self.max_context_tokens > _PAGE and os.getenv("MK_GLM53_REUSE_PAGED_METADATA") == "1":
                paging["reuse_paged_metadata"] = True
            if reuse_tails:
                paging["reuse_indexer_tail"] = True
            batch, positions = self._state.prepare_shared(position, kda=initial, mla=self._mla, indexer=tails, **paging)
            self._token.copy_(input_ids)
            logits = self._model.forward(batch, self._token, positions, None)
            hidden = self._model.last_hidden_states
            graph = self._model._last_graph
            reused = self._graph is not None
            capture_head = getattr(self._model, "capture_lm_head", True)
            if (graph is None or (reused and graph is not self._graph)
                    or (logits is not None) != capture_head or hidden is None):
                raise RuntimeError("VA bridge requires usable outputs from one stable MK graph")
            # VA may allocate its cache tensors as inference tensors. MK's
            # tracked capture is complete by the time VA storage is updated.
            with torch.inference_mode():
                self._commit(position, conv_states, mla, mla_pages, indexer, idx_addresses)
        except Exception:
            self._poisoned = True
            raise

        self._request, self._position, self._locations, self._graph = request_id, position, locations, graph
        self._native_pages = native_pages
        if self._reuse_cache_views:
            self._kda_views = conv_states
        _LOG.info("VA/MK decode request=%s position=%d imported=%s graph=%#x reused=%s pool_boundary=%s "
                  "kda_slots=%s mla_pages=%s indexer=%s", request_id, position, imported, id(graph), reused,
                  (position + 1) % _RATIO == 0, kda_slots,
                  {layer: pages.ids[-1] for layer, pages in mla_pages.items()}, idx_addresses)
        return DecodeResult(logits, hidden, imported, reused)

    def _kda_slot(self, cache, metadata, reads):
        _require((metadata.num_actual_tokens, metadata.num_decodes, metadata.num_decode_tokens,
                  metadata.num_prefills, metadata.num_spec_decodes) == (1, 1, 1, 0, 0)
                 and metadata.spec_sequence_masks is None
                 and reads.values(metadata.non_spec_query_start_loc, 2, "KDA query boundaries") == (0, 1),
                 "VA KDA metadata must represent one non-speculative decode token")
        slot = reads.value(metadata.non_spec_state_indices_tensor, "KDA state index")
        self._validate_kda_slot(cache, slot)
        return slot

    def _validate_kda_slot(self, cache, slot):
        conv_shape = self._conv_shape[::-1] if cache.conv_dim_first else self._conv_shape
        _require(cache.conv.shape[1:] == conv_shape and cache.recurrent.shape[1:] == self._recurrent_shape
                 and cache.conv.dtype == torch.bfloat16 and cache.recurrent.dtype == torch.float32
                 and cache.conv.device == cache.recurrent.device == self._device
                 and 0 <= slot < min(cache.conv.shape[0], cache.recurrent.shape[0]), "invalid VA KDA cache or slot")

    @staticmethod
    def _kda_views_for_slot(cache, slot):
        conv = cache.conv[slot]
        if cache.conv_dim_first:
            conv = conv.transpose(-1, -2)
        return conv, cache.recurrent[slot].transpose(-1, -2)

    def _prefetch_plan(self, metadata, kda, mla, indexer, position):
        """List the native ``(tensor, columns)`` reads the validation will make.

        This mirrors :meth:`_kda_slot`, :meth:`_mla_target` and
        :meth:`_indexer_target` without touching device data: only Python
        geometry and metadata attributes are inspected.  ``_DecodeMetadata``
        ignores any entry it cannot classify and the original lazy read still
        runs, so an incomplete plan costs a host sync, never correctness.
        """
        entries = []

        def add(tensor, columns=None):
            entries.append((tensor, columns))

        for cache in kda.values():
            item = metadata[cache.prefix]
            add(item.non_spec_query_start_loc)
            add(item.non_spec_state_indices_tensor)
        for cache in mla.values():
            item = metadata[cache.prefix]
            add(item.query_start_loc)
            add(item.seq_lens)
            add(item.positions)
            add(item.slot_mapping)
            width = item.block_size
            if type(width) is int and width > 0:
                add(item.block_table, (position + 1 + width - 1) // width)
        pools = (position + 1) // _RATIO
        for cache in indexer.values():
            pool = metadata[cache.pool_prefix]
            tail = metadata[cache.tail_prefix]
            add(pool.query_start_loc)
            add(pool.positions)
            add(pool.start_pos)
            add(pool.raw_seq_lens)
            add(pool.seq_lens)
            add(pool.pool_tail)
            add(pool.slot_mapping)
            add(pool.pooled_slot_mapping)
            rows = cache.pool.shape[1]
            if type(rows) is int and rows > 0:
                add(pool.block_table, (pools + rows - 1) // rows)
            add(tail.block_table, 1)
            cann = tail.cann_inputs
            add(cann.history_indices, _RATIO)
            add(cann.write_indices, _RATIO)
        return entries

    def _mla_target(self, cache, metadata, position, reads, *, previous=None):
        _row_cache(cache.latent, (1, _LATENT), torch.bfloat16, self._device)
        rows = cache.latent.shape[1]
        # The NoPE sparse MLA layer is registered with AscendSFABackend.
        _require((metadata.num_actual_tokens, metadata.num_input_tokens, metadata.num_decodes,
                  metadata.num_decode_tokens, metadata.num_prefills) == (1, 1, 1, 1, 0)
                 and isinstance(metadata.block_size, int) and 0 < metadata.block_size <= rows
                 and rows % metadata.block_size == 0 and metadata.sin is None and metadata.cos is None
                 and reads.values(metadata.query_start_loc, 2, "MLA query boundaries") == (0, 1)
                 and reads.value(metadata.seq_lens, "MLA length") == position + 1
                 and reads.value(metadata.positions, "MLA decode position") == position,
                 "VA sparse NoPE MLA metadata must represent the scheduled decode token")
        pages = reads.active_pages(metadata.block_table, position + 1, metadata.block_size,
                                   cache.latent.shape[0] * rows, previous=previous)
        slot = reads.value(metadata.slot_mapping, "MLA slot")
        _require(slot == pages.slot(position), "VA MLA slot disagrees with the active prefix page")
        return pages

    def _indexer_target(self, cache, pool, tail, position, reads, *, previous=None):
        _row_cache(cache.pool, (1, _INDEX), torch.bfloat16, self._device)
        _row_cache(cache.tail, (2 * _INDEX,), torch.float32, self._device)
        # The native KeyPool writer splits pooled_slot_mapping using cache.shape[1].
        # Its metadata.block_size describes the 128-token SFA kernel divided by 4.
        rows = cache.pool.shape[1]
        prior_pools, prior_tail = divmod(position, _RATIO)
        pools, open_rows = divmod(position + 1, _RATIO)
        _require(pool.num_actual_tokens == 1 and pool.query_metadata is None
                 and pool.compress_ratio == _RATIO
                 and reads.values(pool.query_start_loc, 2, "KPool query boundaries") == (0, 1)
                 and reads.value(pool.positions, "KPool position") == position
                 and reads.value(pool.start_pos, "KPool start") == position
                 and reads.value(pool.raw_seq_lens, "KPool token length") == position + 1
                 and reads.value(pool.seq_lens, "KPool compressed length") == pools
                 and reads.value(pool.pool_tail, "KPool tail length") == open_rows, "invalid VA KPool decode metadata")
        pages = reads.active_pages(pool.block_table, pools, rows, cache.pool.shape[0] * rows,
                                   previous=previous)
        slot = pages.slot(prior_pools) if open_rows == 0 else -1
        pool_page, pool_row = divmod(slot, rows) if slot >= 0 else (None, None)
        _require(reads.value(pool.slot_mapping, "KPool token slot") == slot
                 and reads.value(pool.pooled_slot_mapping, "KPool compact slot") == slot, "invalid VA KPool slot")

        tail_page = reads.active_pages(tail.block_table, 1, 1, cache.tail.shape[0]).ids[0]
        cann = tail.cann_inputs
        capacity = cache.tail.shape[1]
        stride, remainder = divmod(cache.tail.stride(0), 2 * _INDEX)
        _require(cann is not None and cann.retained_indices is None and remainder == 0
                 and cann.page_rows == stride and capacity == tail.block_size == _RATIO
                 and cann.history_indices.shape == cann.write_indices.shape == (1, _RATIO),
                 "invalid VA non-speculative CANN tail metadata or page stride")
        history = reads.row(cann.history_indices, _RATIO, "CANN tail history")
        writes = reads.row(cann.write_indices, _RATIO, "CANN tail writes")
        base = tail_page * stride
        history_rows = tuple((position - prior_tail + row) % capacity for row in range(prior_tail))
        expected_writes = tuple(base + (position + 1 - open_rows + row) % capacity if row < open_rows else -1
                                for row in range(_RATIO))
        _require(history[:prior_tail] == tuple(base + row for row in history_rows) and writes == expected_writes,
                 "VA CANN tail indices do not address the declared tail page and ring rows")
        tail_row = writes[open_rows - 1] - base if open_rows else None
        return _IndexerAddress(pool_page, pool_row, tail_page, history_rows, tail_row, pages)

    def _import_prefix(self, position, kda, mla, mla_pages, indexer, idx_addresses):
        for layer, pages in mla_pages.items():
            if self.max_context_tokens <= _PAGE:
                self._mla[layer][_SCRATCH_PAGE].zero_()
                self._pool[layer][_SCRATCH_PAGE].zero_()
            _copy_prefix_runs(mla[layer].latent, pages, self._mla[layer], position, self._page_table, 1)
            _copy_prefix_runs(indexer[layer].pool, idx_addresses[layer].pages, self._pool[layer],
                         position // _RATIO, self._page_table, _RATIO)
        return {layer: (conv.contiguous(), recurrent.contiguous()) for layer, (conv, recurrent) in kda.items()}

    def _import_tail(self, position, indexer, idx_addresses):
        count = position % _RATIO
        result, pairs = {}, []
        for layer, address in idx_addresses.items():
            keys, gates = self._keys[layer][:count], self._gates[layer][:count]
            for i, row in enumerate(address.history_rows):
                source = indexer[layer].tail[address.tail_page, row]
                pairs.append((keys[i], source[:_INDEX]))  # FP32 CANN key -> BF16 MK key.
                pairs.append((gates[i], source[_INDEX:]))  # Unmodified FP32 raw gate, before APE.
            result[layer] = (self._pool[layer], keys, gates)
        if not (self._copy_pairs is not None and self._copy_pairs(pairs)):
            for destination, source in pairs:
                destination.copy_(source)
        return result

    def _commit(self, position, conv_states, mla, mla_pages, indexer, idx_addresses):
        if self._copy_pairs is not None:
            kda_pairs = []
            for layer, (conv, recurrent) in conv_states.items():
                kda_pairs.append((conv, self._state.kda.conv[layer][0]))
                kda_pairs.append((recurrent, self._state.kda.temporal[layer][0]))
            if not self._copy_pairs(kda_pairs):
                for destination, source in kda_pairs:
                    destination.copy_(source)
            attention_pairs = []
            for layer, pages in mla_pages.items():
                page, row = divmod(pages.slot(position), mla[layer].latent.shape[1])
                mk_page, mk_row = divmod(position, _PAGE)
                attention_pairs.append((mla[layer].latent[page, row],
                                        self._mla[layer][self._page_table[mk_page], mk_row]))
                cache, address = indexer[layer], idx_addresses[layer]
                if address.pool_row is not None:
                    mk_page, mk_row = divmod(position // _RATIO, _PAGE)
                    attention_pairs.append((
                        cache.pool[address.pool_page, address.pool_row],
                        self._pool[layer][self._page_table[_RATIO * mk_page], mk_row]))
                else:
                    tail = cache.tail[address.tail_page, address.tail_row]
                    mk_slot = position % _RATIO
                    attention_pairs.append((tail[:_INDEX],
                                            self._state.indexer.tail_k_buffer[layer][0, mk_slot]))
                    attention_pairs.append((tail[_INDEX:],
                                            self._state.indexer.tail_score_buffer[layer][0, mk_slot]))
            if not self._copy_pairs(attention_pairs):
                for destination, source in attention_pairs:
                    destination.copy_(source)
            return
        # Original per-tensor writeback, unchanged when batching is disabled.
        for layer, (conv, recurrent) in conv_states.items():
            conv.copy_(self._state.kda.conv[layer][0])
            recurrent.copy_(self._state.kda.temporal[layer][0])
        for layer, pages in mla_pages.items():
            page, row = divmod(pages.slot(position), mla[layer].latent.shape[1])
            mk_page, mk_row = divmod(position, _PAGE)
            mla[layer].latent[page, row].copy_(self._mla[layer][self._page_table[mk_page], mk_row])
            cache, address = indexer[layer], idx_addresses[layer]
            if address.pool_row is not None:
                mk_page, mk_row = divmod(position // _RATIO, _PAGE)
                cache.pool[address.pool_page, address.pool_row].copy_(
                    self._pool[layer][self._page_table[_RATIO * mk_page], mk_row])
            else:
                tail = cache.tail[address.tail_page, address.tail_row]
                mk_slot = position % _RATIO
                tail[:_INDEX].copy_(self._state.indexer.tail_k_buffer[layer][0, mk_slot])
                tail[_INDEX:].copy_(self._state.indexer.tail_score_buffer[layer][0, mk_slot])
