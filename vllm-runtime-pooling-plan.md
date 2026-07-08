# Runtime Pooling Method for vLLM BiomedRNA Plugin

## What was built

Per-request `pooling_method` override for the BiomedRNA vLLM plugin, matching
the `bmfm_targets.inference()` API. Each cell/request can specify a different
pooling method at call time; the model default from `checkpoint_metadata.json`
is used when none is specified.

### How it works

`pooling_method` is encoded as a scalar `int64` tensor (`pooling_method_id`)
and carried through vLLM's multimodal kwargs pipeline alongside `gene_ids`,
`expr_values`, and `attention_mask`. Pooling is applied inside `forward()`
where both the real `attention_mask` and the full BMFM `ModelOutput` are
available. `BiomedRnaPooler` is a simple pass-through.

`PoolingParams.extra_kwargs` was considered and rejected: it only reaches the
pooler, which runs after `forward()` and has no access to the real gene
sequence boundaries or BMFM model internals. See `agents.md §
"Why pooling_method travels via mm_kwargs"` for the full rationale.

### Encoding scheme (`constants.py`)

Named methods use **negative** codes so they can never collide with bare
integer token positions (always `>= 0`):

| Code | Method |
|------|--------|
| `-2` | `"first_token"` (model default) |
| `-3` | `"mean_pooling"` |
| `-4` | `"pooling_layer"` |
| `n >= 0` | literal token position `n` |

### Files changed

| File | Change |
|------|--------|
| `vllm_biomed_rna_plugin/constants.py` | `POOLING_METHOD_IDS` / `POOLING_METHOD_NAMES` with negative sentinel codes |
| `vllm_biomed_rna_plugin/biomed_rna.py` | `RnaProcessorItems` encodes `pooling_method_id`; `_pool_batch()` decodes and dispatches; module docstring explains why pooling lives in `forward()` |
| `vllm_biomed_rna_plugin/io_processor.py` | `parse_data()` / `pre_process()` forward `pooling_method` into `multi_modal_data["rna"]` |
| `examples/offline_biomed_rna_example.py` | `--pooling-method` CLI arg; `iter_h5ad_batches()` threads `pooling_method` through to `preprocess_anndata()` |
| `examples/online_biomed_rna_example.py` | `--pooling-method` CLI arg; `pooling_method` injected into `data` payload |
| `tests/test_io_processor.py` | Encoding round-trip tests (named methods, int positions, None default, negative int error, unknown string warning) |

### Tests

```bash
# CPU (no GPU required)
pytest vllm/tests/test_io_processor.py -v       # IO processor + encoding round-trip

# GPU
pytest vllm/tests/test_biomed_rna.py -v -s      # batching correctness + pooling override
pytest vllm/tests/test_regression.py -v -s      # vLLM vs bmfm-targets direct (WCED + MLM)
```

## Supported pooling methods vs `inference.py`

| Method | `inference.py` | vLLM plugin |
|--------|---------------|-------------|
| `"first_token"` | ✅ | ✅ |
| `"mean_pooling"` | ✅ | ✅ |
| `"pooling_layer"` | ✅ | ✅ |
| `int` (token position) | ✅ | ✅ |
| `list[int \| str]` (concatenate) | ✅ | ❌ not yet — see next section |

## Follow-up: `list[int | str]` pooling support

`inference.py` accepts `pooling_method=["first_token", "mean_pooling"]` which
concatenates both embeddings into one vector of shape `[batch, N × hidden_size]`.

The current scalar `pooling_method_id` field cannot encode a list — one integer
per request is not enough. The fix is a fixed-length `pooling_spec` vector.

### Task order

- [ ] **Write unit tests first** (CPU, no GPU) — see test spec below
- [ ] Implement `constants.py` changes
- [ ] Implement `_encode_pooling_spec` / `_decode_spec_row` helpers
- [ ] Update `RnaProcessorItems`, `BiomedRnaDummyInputsBuilder`, `RNA_FIELDS_CONFIG`
- [ ] Update `_pool_batch()` and `forward()`
- [ ] Update public APIs: `create_rna_vllm_input`, `preprocess_anndata`, `io_processor`, examples
- [ ] Run full test suite (CPU + GPU)

### Unit tests to write first (`tests/test_io_processor.py` or new file)

All CPU-only. Cover before touching any implementation:

```python
# 1. _encode_pooling_spec round-trips
assert encode("first_token")               == [-2, -1, -1, -1]
assert encode("mean_pooling")              == [-3, -1, -1, -1]
assert encode(5)                           == [ 5, -1, -1, -1]
assert encode(["first_token","mean_pooling"]) == [-2, -3, -1, -1]
assert encode(["first_token", 3])          == [-2,  3, -1, -1]
assert encode(None)                        == [-1, -1, -1, -1]

# 2. _decode_spec_row round-trips
assert decode([-2, -1, -1, -1]) == "first_token"
assert decode([-3, -1, -1, -1]) == "mean_pooling"
assert decode([ 5, -1, -1, -1]) == 5
assert decode([-2, -3, -1, -1]) == ["first_token", "mean_pooling"]
assert decode([-2,  3, -1, -1]) == ["first_token", 3]
assert decode([-1, -1, -1, -1]) == "first_token"   # all-PAD → model default

# 3. Error cases
raises ValueError  for encode([-1])          # negative int position reserved
raises ValueError  for encode(["bad_method"]) # unknown string
raises ValueError  for encode([1,2,3,4,5])   # list too long for POOLING_SPEC_LEN=4

# 4. _pool_batch with mixed-width output
#    Two rows: one requests ["first_token","mean_pooling"] (→ 768), one requests
#    "first_token" (→ 384).  Result must be list[Tensor] of lengths [768, 384].
#    (This test needs a fake ModelOutput with known hidden_states — no GPU needed.)

# 5. _pool_batch uniform output
#    Two rows both requesting "first_token" → result is Tensor[2, 384], not a list.

# 6. Regression: existing callers that never set pooling_method get identical
#    output to current implementation (all-PAD spec → default method).
```

### Design

#### `constants.py` — new constants

```python
POOLING_SPEC_LEN  = 4    # max methods per request; bump if needed
POOLING_CODE_PAD  = -1   # unused slots in pooling_spec tensor
# Existing negative codes unchanged: first_token=-2, mean_pooling=-3, pooling_layer=-4
```

Replace `pooling_method_id` with `pooling_spec` in `RNA_FIELDS_CONFIG`:
```python
"pooling_spec": MultiModalFieldConfig.batched("rna"),  # int64[POOLING_SPEC_LEN]
```

Because every request emits a fixed-shape `int64[POOLING_SPEC_LEN]` tensor,
`MultiModalBatchedField._reduce_data()` always stacks into `Tensor[batch, POOLING_SPEC_LEN]`
— no ragged fallback, no special casing.

#### `_encode_pooling_spec` helper (add to `biomed_rna.py` or new `pooling_spec.py`)

```python
def _encode_pooling_spec(
    raw_method: str | int | list[str | int] | None,
    length: int = POOLING_SPEC_LEN,
) -> torch.Tensor:
    """Encode str | int | list | None → int64[length], PAD=-1 fill."""
    if raw_method is None:
        return torch.full((length,), POOLING_CODE_PAD, dtype=torch.long)

    items = raw_method if isinstance(raw_method, list) else [raw_method]
    if len(items) > length:
        raise ValueError(
            f"pooling_method list has {len(items)} entries, max is {length} "
            f"(POOLING_SPEC_LEN). Increase POOLING_SPEC_LEN if needed."
        )
    codes = []
    for item in items:
        if isinstance(item, str):
            if item not in POOLING_METHOD_IDS:
                logger.warning("Unknown pooling_method %r — using model default.", item)
                codes.append(POOLING_METHOD_DEFAULT_ID)
            else:
                codes.append(POOLING_METHOD_IDS[item])
        elif isinstance(item, int):
            if item < 0:
                raise ValueError(
                    f"Integer pooling_method must be >= 0 (token position); "
                    f"got {item}. Negative values are reserved."
                )
            codes.append(item)
        else:
            raise ValueError(f"Unsupported pooling_method item: {item!r}")

    codes += [POOLING_CODE_PAD] * (length - len(codes))
    return torch.tensor(codes, dtype=torch.long)
```

#### `_decode_spec_row` (new method on `BiomedRnaForSequenceEmbedding`)

```python
def _decode_spec_row(self, row: list[int]) -> str | int | list[str | int]:
    """Decode one pooling_spec row → form get_embeddings_from_outputs() expects."""
    items = [POOLING_METHOD_NAMES.get(code, code)
             for code in row if code != POOLING_CODE_PAD]
    if not items:
        return POOLING_METHOD_NAMES.get(
            self.default_pooling_method_id, self.default_pooling_method_id
        )
    return items[0] if len(items) == 1 else items
```

#### New `_pool_batch()` — handles same-width and mixed-width

```python
def _pool_batch(
    self,
    output,
    attention_mask: torch.Tensor,   # [batch, seq_len]
    spec_rows: list[list[int]],     # [batch, POOLING_SPEC_LEN]
) -> torch.Tensor | list[torch.Tensor]:
    """
    Apply per-request pooling. Returns:
      - Tensor[batch, embed_dim]   when all rows produce the same-width embedding
      - list[Tensor[embed_dim_i]]  when rows produce different widths
        (vLLM PoolerOutput accepts list[Tensor] natively — each user gets the
        correctly-sized embedding they asked for even in a mixed-user batch)
    """
    decoded = [self._decode_spec_row(row) for row in spec_rows]

    # Fast path: all identical → one batched call
    if len({str(m) for m in decoded}) == 1:
        return get_embeddings_from_outputs(
            output, attention_mask, pooling_method=decoded[0]
        )

    # Mixed: one call per unique method, slice out the right row
    cache: dict[str, torch.Tensor] = {}
    results: list[torch.Tensor] = []
    for i, method in enumerate(decoded):
        key = str(method)
        if key not in cache:
            cache[key] = get_embeddings_from_outputs(
                output, attention_mask, pooling_method=method
            )
        results.append(cache[key][i])

    # Stack if all same width (common); fall back to list if widths differ
    try:
        return torch.stack(results)
    except RuntimeError:
        return results   # mixed-width: list[Tensor] — vLLM handles natively
```

#### `forward()` — only the changed lines

```python
# Replace pooling_method_id extraction:
spec_tensor = kwargs.get("pooling_spec")   # Tensor[batch, POOLING_SPEC_LEN]

if spec_tensor is None or isinstance(spec_tensor, list):
    batch_size = gene_ids.shape[0]
    default_row = [self.default_pooling_method_id] + [POOLING_CODE_PAD] * (POOLING_SPEC_LEN - 1)
    spec_rows = [default_row] * batch_size
else:
    spec_rows = spec_tensor.tolist()   # Tensor[batch, POOLING_SPEC_LEN] → list[list[int]]

# Replace _pool_batch call + return:
pooled = self._pool_batch(output, attention_mask.float(), spec_rows)

if isinstance(pooled, list):
    return pooled          # mixed-width: list[Tensor] goes straight to PoolerOutput
return pooled.unsqueeze(1) # uniform: Tensor[batch, 1, embed_dim] → BiomedRnaPooler unwraps
```

`BiomedRnaPooler.forward()` is **unchanged** — when `forward()` returns
`list[Tensor]`, vLLM's model runner routes it directly into `PoolerOutput`
without calling the pooler on it.

## Remaining known issue (deferred)

**`prompt_token_ids` inconsistency:** `preprocess.py` and `test_plugin_integration.py`
still use `[0] * seq_len` (old pattern). The canonical design uses a single
dummy token `[1]`. Functionally harmless for this model (no KV cache, values
ignored) but inconsistent with the module docstring. Safe to fix in a
dedicated cleanup commit.
