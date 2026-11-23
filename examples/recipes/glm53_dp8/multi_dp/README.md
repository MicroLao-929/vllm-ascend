# GLM-5.3 Flash multi-source ordinary decode

Install the strict v0.30 worker adapter with `python examples/recipes/glm53_dp8/prepare.py <v0.30-source> <new-output> --multi-dp`. The same command supports the stock pinned runner and the prepared deployment runner recorded in [manifest.json](manifest.json). Put the resulting directory on the worker PYTHONPATH using the existing GLM-5.3 serving launcher.

Set `MK_GLM53_MULTI_DP=1` on every worker to use the installed path. It requires the matching blockrt module `blockrt.models.glm_5_3_flash.multi_dp`, capability `glm53.tp1.dp8.ep8.multi_source_b1.v1`, its matching GLM model implementation, and the strict TP1/DP8/EP8 expert objects. The loader validates backend capability, companion signature, topology and rank agreement at startup. Native prefill, ordinary decode, synchronous execution and no MTP are the supported setup. The default flag value leaves the existing execution path active.

Each rank may have a different request ID and position. An admitted epoch requires one actual unpadded decode token on every active rank at row zero. The service may have multiple scheduler slots. Actual local B2, prefill, ambiguous ownership or migration returns the group to native execution. Idle ranks execute the expert companion once. A failed preparation can fall back before any rank is granted a device launch. A failure after launch stops the generation because some destinations may already have committed that token. The optional single-owner native-builder deferral and local-DP-sync recipes make different assumptions about the prior global owner; the new option uses the strict adapter profile.

For routing evidence set `MK_GLM53_MULTI_DP_RECEIPTS=1 MK_GLM53_MULTI_DP_TRACE=1` on all ranks before graph capture. A pre-created directory supplied through `GLM53_MULTI_AUDIT` receives one JSONL per rank/PID. The logs name the actual owner set, local full/companion graph, request ID, position, epoch and 42 routed-layer receipts. Merge the files with `python multi_dp/merge_audit.py <audit-directory>` and validate source conservation using the accompanying blockrt receipt validator. A genuine C8 epoch has eight distinct source identities and eight full local graphs. C16 at one slot per rank measures queued requests with up to eight simultaneous local B1 decodes. Compare with native B2 only as a separately labeled configuration.

Run the Linux CPU integration checks with the actual prepared output.

```sh
python examples/recipes/glm53_dp8/multi_dp/cpu_check.py <new-output> <cpu-artifacts>
python examples/recipes/glm53_dp8/multi_dp/startup_check.py <new-output>
python -m pytest tests/ut/recipes/test_glm53_multi_dp.py
```

The CPU checks use real IPC/Gloo, production worker methods and a synthetic tensor exchange to check request ownership, local logits, failure consensus and fallback. They do not measure hardware speed or establish NPU model accuracy. Use a separate untimed receipt service and a fresh service without receipt retention for performance. Record actual returned token IDs, the first divergence versus native graph, TPOT, total throughput and per-card memory for the same requests and same host.
