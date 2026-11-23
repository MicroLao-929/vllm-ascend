# GLM-5.3 Flash 普通解码的 DP8 集成

此配方针对 vLLM Ascend v0.30 的 GLM-5.3 钩子。固定源码为 `grandolantow/vllm-ascend@cd87230b21d74bbc1958dacd57d3b496da6f8919`。配方按 SHA-256 核对参与替换的 worker 文件，再在单独的目录生成包。原生 graph 服务直接使用原有环境，只有显式设置新包路径的 MegaKernel 服务才会加载该版本。

```bash
python3 examples/recipes/glm53_dp8/prepare.py "$V030_ASCEND" "$OUTPUT"
export PYTHONPATH="$OUTPUT${PYTHONPATH:+:$PYTHONPATH}"
```

`V030_ASCEND` 指向上述版本的源码或已按该版本准备的运行环境，须含 `vllm_ascend/`。 `OUTPUT` 使用一个新目录。模型、兼容的 vLLM 核心、CANN 和 MegaKernel 运行时由部署环境提供。运行 `python3 tests/ut/recipes/test_glm53_dp8_overlay.py` 可验证安装隔离及版本不符时的处理。

## TP1、DP8、EP8 单请求

八个 rank 分摊专家，每张卡保留完整注意力。原生 prefill 后，持有该请求的 rank 回放 MegaKernel 主模型，其余七个 rank 执行相应的专家层。每个 rank 复用本地原生专家权重。普通解码关闭 MTP，允许页式上下文至 2,048 token。本轮 45 层 W4A4，单请求 128 进、128 出，逐层回读版本的五次请求 TPOT 中位数为 38.955 ms；同轮原生 decode graph 为 16.829 ms。

已有 v0.30 MK 工作环境中，服务的公共参数包括 `--tensor-parallel-size 1 --data-parallel-size 8 --enable-expert-parallel --max-model-len 2048 --max-num-seqs 1`。原生服务使用 `--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2]}'`。MK 服务使用 `--enforce-eager` 让 native prefill 走常规入口，decode 由 worker 内的 MK graph 捕获和回放。MK 需设置 `VLLM_ASCEND_GLM53_MK_EP=1`、`VLLM_ASCEND_GLM53_MK_NATIVE_EXPERTS=1`、`VLLM_ASCEND_GLM53_MK_MAX_CONTEXT_TOKENS=2048`、`VLLM_ASCEND_GLM53_MK_CHECKPOINT` 以及八个 rank 共用的 `VLLM_ASCEND_GLM53_MK_EP_ENDPOINT`。普通解码省略 speculative config。
