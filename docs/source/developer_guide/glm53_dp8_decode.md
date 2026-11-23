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

## 同一步元数据复用、批量读取专家计数和定长控制消息

worker 在一次 decode 内复用桥接元数据，合并连续页复制，将辅助 rank 的专家计数收集到一次热回放后的回读。普通热路径的控制协议改用固定 171 字节格式；需附带异常信息或者超长请求标识时，所有 rank 通过原协议交换详情。

用 `GLM53_PLAIN_COMPACT_GATE=1` 与 `VLLM_ASCEND_GLM53_MK_BATCH_COMPANION_READBACK=1` 启用这组改动。同一轮五次 128/128 请求的中位 TPOT 从 38.955 ms 降到 33.578 ms。再设置 `SCHEDULING_MODE=static` 时测得 30.418 ms。运行 `python3 tests/ut/recipes/test_glm53_dp8_control.py` 可验证所有 rank 针对普通输入、混合请求、异常、乱序和控制消息回退作出一致决定。

## 单机八进程控制通道

在定长协议上设置 `GLM53_DP8_LOCAL_CONTROL=1`，八个 worker 用 Linux 共享内存和本地通知完成每步的选择、就绪和完成确认。初始化通过 Gloo 协商并核对网络和进程命名空间、对端 rank 身份。此项要求八个进程位于同一主机及命名空间。分机部署保留原 Gloo 配置。

八进程 CPU 微基准中，每次选主和两次确认的中位时长从 Gloo 的 2.017 ms 降到 0.168 ms，包含 160 步并跳过前十步。端到端服务还包含模型计算和专家通信。运行 `python3 tests/ut/recipes/test_glm53_dp8_ipc.py` 可验证八种来源 rank 及多个来源争用时的集体决定。
