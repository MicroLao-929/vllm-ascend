# GLM-5.3 Flash 普通解码的 DP8 集成

此配方针对 vLLM Ascend v0.30 的 GLM-5.3 钩子。固定源码为 `grandolantow/vllm-ascend@cd87230b21d74bbc1958dacd57d3b496da6f8919`。配方按 SHA-256 核对参与替换的 worker 文件，再在单独的目录生成包。原生 graph 服务直接使用原有环境，只有显式设置新包路径的 MegaKernel 服务才会加载该版本。

```bash
python3 examples/recipes/glm53_dp8/prepare.py "$V030_ASCEND" "$OUTPUT"
export PYTHONPATH="$OUTPUT${PYTHONPATH:+:$PYTHONPATH}"
```

`V030_ASCEND` 指向上述版本的源码或已按该版本准备的运行环境，须含 `vllm_ascend/`。 `OUTPUT` 使用一个新目录。模型、兼容的 vLLM 核心、CANN 和 MegaKernel 运行时由部署环境提供。运行 `python3 tests/ut/recipes/test_glm53_dp8_overlay.py` 可验证安装隔离及版本不符时的处理。

## TP1、DP8、EP8 单请求

八个 rank 分摊专家，每张卡保留完整注意力。原生 prefill 后，持有该请求的 rank 回放 MegaKernel 主模型，其余七个 rank 执行相应的专家层。每个 rank 复用本地原生专家权重。普通解码关闭 MTP，允许页式上下文至 2,048 token。本轮 45 层 W4A4，单请求 128 进、128 出，逐层回读版本的五次请求 TPOT 中位数为 38.955 ms。同轮原生 decode graph 为 16.829 ms。

已有 v0.30 MK 工作环境中，服务的公共参数包括 `--tensor-parallel-size 1 --data-parallel-size 8 --enable-expert-parallel --max-model-len 2048 --max-num-seqs 1`。原生服务使用 `--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2]}'`。MK 服务使用 `--enforce-eager` 让 native prefill 走常规入口，decode 由 worker 内的 MK graph 捕获和回放。MK 需设置 `VLLM_ASCEND_GLM53_MK_EP=1`、`VLLM_ASCEND_GLM53_MK_NATIVE_EXPERTS=1`、`VLLM_ASCEND_GLM53_MK_MAX_CONTEXT_TOKENS=2048`、`VLLM_ASCEND_GLM53_MK_CHECKPOINT` 以及八个 rank 共用的 `VLLM_ASCEND_GLM53_MK_EP_ENDPOINT`。普通解码省略 speculative config。

## 同一步元数据复用、批量读取专家计数和定长控制消息

worker 在一次 decode 内复用桥接元数据，合并连续页复制，将辅助 rank 的专家计数收集到一次热回放后的回读。普通热路径的控制协议改用固定 171 字节格式。需附带异常信息或者超长请求标识时，所有 rank 通过原协议交换详情。

用 `GLM53_PLAIN_COMPACT_GATE=1` 与 `VLLM_ASCEND_GLM53_MK_BATCH_COMPANION_READBACK=1` 启用这组改动。同一轮五次 128/128 请求的中位 TPOT 从 38.955 ms 降到 33.578 ms。再设置 `SCHEDULING_MODE=static` 时测得 30.418 ms。运行 `python3 tests/ut/recipes/test_glm53_dp8_control.py` 可验证所有 rank 针对普通输入、混合请求、异常、乱序和控制消息回退作出一致决定。

## 单机八进程控制通道

在定长协议上设置 `GLM53_DP8_LOCAL_CONTROL=1`，八个 worker 用 Linux 共享内存和本地通知完成每步的选择、就绪和完成确认。初始化通过 Gloo 协商并核对网络和进程命名空间、对端 rank 身份。此项要求八个进程位于同一主机及命名空间。分机部署保留原 Gloo 配置。

八进程 CPU 微基准中，每次选主和两次确认的中位时长从 Gloo 的 2.017 ms 降到 0.168 ms，包含 160 步并跳过前十步。端到端服务还包含模型计算和专家通信。运行 `python3 tests/ut/recipes/test_glm53_dp8_ipc.py` 可验证八种来源 rank 及多个来源争用时的集体决定。

## 缓存视图、Indexer 尾部与写回

同一请求的连续 token 共用缓存视图和已验证的页表绑定。桥接层批量读取原生元数据、批量提交格式匹配的缓存写回。缺少接口或批量复制失败时，执行逐项路径。Indexer 尾部和页表复用依赖 MegaKernel 侧配套的 `MKDecodeState.prepare_shared` 接口，部署时先安装该仓库的 GLM-5.3 页式状态补丁。

在上节配置之上打开 `MK_GLM53_BATCH_NATIVE_METADATA=1`、`MK_GLM53_REUSE_CACHE_VIEWS=1`、`MK_GLM53_BATCH_WRITEBACK=1`、`MK_GLM53_REUSE_INDEXER_TAIL=1` 和 `MK_GLM53_REUSE_PAGED_METADATA=1`。单机本地控制与这些选项合用后，支持采集的服务在暂停采集期间七次请求中位 TPOT 为 29.561 ms。另起的未插桩静态基线为 31.012 ms。两者运行条件不同，没有对每个应用层开关拆分端到端收益。

七条固定输入与此前 MK 的输出完全一致，另十二条自然输入的 1,536 个输出位置全同，八个 rank 的 7,112 条专家回执摘要也一致。各次 TPOT 与摘要在 `examples/recipes/glm53_dp8/measurements.json`。独立的拷贝及失败回退回归见 `tests/ut/recipes/test_glm53_dp8_cache_copy.py`。这些精度统计的参照为改动前 MK。与原生 target 的对比应单独报告。

## 短上下文延迟发布页表候选

参考优化来自固定版本的 255K 预热前缀、TP8/DP1/EP8 实验。其原生前后平均为 17.9917 ms/token，MK 为 13.5812 ms/token。该实验输入 1 个新 token、生成 200 个，拓扑和前缀状态均不同。本候选仅移植可以独立验证的两项延迟操作，即发布 VA 设备页表和构建 slot mapping。[参考分支](https://gitcode.com/xupengpeng1/megakernel/commits/codex/glm53-255k-optimizations)

请求 rank 上一轮已由 MK 完成、地址影子比对达到阈值且没有分歧、没有缓存或页迁移更新时，才允许延迟发布。早期判定或集体投票拒绝时，原生入口须先发布页表和 slot，再构建 attention metadata，并更新实际 ForwardContext。专家 companion 使用此前的轮次协议。

用同一个实验包分别启动 control 和 candidate，保持普通 128/128、TP1/DP8/EP8、单请求、关闭 MTP 及严格模式的其它变量相同。

```bash
python3 examples/recipes/glm53_dp8/prepare.py "$V030_ASCEND" "$EXPERIMENT" --lazy-tables
GLM53_TEST_ASCEND="$EXPERIMENT" python3 tests/ut/recipes/test_glm53_dp8_lazy.py
export PYTHONPATH="$EXPERIMENT${PYTHONPATH:+:$PYTHONPATH}"
export MK_GLM53_NATIVE_BUILDER_DEFERRAL=on MK_GLM53_EARLY_MIN_SHADOW=8
# Control process
export MK_GLM53_DP8_LAZY_TABLES=0
# Candidate process, in a separate launch
export MK_GLM53_DP8_LAZY_TABLES=1
```

十项 CPU 回归覆盖普通解码、warmup 和 prefill、回退时机、两处原生消费者以及异常一致性。候选的八卡命中率、逐 token 一致性和 TPOT 收益均待同机 A/B；不将参考分支的 24.51% 计入本配置。默认安装仍使用上一节已测的六个 worker。

## 单机 DP 元数据同步

八个 DP rank 可以在连续普通 MK 解码中尝试本地交换 token 数和图模式。各步资格及原生回退条件、CPU 对照和硬件验收口径见[本地 DP 同步实验](glm53_local_dp_sync.md)。

## 八个独立请求共同执行一次专家通信

多来源配方、逐卡请求和日志验收方法及短程设备结果见[GLM-5.3 多来源 DP 解码](glm53_multi_dp.md)。
