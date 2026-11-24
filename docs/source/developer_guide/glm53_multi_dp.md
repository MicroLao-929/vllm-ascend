# GLM-5.3 Flash 多来源 DP 解码接入与验收

本配方用于普通解码，TP1＋DP8＋EP8、关闭 MTP。原生模型负责 prefill，实际有一个解码 token 的活动 rank 在同一轮执行自己的完整 MK 图，使用本 rank 的注意力 KV 并返回本地 hidden/logits。空闲 rank 同轮参与专家通信。默认关闭多来源接管。

## 接入

1. 在配套 MegaKernel 仓库准备 GLM-5.3 多来源后端，包括能力标识为 `glm53.tp1.dp8.ep8.multi_source_b1.v1` 的 `blockrt.models.glm_5_3_flash.multi_dp`、匹配的模型模块以及 strict TP1/DP8/EP8 专家对象。vLLM-Ascend 配方会安装相应的 companion、gate 与 worker。运行时检查 ABI、companion 接口及所有 rank 的配置。
2. 执行 `python examples/recipes/glm53_dp8/prepare.py <v0.30源码或部署源> <新输出> --multi-dp`，让各 worker 从该输出载入 vLLM-Ascend。目标可为固定 stock v0.30 或已支持的部署 runner。各 rank 设置 `MK_GLM53_MULTI_DP=1` 才启动多来源路线。未加配方参数时沿用原有准备流程。
3. 需要路由证据时，在捕图前对全部 rank 设置 `MK_GLM53_MULTI_DP_RECEIPTS=1 MK_GLM53_MULTI_DP_TRACE=1`，将 `GLM53_MULTI_AUDIT` 指向已创建、可写的独立输出目录。每个 rank 写独立 JSONL。用配方的 `multi_dp/merge_audit.py` 汇总，再用匹配后端的来源回执检查器验证。性能测试另起同配置服务并关闭回执保留、逐步日志。

## 八卡证据

180 机器上，R1 完成了短输出的真实设备审计。按外部请求 ID 对齐后，14 个 epoch 中同时有 8 个不同 owner，各自运行一份本地 full graph。另有 1 个三 owner epoch 和 3 个全组原生 epoch。在全部 42 个路由层，每个真实来源发出的 8 行与所有目的 rank 的收取量按来源相加一致，峰值单目的单层收到 25 行。八个请求中，3 个核验了位置 128 至 142，另 5 个核验了位置 129 至 142。记录标识为 `mk-multi-audit-static-stock13-h180-202426-70659`，文件为 `audit-client/validated.json`、`audit-client/combined.log` 和 `runtime-manifest.json`。完整 128 输出的 token 与未插桩 TPOT 必须使用相应运行的实测数据。

发布配方采用 R2。与上述设备审计的 R1 相比，R2 仅调整冷启动阶段的错误处理，包括提前检查 runner 配置、全组传递 bootstrap/初始化/构造错误，以及显式报告返回空 hook。成功解码的 gate、KV、图和算子行为沿用 R1。stock 与部署 v0.30 都实际完成了正常配方构建和八进程 CPU 检查，覆盖逐源本地 logits、全组原生回退、提交后异常与 Gloo/compact 两种启动协议。单个 rank 注入初始化或构造错误时，八个 rank 能获得相同的原始错误。可复现命令见配方目录的 `multi_dp/README.md`。普通 native graph 基线独立运行。

## 适用范围

服务可以配置每 rank 两个调度槽。是否接管取决于当步真实 batch：每个活动 rank 须只有一条请求、一个 token，位于实际 row 0。遇到真实本地 B2、prefill、迁移、重复请求标识，所有 rank 共同选择原生。设备写回开始后的异常会中止该轮，避免对同一 token 再写一次。C16 且每 rank 仅一个槽位时可测 16 个到达请求的排队行为，实际最多八个同时使用 MK。真正的每 rank B2 双请求图属于独立能力。

## 同机完整输出与正式 R2 入口

在 103 的同一套 A5 设备上，双方均使用 TP1、DP8、EP8，关闭 MTP。每条请求输入和输出各 128 个 token，每个并发点做一轮预热和三轮正式测试。整机输出吞吐包含 prefill，解码吞吐按非首 token 数除以该波首个至最后一个输出 token 的到达时间计算。表中吞吐取三波中位数，TPOT 取逐请求中位数。

| 并发 | 实现 | TPOT（ms/token） | 整机输出（token/s） | 解码（token/s） | 逐步后端计数 |
| ---: | --- | ---: | ---: | ---: | --- |
| 8 | 原生 graph 配置 | 18.061 | 417.265 | 442.786 | native_eager 2、native_graph 3046 |
| 8 | R1 候选服务 | 26.590 | 280.286 | 300.783 | mk_target 3048 |
| 16 | 原生 graph 配置 | 19.899 | 753.567 | 803.359 | native_eager 1、native_graph 6095 |
| 16 | R1 候选服务 | 171.449 | 92.879 | 93.309 | mk_fallback_native_eager 6096 |

C8 的 3,048 个 decode 步全部执行 MK。相对原生 graph 配置，整机输出吞吐降低 32.83%，逐请求 TPOT 增加 47.23%。C16 的候选服务所有 decode 步均退回原生 eager，这组数据描述回退的服务时延。此前 103 的单请求短上下文对照中，原生 graph 为 16.861 ms/token，严格 MK 为 25.195 ms/token。单请求原生与 MK 编译方式沿用前述配置。

从 C8 第一波取八个输入，逐条在同一模型、同一主机上以 B1 重放。两种并发方式的 1,024 个输出 ID 全部一致，串行对照的 1,016 个 decode 步也全部命中 MK。将 C8 全部 24 个请求与原生 target 比较，13/24 个整序列一致，1,932/3,072 个位置相同。有分叉的请求首次分叉在第 2 至 15 个 token，按 1 起计。

正式 R2 配方在 110 上通过独立的短输出硬件审计。正式项目入口加载上述成对包，八个不同请求在 14 个 epoch 各自执行完整图，另有 1 个三 owner epoch。全部 42 层的逐来源发送/接收计数守恒。单目的 rank 峰值 26 行。各请求起点为第 128 或 129 个输入位置，本轮核对至第 142 个位置。R1 的完整计时与 R2 的短程正确性分别列示。

波级吞吐、逐请求 TPOT、逐输入首次分叉和原始证据 SHA-256 见[可复算测量记录](glm53_dp8_measured_20261123.json)。

## 独立本地双行实验

每卡两个请求的实际吞吐、逐 ID 分叉和归因限制见[双行实验记录](glm53_two_row_experiment.md)。
