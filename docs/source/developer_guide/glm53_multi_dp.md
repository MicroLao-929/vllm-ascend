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
