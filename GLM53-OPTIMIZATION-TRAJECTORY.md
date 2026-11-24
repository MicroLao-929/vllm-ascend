# GLM-5.3 Flash 优化轨迹

更新于 2026-11-24。此文件记录 `megakernel-glm5_3flash` 的 vLLM Ascend 接入、缓存和请求调度。配套 MegaKernel 分支为 `vllm-mega-glm53flash`。最终集成方向为原生 prefill、MegaKernel 主模型、模型自带 NextN MTP draft，并保留独立的原生 graph 入口。当前普通解码配方均在 Runner V1 验收。

## 从接入到可复现对照

| 时间与提交 | 改动 | 实测结果和边界 |
| --- | --- | --- |
| 11-19 至 11-21，早期工程接入 | 对齐 45 层 W4A4 模型视图、MXFP8 KDA、MLA 与压缩 Indexer 元数据。打通原生模型跑分、graph 捕获、decode hook 和 KV 写回 | 已完成 45 层八卡服务与实际重放。早期 eager 与 graph 的编译模式和精度状态不同，后续统一用同机、同拓扑原生 decode graph 做正式对照 |
| 11-21，MTP 历史验收 | V1 使用真实第 45 号 NextN。早期实验由 MK 提草稿，原生 target 验证；逐 token 验证普通原生和带草稿输出，并测原生默认入口回归 | 在 TP1 + DP8 + EP8，16/32 的普通原生 / 自带原生 MTP / MK draft 分别为 17.87 / 16.06 / 18.07 ms。16K/64 为 17.13 / 14.65 / 17.01 ms。自然文本、随机样本和长输入测试的输出 ID 均与普通原生一致，此后开发重心转向 MK target |
| 11-23，原生草稿与 MK target | 原生 prefill 和 NextN proposer 继续运行，接入可选 external decode adaptor。TP8 分片、EP8，使用原生词嵌入、LM head 和采样。K=1 的目标验证按两次 B1 回放执行 | 128/128 原生 graph MTP / MK target 为 12.648 / 28.201 ms，1024/128 为 12.757 / 28.099 ms。128 输出自然文本 0/12 全串相同，366/1,536 个位置相同，首次分叉最早第 4、中位第 22.5 个输出。128/128 及另外两档详细数据在工程历史基线 |
| 11-23，`2550d8464` | 固定 vLLM 0.30.0 的普通解码配方。DP8 首版每轮由单个请求 rank 执行主模型，其余 rank 执行专家层，关 MTP | 单请求 128/128 原生 graph 16.829 ms，初版 MK 38.955 ms |
| 11-23，`632bac02a` | 合并逐层专家回读并用定长控制消息，降低 Python/主机协调开销；另对比 static 调度和单一 companion 图 | 批量回读与定长消息同开为 33.578 ms，进一步使用 static 为 30.418 ms。把辅助卡的 42 个图合成一张，在三档 TPOT 变化仅为 −0.60、+0.35、−0.59 ms；取完整同口径记录判断 |
| 11-23，`3a242b69d`、`e26120ffc` | 单机共享内存控制通道，复用本步观察到的元数据和缓存视图，连续运行按有效页段批量同步写回，并与 MK 的页表、尾部复用组合 | 八进程 CPU 门禁每轮中位 Gloo 2.017 → 0.168 ms。组合配置的七次 128/128 TPOT 中位 29.561 ms，固定和自然输入均同旧 MK，逐 rank 专家摘要一致。端到端数字属于组合 |
| 11-23，`33ad9f3bb`、`a155a00e5` | 同一 decode 连续路径尝试推迟 CPU 页表发布、slot mapping 和 metadata builder。另实现本地 DP 元数据交换，保留混合 prefill、失配和异常时的原生同步 | lazy tables 有 10 项 CPU 回归，本地 DP 同步组合按八进程差分校对。实机观测原生具名 Gloo 两次采集为 1.434 / 1.485 ms，MK 的对应整个调度段为 1.539 ms。这两项单独的端到端收益待隔离 |
| 11-23 至 11-24，`1c5a7a265`、`a021b60d6`、`b292f343b` | 多来源协议允许八个独立请求各自持有 KV，在同一轮执行 MK target 和 EP。按 request ID、位置、rank 和 epoch 关联真实路由与回执 | 103 正式 C8 的 3,048 个步全由 MK 执行；八条同 rank 串行对照 1,024/1,024 ID 相同。正式 R2 在 110 确认八来源 14 个 epoch 和 42 层逐来源守恒。每卡真实 B2 或混合 prefill 会由常规配方一致回退 |
| 11-24，参考原码复现 | 固定参考提交 `2d4f9647`，TP8 热前缀在原生 ACL graph 与 MK 间做同机三臂对照。补齐环境、运行件、计划及逐 rank 验收 | 原参考只在命中可接管前缀时获得长前缀收益。原版冷请求试验中，八 rank 明确拒绝接管，7,112 次预期执行里实际 MK 为 0。161.295 ms 属于原生 eager 回退 |
| 11-24，`a6d3c407c`、`da1ce9fa9` | 新增独立冷 128/128 overlay。在原生完成冷 prefill 后，确认请求生命周期和缓存绑定；首次 decode 导入原生 KV，后续同步写回。只改接管层，保留参考图和八个静态计划 | 7,112/7,112 次实际 MK，56 次首次导入及 7,056 次后续复用。正式 TPOT 下降 8.21%，完整 ID 与显存见下节 |

上述早期 MTP 和 target 历史数据可从 `mega-glm/baselines/glm53-mtp-final-20261121.json`、`glm53-mtp-paged-20261121.json`、`glm53-45l-native-graph-vs-mk-target-k1-20261123.json` 回查。早期普通单源 DP8 各负载、逐 rank 回执和原生回归在同目录 `glm53-dp8-ordinary-20261123.json`。

## 已验收的普通解码结果

所有正式对照均采用 45 层 GLM-5.3 Flash W4A4、Ascend 950DT 八卡、Model Runner V1，关闭 MTP。原生臂使用 decode graph 配置。每条的 TPOT 按首末输出到达时间差除以输出间隔数计算。热前缀取三次请求均值，短请求取逐请求中位数。每行原生与 MK 使用同一台机器。

| 配置 | 原生 TPOT（ms/token） | MK TPOT（ms/token） | 输出与执行证据 |
| --- | ---: | ---: | --- |
| 184，TP8 + DP1 + EP8，单并发，261,120 个缓存前缀 + 1 个新输入，200 输出 | 17.859 | 13.090 | TPOT 降低 26.70%，三轮共 600/600 个输出 ID 与黄金相同，每轮 1,592 次逐 rank 回放 |
| 118，TP8 + DP1 + EP8，单并发，冷 128 入 / 128 出 | 14.733 | 13.523 | TPOT 降低 8.21%，7,112/7,112 次逐 rank 执行，5/7 条整串相同，689/896 个位置相同，第 4、6 条分别从第 2、39 个输出分叉 |
| 103，TP1 + DP8 + EP8，八并发，128 入 / 128 出，严格 PRE | 18.061 | 26.590 | 3,048/3,048 个 MK 步，整机输出 417.265 → 280.286 token/s，13/24 条与原生完全相同，1,932/3,072 个位置相同 |
| 184，TP1 + DP8 + EP8，八并发，128 入 / 128 出，可选 mHC PRE 06 | 19.034 | 25.399 | 3,048/3,048 个 MK 步，整机输出 399.495 → 295.904 token/s，7/24 条与原生完全相同，1,377/3,072 个位置相同 |

184 的 DP8 原生臂有 3,044/3,048 步实际 ACL replay。其余四步处于另一批 rank 同时 prefill 的窗口。只比较实际逐步 replay 的第 1、3 波时，原生为 19.003 ms/token，PRE 06 为 25.469 ms/token，仍慢 34.03%。

冷 128/128 独立显存复测用 DCMI 验证了两臂各 7/7 条请求在首末输出之间完成的八卡读取。原生与 MK 的八卡整卡合计观测峰值为 734,111 和 732,674 MiB，作业进程每卡约 86,975 和 86,790 MiB，带采样 TPOT 为 14.670 和 13.501 ms/token。每个实现与其不带高频采样的先前运行均为 896/896 个输出 ID 一致。

三种拓扑和前缀负载分别解读。八并发的每请求 TPOT 和整机输出吞吐分别评价，单流收益不能外推到 DP8 八并发。

## 影响精度和并行的几个结论

- **DP8 八并发已经真正进入目标图。** 每个活跃 rank 维护独立 KV。C16 指客户端并发，若每 rank 仅一槽，最多仍是八个请求同时执行。历史 R1 配成每 rank 两槽时曾发生 6,096/6,096 个步回退并测得 171.449 ms/token。独立双行 MK 路线通过了执行计数，TPOT 33.969 ms，16 个相同输入的同 rank B1 对照中 0/16 条整串相同。[DP8 验收](docs/source/developer_guide/glm53_multi_dp.md) · [双行记录](docs/source/developer_guide/glm53_two_row_experiment.md)
- **mHC PRE 06 的收益来自同机 A/B/A。** 184 在 DP8 八并发下，严格 MK 27.075 ms 降到 25.399 ms，下降 6.19%，整机吞吐从 278.142 提高到 295.904 token/s。对原生，候选 TPOT 仍高 33.44%，24 条中 17 条分叉，最早第 2 个输出。该算子组合在 MegaKernel 侧保持可选。
- **本地 TP8 typed owner 仍在取证。** 118 的配对 smoke 中，同一 token 的实际派发行由 64 降到 8，首步第 3 层专家 ID、按槽还原的 BF16 combine 均一致，但 FP32 TopK 权重摘要已经不同，第 4 层开始专家 ID 也分叉。两个 16 输出样本分别在第 6、2 个 token 分叉，未进入正式计时。原码长前缀及已验收冷适配的性能只归各自固定运行时。
- **回退与写回。** 先做全组接管资格判定，再进设备 collective。在设备写回开始后遇到异常应终止这一轮，避免在同一 token 上让原生重复写缓存。需要关联 per-rank 真回放和输出序列，单看环境开关无法确认接管。

## Model Runner V2 迁移边界

V2 适配处于设计阶段。首先在选用 `VLLM_USE_V2_MODEL_RUNNER=1` 的独立环境跑通同权重的原生 graph。把 V1 的请求位置、生命周期和缓存视图映射到 V2 的 `InputBatch`、`req_states`、`block_tables` 和逐层 metadata。MK 接在 V2 缓存准备完成后的目标 forward 边界，以 hidden state 进入 V2 自己的 LM head、采样及状态更新。先取 TP8 单请求冷 128/128 普通解码的逐 rank、逐 token、显存和时延，再覆盖热前缀及 DP8。

自带 MTP 的迁移要把 draft 保留在 V2 原生 speculator，MK 仅运行 target 验证。KDA conv/recurrent、MLA 写槽、KPool tail 的候选状态必须与 rejection sampler 的接受长度一致提交或回滚。常规单 token 的同步写回语义需要扩展才能覆盖这个阶段。GLM-5.3 Flash 的 V2 端到端结果仍待实机验证。[V2 Runner](vllm_ascend/worker/v2/model_runner.py) · [GLM-5.2 V2 MTP 回归](tests/e2e/pull_request/eight_card/model_runner_v2/test_glm5_2.py)

## 运行入口与证据

从集成工程根目录 `/mnt/share/l00623238/workspace/megakernel-agent/mega-glm` 调用 `scripts/run_glm53_reference255.sh`、`scripts/run_glm53_reference128_cold.sh` 或 `scripts/run_glm53_multi_dp.sh`。同一组对照由一个八卡租约覆盖。记录有效执行步、每条首次分叉、TPOT、吞吐和显存窗口。较新的实验集中在工程同级的 `dp8-pool-20261123-PG140j`。

- [普通解码配方、CPU 检查和历史性能](docs/source/developer_guide/glm53_dp8_decode.md)
- [单机同步迁移和差分验证](docs/source/developer_guide/glm53_local_dp_sync.md)
- [C8 正式流量和 R2 短程设备路由证据](docs/source/developer_guide/glm53_multi_dp.md)
- [冷 128/128 复现与逐请求精度和显存](examples/recipes/glm53_ref_cold128/README.md)

本次本地源码快照为 vLLM Ascend `a1b4ad89c`、MegaKernel `aafe0e6f3`。它保存工作树中的 GLM-5.3 Flash 模型、W4A4/KDA、V1 hook、外部 target 和历史外部 draft 相关源码与测试。表中结果来自各历史独立冻结包，当前整个快照的回归状态应以新实测为准。

本次保存后通过 Python 语法与 Git 空白检查。118 容器中以 CPU 复跑定长门禁与多来源协议，共 18 个用例通过。两仓同一回执还覆盖 MK 侧的 3 个用例。回执位于上述实验根的 `source/optimization-trajectory-20261124/evidence/glm53-checkpoint-cpu-glm-4.7-154440.json`。本轮未为代码快照重跑八卡整网。
