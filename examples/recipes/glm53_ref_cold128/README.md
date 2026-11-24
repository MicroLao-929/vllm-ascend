# GLM-5.3 Flash 参考内核的 128/128 冷解码实验

这份配方针对 45 层、TP8、DP1、EP8、单请求、关闭 MTP 的 GLM-5.3 Flash。原生模型完成 128 个输入 token 的 prefill。EP worker 在首次 decode 导入本请求 KV，后面复用相同的私有缓存并同步写回。实验通过 `MK_GLM53_REFERENCE_COLD_SHORT=1` 显式启用。普通安装及默认原生执行入口保持原配置。

配方以 GitCode 固定提交 `2d4f96473945a4a1c4149d0360eeb8b64e848369` 为输入，生成独立 overlay。25 个 overlay 文件中只有 `glm53_mk_ep.py` 有改动。原生和 MK 两个服务均关闭前缀缓存，最大上下文继续使用 262,080。原生使用 FULL_DECODE_ONLY 图，MK 主模型保留参考分支的权重与设备库、TP 复制 token 的 typed owner 以及原生 prefill。

```bash
python3 -IB check.py "$ORIGINAL_REFERENCE"
python3 -IB prepare.py "$ORIGINAL_REFERENCE" "$NEW_BUILD"
bash "$NEW_BUILD/run.sh" native "$NEW_GROUP/native" required
bash "$NEW_BUILD/run.sh" mk "$NEW_GROUP/mk" required
python3 -IB "$NEW_BUILD/audit.py" compare "$NEW_GROUP/native" "$NEW_GROUP/mk" "$NEW_GROUP/compare.json"
```

`run.sh` 消费已获得的独占八卡租约。两个服务都使用相同的 1 次预热和 7 个固定合成输入，贪心生成 128 个 ID。选项 `required` 要求八个 rank 全部应用原计划，文件 SHA256 为 `5ee116026e720cdd348a5ed164713dfcbd6edfdf3d69b297c3bb55d6dbea03a5`。选项 `stock` 显式移除离线计划。两种模式单独验收，设备权重和并行布局一致。

## 实机结果

| 工作负载和实现 | 原生 graph | 对照服务 | 真实 MK 覆盖 | 输出一致性 |
| --- | ---: | ---: | --- | --- |
| 原码，命中 261,120 前缀，再生成 200 | 17.859 ms/token | 原参考 MK 13.090 | 每轮 199 × 8，三轮 | 三臂各 600/600 与原黄金相同 |
| 原码，128 入 128 出，缓存命中 0 | 15.219 ms/token | 原生 eager 回退 161.295 | 0/7,112 | 回退与原生 graph 为 896/896 |
| 本冷适配，128 入 128 出，双方禁用前缀缓存 | 14.733 ms/token | 实际 MK 13.523 | 7,112/7,112，八 rank 应用原计划 | 5/7 条完整相同，689/896 个位置相同 |

三行各自使用同一台 A5 的原生对照。长前缀的普通 TPOT 降低 26.70%，两次原生均值作分母。冷适配的普通 TPOT 中位数降低 8.21%。该冷测试中第 4 条在第 2 个输出 token 首次分叉，第 6 条在第 39 个首次分叉，其他五条的 128 个输出全部相同。首次分叉后继续贪心生成，所以后续位置的差异包含不同历史输入的影响。该配方保留实验状态。

`audit.py` 核对每条 HTTP 结果、128 至 254 的每个位置、八卡真实 MK 回执、首轮 KV 导入和计划。实机另从每个 worker 的 PID 和 TP/EP rank 检查 56 次各请求首次导入和 7,056 次后续复用。所有 7,112 次图执行均有同步写回。CPU 测试使用真实原始方法、可追溯的计划日志片段和反例，不计算设备性能。

[机器可读的短解码实测](measured-cold128.json)　[上游固定分支](https://gitcode.com/xupengpeng1/megakernel/commit/2d4f96473945a4a1c4149d0360eeb8b64e848369)

## 正式入口的显存复测

118 的已部署冷适配入口以同样的七条 128/128 请求复测。每个请求的首末 token 到达之间均有一次完整八卡 DCMI 扫描，读取开始、结束时的 worker 身份均属于当前服务。下表为七次观测中同一次扫描的八卡总峰值及各卡进程峰值范围。

| 模式 | 有效请求窗 | 八卡板卡总用量峰值（MiB） | 八个作业进程合计峰值（MiB） | 各卡作业进程峰值范围（MiB） |
| --- | ---: | ---: | ---: | ---: |
| 原生 graph | 7/7 | 734,111 | 695,796.84 | 86,974.59–86,974.71 |
| 冷适配 MK | 7/7 | 732,674 | 694,324.19 | 86,789.84–86,790.84 |

该独立采样段的 TPOT 中位数为原生 **14.670 ms/token**、MK **13.501 ms/token**。主性能结论使用先前没有这项高频采样的 **14.733 与 13.523 ms/token**。每侧与自身先前输出均为 7/7 条、896/896 个位置完全一致，两侧之间仍为 689/896 个位置一致。
