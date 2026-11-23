# GLM-5.3 单机 DP8 解码元数据同步实验

普通解码时，vLLM Ascend 原有路径通过 CPU Gloo all-reduce 收集各 DP rank 的 token 数和图模式。这个实验面向同机八个进程、TP1、DP8、EP8、单请求、关闭 MTP 的 GLM-5.3 Flash，替换相邻两个成功的 MegaKernel target decode 之间的一次元数据交换。数学算子、专家通信、原生 prefill 和 token 采样沿用当前实现。

## 已测时长

| 测量对象 | 样本 | 中位时长 |
| --- | --- | ---: |
| 原生 graph 的具名 Gloo all-reduce，第一次采集 | 6 步 | 1.434 ms |
| 原生 graph 的同一 Gloo 调用，第二次采集 | 6 步 | 1.485 ms |
| MK 请求 rank 的 `_determine_batch_execution_and_padding` 整段 | 252 个真实 decode 步 | 1.539 ms |

最后一行包含执行模式和 padding 的其它准备工作。以上均为既有路径的测量。本候选的八卡命中率、逐 token 一致性和端到端 TPOT 收益仍待同机 A/B 实测。

## 构建和使用

用已有的 v0.30 环境生成独立目录。直接启动原有原生 graph 服务时继续使用原包。下面的 `V030_ASCEND` 与 [DP8 普通解码配方](glm53_dp8_decode.md) 相同，可指向固定的公开版或已准备的部署版。

```bash
python3 examples/recipes/glm53_dp8/prepare.py "$V030_ASCEND" "$NEW_ASCEND" --local-dp-sync
export PYTHONPATH="$NEW_ASCEND${PYTHONPATH:+:$PYTHONPATH}"
```

若需与延迟发布页表联合验证，可在创建另一个新目录时同时传 `--lazy-tables --local-dp-sync`。对照与候选均沿用原配方的 `GLM53_PLAIN_COMPACT_GATE=1` 与 `GLM53_DP8_LOCAL_CONTROL=1`，保持其它开关相同。

| MK 服务进程的 `MK_GLM53_LOCAL_DP_SYNC` | 运行方式 |
| --- | --- |
| `off`，默认 | 走原有 Gloo 流程 |
| `observe` | 走 Gloo，记录候选资格与同步调用时间 |
| `on` | 在八卡一致认可的连续普通 decode 上使用独立本机通道 |

首次 prefill 没有上一轮成功的 MK 记录，直接执行原有 Gloo。上一轮完成后若新请求切入、位置或 owner 改变、出现多个请求、图模式变化或进入 draft，八个 rank 共同回到原有 reduce。通道初始化校验同一主机、命名空间、用户及 DP rank 顺序。帧损坏、epoch 不符或对端退出会中止当前交换。

## 验证与命中计数

```bash
GLM53_BASE_ASCEND="$V030_ASCEND" GLM53_TEST_ASCEND="$NEW_ASCEND" \
  python3 tests/ut/recipes/test_glm53_local_dp_sync.py
```

在公开及部署基座上分别构建 strict 和 lazy 版本，四种组合各用八个真实 CPU 进程逐步对比新旧 runner 返回值，包含两行八列的 int32 元数据。每个 rank 的 36 个差分步骤中有 15 步采用本地交换、21 步一致回退，覆盖来源轮换、首轮 prefill、非均匀 token 数、图模式、draft 和对端退出。部署基座的三份相关 worker 与冻结的候选源码逐字一致。固定同步模块的 SHA-256 为 `bc93662fd6510f907eb020d50447e86dedf5161fa698dad59cbc406938c73924`。

验收命中率时单独设置 `MK_GLM53_LOCAL_DP_SYNC_TRACE=1`。在输出包含八个 rank 的完整服务日志后运行以下命令，按请求、步序、位置、owner 和 rank 关联。正式测 TPOT 时关闭逐步日志，比较相同的 128 输入、128 输出请求。

```bash
python3 examples/recipes/glm53_dp8/local_dp_sync/count_hits.py "$SERVICE_LOG" --require-local
```
