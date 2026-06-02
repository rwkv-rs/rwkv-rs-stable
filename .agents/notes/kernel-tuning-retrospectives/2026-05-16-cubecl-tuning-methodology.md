# CubeCL Kernel 调优方法论复盘

- 日期：2026-05-16
- 范围：复盘 `.agents/notes/kernel-tuning/` 下的原始实验笔记，抽取 CubeCL / Burn kernel 调优的方法论。
- 边界：这是一份二次整理文档，不属于原始实验流水账；不修改 kernel 源码、autotune cache、benchmark baseline 或远端 mirror。

## 总结

CubeCL 调优的基本单位不是一次代码修改，而是一个完整实验边界：

1. 先说明要保护的模型阶段和 trace contract。
2. 再判断热点属于哪类 surface：项目自有 kernel、Burn/Cubek matmul、通用 elementwise、固定 dispatch，还是递归算法边界。
3. 在动手前写好分支级 note，记录已有证据、假设、命令和 keep/revert 条件。
4. 每次只做一个修改或一个 profiler run。
5. 先验证 trace correctness，再看标准 timing，再用 targeted profiler 做归因。
6. 只有当 targeted win 能撑住整个实验边界时才保留。

这批失败实验不是浪费，它们把边界摸清楚了：

- WKV7 是算法和 recurrence contract 问题，不是继续调 `row_tile` 或 shared memory 的问题。
- BF16 LayerNorm 是确定性数值边界问题，不是普通 block size 竞速问题。
- `lm_head/projection` 和 residual epilogue 是 Cubek/TMA operator boundary 问题，不是项目里写一个 scalar kernel 能解决的问题。
- `LocalTuner` 是 dispatch 选择工具，不是 accuracy oracle，也不意味着每个参数都值得 tune。

## 做对的地方

### 1. 替换真实的 launch 链

`weight_decay_transform` 是最干净的成功案例。它把 Burn 表达式里的 scalar/unary softplus transform 链替换成一个项目自有 CubeCL kernel，语义边界很窄，key 覆盖 hardware/shape，远端 activation/timing 通过，并且 `nsys` 证明目标 generic launch 被替换或大幅减少。

这个模式值得复用：找到一个真实存在的 launch 链，确认输出 contract 很窄，然后用项目自有 kernel 精确替换。

### 2. 在既有 kernel 内做小的算法清理

GB10 上的 `lm_head_l2wrap_ce` direct target-logit loading 是有效的小改。它没有改变 full-vocab denominator scan，只是移除了一个几乎全零的 target reduction，row kernel registers 从 `53` 降到 `40`，targeted `nsys` 从 `13.879ms / 3` 改到 `12.851ms / 3`。

这类小改的合格标准是：trace correctness 通过，targeted profiler 确认目标 kernel 变好，并且标准 compare 没有引入可归因的整体回退。

### 3. 正确区分 dispatch-correctness 和 speedup

`value_residual_gate` vector-axis 修复值得保留，因为它修正了候选集资格判断：`gate_base` 应该按自己的 1D embedded axis 判断，sequence tensor 按 3D embedded axis 判断。这个修复让 dispatch 从 `line_size_1` 变成 `line_size_2`，activation 通过，总体 speedup 仍然大于 `1.0`，但 targeted `nsys` 是中性或略慢。

正确结论是：这是 dispatch-correctness fix，不是性能提升。调优文档必须能保留这种修复，同时明确不声称 speedup。

### 4. 把 autotune key 和 fixed policy 账算清楚

autotune key audit 最终收敛出的规则是：

- Tuned candidate 的 key 必须覆盖 runtime/backend、dtype、shape、rows、hardware fingerprint、候选参数、alias/in-place 状态和 deterministic policy。
- 候选参数必须体现在 tunable name 或 group 里，这样 persistent cache checksum 才会随候选集变化失效。
- Fixed-dispatch kernel 可以存在，但以后改常量时，要么先引入 keyed dispatch/autotune wrapper，要么明确记录 fixed-policy 证据。

这不是性能优化本身，而是防止 stale cache 和单机硬编码常量污染后续实验。

## 犯过的错误

### 1. 实验边界没立住就开始跑

重复最多的问题是流程错误：

- 重跑已经记录过的负例，尤其是 residual-add / Burn-add 这类实验。
- 在共享 dirty tree 里混多个实验，导致代码状态和证据边界不清楚。
- 一口气改很多东西、跑很多命令，最后再补写总结。
- 把 `cargo check` 当性能验证。
- 把第一次 retune 的 timing 当 steady-state timing。
- 把 profiler mode 下 `repeat/warmup` 不匹配的 `.time.json` 当 acceptance timing。
- 远端操作里重复踩坑：没带 SSH key、zsh 的 `status` 是只读变量、远端没有 `rg`、远端 mirror 没有 `.git`。
- 在已经证明 `ERR_NVGPUCTRPERM` 后，还尝试用普通用户跑 `ncu` counter。

这些不是小失误，会直接污染调优结论。

### 2. 误用 LocalTuner

反复出现的错误假设是：加一个 `LocalTuner` 候选就更高级、更可能更快。实际结果多次反证：

- KeyPrepare `warps_per_cube` 选了 `2`，block 数翻倍，targeted profiler 比固定 `4` 略慢。
- WKV7 output factorization 是合法候选，但额外 per-step reduction/sync 的成本超过了省下的算术。
- 多个 line-size 选择只是近似打平，强行放宽 vector width 或清 cache 只是在追噪声。

`LocalTuner` 的正确用法是：

- 只 tune 语义上不同、数值安全的候选。
- 候选集要小。
- key 必须覆盖影响正确性和性能的维度。
- 第一次运行当作 retune/warm-cache setup。
- 必须读 autotune log，确认实际 winner。
- 再用 warm standard compare 或 `nsys` 判断 winner 是否真的帮助了目标 stage。

`autotune-checks` 只比较候选之间的一致性，不知道项目 trace tolerance，也不能替代 trace-backed correctness。

### 3. 低估 BF16 数值边界

LayerNorm 暴露了最主要的数值陷阱。BF16 `D=768` 下，小 block size 可以更快，但仍然不合法。普通 `256`、`512`、`768` 都出现过 downstream drift。CPU 诊断证明 intended ordered-256 math 可以精确匹配 safe `1024` reduction order，但重建的 device ordered-256 kernel 仍然 activation 失败。

结论比“reduction 要小心”更强：一个 reduction candidate 不是因为代数等价就可以进入候选集，必须由 device implementation 证明它保持相同 observable trace boundary。这个 repo 里，BF16 LayerNorm candidate 必须先有 deterministic guard 和 trace-backed proof，才允许参与 timing 竞争。

### 4. 只盯局部 kernel

有些实验看局部是正的，但整体边界失败：

- GatedReadout row-pack 在 `nsys` 里让 combine kernel 变快，但标准 compare 比保留的 two-warp 实现更差，所以回退。
- KeyPrepare launch packing 在 autotune 里接近噪声，在 targeted profiler 里略慢。
- residual broad fusion 让部分 `.time.json` row 看起来更好，但 `nsys` 证明 residual `kernel_binop_c_bf16_n_8` 仍然存在，而且总 CUDA kernel time 更差。

规则是：targeted profiler 负责解释，标准 trace compare 负责裁决。局部 kernel row 赢，不等于 stage 或 total boundary 赢。

### 5. 选错 operator boundary

matmul 相关实验反复撞到同一堵墙：

- `lm_head/projection` 是 Cubek/TMA BF16 matmul。
- projection 后面的 residual add 是单独 binop launch。
- Burn/CubeCL 公开 matmul path 暴露的是完整 output tensor binding，没有项目侧可用的 TMA epilogue hook。
- broad Burn fusion 没有消除 residual binop，还让总 CUDA kernel time 变差。
- naive project-local row-dot projection/loss fusion 会丢掉 GEMM/TMA 质量，而且还缺 backward contract。

正确结论是：projection/loss fusion 和 residual matmul epilogue 属于 upstream/operator-boundary 工作。有效方案必须是 GEMM 质量的 fused operator 或 Cubek/TMA epilogue，并且要有匹配的 backward 语义。

### 6. 把 WKV7 当成 launch 参数问题

WKV7 是最典型的“微调不能替代算法设计”的例子。

当前 fast output path：

- 对 `B=16,H=12,row_tile=64` 只启动 `12 * 16 = 192` 个 cube。
- 每个 active unit 都在 serial `T=512` recurrence 里保存一个 `Array<f32>(head_size=64)` 状态。
- `row_tile=64` 能最小化重复 shared input-vector load，所以更小 row tile 虽然增加 block 数，反而更慢。

已经拒绝的方向：

- 强制更小 row tile。
- shared-lanes / shared-state reduction。
- 不走 shared staging，直接 global load `value`。
- output factorization，额外 per-step reduction。
- dense segment transform/scan/recompute，带全局 f32 `P/Q` tensor。
- low-rank / state-scan 变体，仍然需要昂贵 recompute 或 dense correction。

下一次 WKV7 工作必须改变 recurrence contract 或 state representation。它要么增加真实并行工作，要么减少 per-row live state，并且不能重复 global f32 transform tensor、重复 input load、高同步 shared-state reduction 这些失败模式。

## 重复犯的错误模式

1. 看到 profiler hotspot 就直接改代码，没有先分类 surface 和查重 notes。
2. 给本该 fixed-policy 的参数加 autotune candidate。
3. 从单机 timing 直接强行固定 line size、warp count、block size，没有 hardware/shape/deterministic key。
4. 在 acceptance boundary 是远端 standard compare 时，错误相信本地 timing 或 profiler-mode timing。
5. 目标 kernel 变快就宣布成功，忽略 stage/total compare 变差。
6. 说要“减少 launch”，但没有用 `nsys` 证明 launch 真的消失。
7. 对 BF16 reduction 只看代数等价，不看 device trace boundary。
8. 用项目侧 scalar kernel 去碰 Cubek/TMA matmul surface。
9. 让 stale cache、stale binary、错误 branch/path、错误 baseline 混进结论。
10. 事后才把错误写进 skill/guardrail，而不是把 preflight ledger 当成正常起点。

## 正确的 CubeCL 调优范式

### 1. 从 contract 开始

先写清楚：

- 要保护的 model stage 和 trace output。
- shape、dtype、hardware、baseline path。
- 输出是 public API contract、training-only loss boundary，还是内部 tensor。
- backward semantics 是否属于本次修改的一部分。

如果修改不能保护 observable contract，它就是设计项目，不是调优 patch。

### 2. 分类 surface

| Surface | 正确杠杆 |
| --- | --- |
| 项目自有 elementwise/reduction 链 | 在 trace 语义精确时，融合成窄 custom kernel。 |
| 项目自有 kernel，且存在离散安全候选 | 用 LocalTuner 或 keyed runtime dispatch。 |
| Fixed launch policy | 保持 fixed，除非新分支引入 keyed dispatch 或证明新 fixed policy。 |
| BF16 reduction | deterministic guard + trace-backed candidate admission。 |
| Cubek/TMA matmul | upstream-style epilogue/operator work，不能用 scalar row-dot 替代。 |
| WKV7 这类 recurrent state kernel | 做算法或 state representation 设计，不继续 churn launch constants。 |

### 3. 先设计 candidate admission，再谈 timing

一个 candidate 允许进入候选集，必须满足：

- 数学语义对目标 stage 合法。
- BF16 / deterministic 行为在 selection 前被 guard。
- 候选参数进入 tunable name。
- autotune key 覆盖 runtime、dtype、shape、rows、hardware fingerprint、alias/in-place、deterministic policy。
- 已知 invalid candidate family 在 tuning 前排除。

### 4. 分清每种测量的角色

- `cargo check`：compile / wiring gate。
- candidate/cache 改变后的第一次 standard compare：retune / warm-cache setup。
- warm standard compare：acceptance timing + activation。
- `nsys`：归因、launch geometry、registers、shared memory、launch count、目标 kernel 是否消失。
- `ncu`：只有权限允许时才用于 occupancy / SOL / stall counters。

profiler mode 下 `repeat/warmup` 不匹配的 `.time.json` 只能当 diagnostic，不能当 acceptance。

### 5. 用整体边界做 keep/revert

保留条件：

- activation 通过 trace fixture。
- 对应 standard compare boundary 可接受。
- targeted profiler 支持假设。
- 没有引入可归因的 broad regression。
- note 记录了代码状态、cache/binary provenance、remote/local 边界。

回退条件：

- activation drift。
- 目标 kernel 变快但 end-to-end stage 变差。
- launch count / kernel name 证据否定了原假设。
- 改动依赖不属于本分支的 broad backend behavior。
- 收益只是噪声级，同时增加维护成本。

### 6. 保存负例证据

失败分支有价值，前提是记录：

- 精确 changed boundary。
- 命令和 invalid command。
- activation / timing / profiler 结果。
- 失败原因。
- 未来什么条件变化才算 materially different。

不要删证据。只有当 guardrail 能阻止真实重复错误时，才把它写进 skill。

## 下一次调优前的清单

1. 搜索 `.agents/notes/kernel-tuning/`、`.agents/skills/kernel-tuning/SKILL.md` 和相关 memory，查 kernel、shape、backend、hardware、baseline、binary、optimization idea。
2. 如果同一边界已经记录过，停止。
3. 创建或选择本次 attempt 的准确 branch/worktree。
4. 先写 note：dirty-tree constraint、matched evidence、hypothesis、candidate parameters、commands、keep/revert boundary。
5. 只做一个 edit 或一个 command。
6. 在下一个 command 前追加结果。
7. invalid command、stale binary、stale cache、wrong baseline、wrong shell、profiler mismatch 都要明确标记为 invalid evidence。
8. 结尾写清 keep/revert state；必要时才补一个简洁 skill guard。
