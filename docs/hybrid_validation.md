# 三输入与混合统计版本

## 0.5.0 改动

主入口统一为 `run(phenotype_csv, covariate_csv, cache_directory, **hyperparameters)`。缓存内的标准数字样本轴取代原 GDS 物理行索引，CSV 按数字 eid 排序并逐表型采用完整观测。普通 Gaussian 零模型显式以 FP64 拟合，关联计算仍为 TF32/FP32。

Single 延续 0.4.0 的样本轴验证、MAC 安全上界预筛、压缩 frame 对齐、1024 有效变异合批及 allocator 自有缓存预算。5000 变异分组和统计公式保持原规则；输出按完整分组边界分片，保存全染色体行数、分片校验和及最终 factor levels。

长 mask 保留完整 FP32 协方差。4096 输出分块复用原始/加权 genotype 和投影，score/投影仍以 512 准备，正反方向协方差取平均。完整驻留不满足 40 GiB/live GPU 预算时采用双 panel；超限会报告失败。缓存协方差阶段结束后释放 genotype panel，后续统计预算独立检查，两个阶段的工作区不直接相加。

FastSKAT 使用前 512 个谱分量和残差两矩的连续卡方表示，再计算 saddlepoint tail。最终 Ritz 子空间、trace 和平方矩用分块 FP64 细化，完整矩阵保持 FP32；这是明确的谱计算精度边界。相消严重时直接分块计算残差，实质负残差会停止该 mask。输出记录近似方法、M、rank、seed、probes=0、exact-dense 残差矩来源和收敛护栏。该近似方法与 Davies 不完全等价。

跨 mask 的通用 FP64 batch 不用于 native TF32；native 路径保留每个 mask 内部的成熟权重批处理。普通 mask 采用完整谱和原 Saddle/矩匹配尾概率，不将完整谱解释为 Davies 精确尾部。

## 已有真实验证与本版范围

| 验证 | 样本与范围 | 计时 | 精度及范围 |
|---|---|---|---|
| 0.4.0 Single | 339,013 人、chr21，1,065,735 输出 | 作业 2331.276 → 1125.923 s；新版启动至输出 1183.551 s | 全部 P 对旧 TF32，最大 log10 差 7.0916163e-6；官方 R 只对 28 个 P |
| 缓存协方差 | 339,013 人、M=1428 | 4.024 → 0.493 s | U/V 逐位相同，协方差阶段 |
| 缓存协方差 | 339,013 人、M=9854 | 64.938 → 4.088 s | U/V 逐位相同，峰值 allocated 25.94 GiB；预热中位，不含数据读取 |
| 谱数值修复 | 8 个先前失败的真实 mask、16 个两权重同协方差 dense 对照 | 与另 2 个普通 mask 合计 398.012 s | 最大 log10 差 6.5639585e-5；两个普通 mask 的 132 个 P 逐位一致 |

上述记录来自分别冻结的真实运行；全基因组新输入、缓存逻辑轴和双 4096 panel 的端到端验证另行记录。SLQ 路径未通过精度验收，未用于生产默认。选定 mask 通过不能代表全基因组所有字段通过官方 R 对照。

2026-10-09 的本版三输入预飞使用真实 339,013 人、独立缓存元数据及同一 CSV 拟合模型。两张 GPU 分别运行缓存 4096 和旧 512 后端；Single 仅覆盖选定区间，另选一个 coding、一个 noncoding 和一个 M=9854 的 ncRNA 作业。两侧各 4 项全部完成，11,237 个概率值及 11,063 个 logP 逐位相同，原生输出结构与非概率字段一致，最大 `|Δ(-log10 P)|=0`。

| 本版选定作业 | 缓存 4096 wall | 旧 512 wall |
|---|---:|---:|
| Single 区间，11,063 行 | 36.716 s | 37.203 s |
| coding，5 行 | 13.020 s | 13.021 s |
| noncoding，6 行 | 100.162 s | 98.848 s |
| ncRNA，M=9854 | 135.279 s | 165.223 s |

本次将预飞预算设为 27 GiB，实际触发双 4096 panel；生产默认仍为 40 GiB。长 mask 协方差阶段为 `8.371 vs 40.419 s`，缓存峰值 allocated/reserved 为 `21,896/24,396 MiB`，协方差没有 D2H 复制。两侧同时运行于不同 GPU；缓存冷热和外部负载未受控，这些是单次观测，不能推导全量速度提升。完整分步骤计时见[匿名预飞记录](../benchmarks/torchstaar_three_input_preflight_2026-10-09.json)。该对照验证新旧 TF32 后端一致性，官方 R 精度和全染色体覆盖仍需分别验收。

GPU 组件检查另外覆盖了 513、4097、4609 列的 C/F 布局、完整驻留和双 panel；131 项检查通过。修复了 `M % 512 == 1` 时尾列因 GEMV 与 TF32 MMA 路径差异产生的协方差不一致，修复后 U/V 与旧 512 后端逐位一致。组件检查采用小样本，用于定位尾块错误，不作为真实人群 benchmark。

默认精度标准为各匹配概率字段 `|Δ(-log10 P)| < 0.001`。旧结果的单一已接受例外必须以结果和来源哈希绑定，不能对新模型、新输出或其他 mask 放宽标准。

阶段报告包含输入/metadata setup、读取与准备、score/协方差、统计尾部、输出、主存等待和总墙钟。host wall、CUDA stream 时间及嵌套阶段会重叠，不直接相加；共享 GPU 的实测耗时需注明共享情况。40 GiB 是 worker 的显存上限，不保证大 mask 可以存储完整矩阵或在限定时间内完成。

## 原实现和参考

原软件入口为 STAARpipeline 的 Individual_Analysis、Gene_Centric_Coding、Gene_Centric_Noncoding 和 Gene_Centric_Noncoding_ncRNA；原调用及完整参数见[完整流程](torchstaar.md#参考)。STAAR 使用的权重、Burden、SKAT、ACAT-V 与 CCT 算法来源保留在各实现的 SPDX 与函数说明中。

- [STAAR](https://github.com/xihaoli/STAAR)、[STAARpipeline](https://github.com/li-lab-genetics/STAARpipeline)。
- Lumley et al. (2018), *FastSKAT: Sequence kernel association tests for very large sets of markers*，Genetic Epidemiology，doi: [10.1002/gepi.22136](https://doi.org/10.1002/gepi.22136)。
