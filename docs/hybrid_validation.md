# 三输入与混合统计版本

## 0.6.0：样本 compact 复用与 CPU 预取

修复 portable reader 的 GPU 能力识别：旧检查只识别原 GDS 的 flat reader，使三输入 gene 准备回落到 CPU sparse 路径。新版在原显存门控内使用 reader 声明的 CUDA 能力。gene 的并集与各 mask 的独立列选择继续保留。

派生 compact 缓存绑定完整样本与变异轴、MAC 阈值、人口来源和解码版本，保存原 dtype 的六状态异常值及 allele summaries。同一人群的多个表型可复用；新的样本 subset 使用不同条目。默认共享磁盘预算 64 GiB；每 worker 的 CPU 预取队列为 2 条、256 MiB，GPU 计算仍在消费线程执行。参数、预算边界及禁用方法见[三输入指南](cache_only_run.md#按样本与-mask-复用-compact-数据)。

### 最终多进程版验证

最终冻结实现 `51810fd` 启用 CPU 准备进程池，在真实 339,013 人上重新运行以下四项作业。模型先由三输入独立拟合，参考和两组候选读取同一个新模型；冻结 0.5.0 参考 `cba9fed` 在本轮重新计算，没有复用早期结果。三个配置在同一 GPU 顺序执行。两组候选共比较 44,600 个 P/logP 值，最大 `|Δ(-log10 P)|=0`，严格小于 0.001，未使用精度例外；原生输出结构、非概率字段、校验和与逐值 readback 均通过。相同最终源码另有 275 项 GPU 组件测试及 33 项子测试通过。

| 选定作业 | 本轮冻结 0.5.0 | 最终版首次构建 | 最终版命中 |
|---|---:|---:|---:|
| Single 区间，11,063 行 | 50.140 s | 46.927 s | 34.537 s |
| coding，5 行 | 18.636 s | 9.461 s | 7.378 s |
| noncoding，6 行 | 115.758 s | 105.179 s | 91.665 s |
| ncRNA，M=9854 | 144.430 s | 101.562 s | 87.052 s |
| 整组启动至完成，不含模型拟合 | 342.114 s | 275.148 s | 228.534 s |

本轮候选每组使用 1 个 GPU 主线程、1 个读取/校验线程和 4 个 CPU 准备子进程；冷组四个子进程均核验为未初始化 CUDA，暖组全部命中，因此没有启动准备子进程。生产默认按全部 worker 共享的额度自动分配：40 核与 8 张卡时为 8 个主线程、8 个读取线程、16 个准备子进程、4 核后台预算和 4 核预留，每张卡分到两个准备子进程。这是默认生产分配，四项计时采用上述单卡四子进程配置。

冷组从空派生缓存发布 271 个条目，准备子进程合计读取 300 个人口 frame；暖组 271 次命中、人口 frame 读取为 0。普通 reader 计数已包含子进程完成量，不再重复相加。多个 reader 重复读 frame，使冷组读取次数高于参考的 204 次。冷组子进程累计 CPU 时间为 43.515 s、累计 wall 为 69.257 s，父线程读取/校验为 14.592 s；暖组父线程读取/校验为 15.655 s。这些与 GPU 计算和输出重叠，不能相加为整组耗时。noncoding 的共享注释索引设置仍耗时 109.170/103.996/90.709 s，是该作业的主要成本。

本轮保持 27 GiB 预飞预算与原统计参数以覆盖双 panel，峰值 reserved 为 24,396 MiB；生产每 worker 上限为 40 GiB。64 GiB 为共享 compact 磁盘预算，256 MiB 为每 worker 父队列预算，两者均不是总 RSS；子进程工作区、reader 缓存和消费数据由独立主存门控处理，默认主机预算 200 GiB、余量 20 GiB。长 mask 的 rank512、seed1729、probes0 及近似标记均已核验。冷暖只描述 compact 缓存，OS page cache 和共享 GPU 外部负载未受控。本轮验证相对冻结 TF32 参考的选定作业精度；官方 R 与全基因组验收另行完成。配置、来源哈希和分步骤记录见[最终多进程匿名报告](../benchmarks/torchstaar_cohort_compact_process_preflight_2026-10-09.json)。

### 早期线程版验证

初轮实现 `6474f20` 使用真实 339,013 人的三个配置，在同一 GPU 顺序运行，分别为冻结 0.5.0、首次构建 compact 的 0.6.0、命中 compact 的 0.6.0。每组包括 Single 选定区间、普通 coding/noncoding 和一个 M=9854 的 ncRNA 长 mask；同一模型与作业输入保持哈希绑定。两组候选共比较 44,600 个概率和 logP 值，最大 `|Δ(-log10 P)|=0`；原生输出结构、非概率字段、文件校验和与 readback 均通过。该初轮 GPU 组件另有 198 项测试及 24 项子测试通过。

| 选定作业 | 冻结 0.5.0 | 0.6.0 首次构建 | 0.6.0 命中 |
|---|---:|---:|---:|
| Single 区间，11,063 行 | 39.143 s | 61.668 s | 38.905 s |
| coding，5 行 | 15.224 s | 9.712 s | 7.310 s |
| noncoding，6 行 | 110.585 s | 94.636 s | 91.817 s |
| ncRNA，M=9854 | 178.826 s | 107.777 s | 105.238 s |
| 整组启动至完成 | 354.025 s | 281.260 s | 253.817 s |

首次构建发布 271 个条目，人口 frame 读取为 204 次；命中组 271 次命中、人口 frame 读取为 0。Single 首次写缓存耗时 22.616 s，命中读取及校验 16.444 s，因此本区间的 Single wall 基本持平。gene 提速同时包含 GPU 准备入口修复，不能全部归因于 compact 缓存。noncoding 首次共享注释索引准备为 101.175/92.908/90.840 s，仍占主要时间。

这三组保持旧预飞的 27 GiB 预算以触发双 panel，生产上限仍为 40 GiB，峰值 reserved 为 24,396 MiB。长 mask 的 rank512、seed1729、probes0 和近似标记均核验。冷暖仅指派生 compact 缓存，OS page cache 未清空；这是共享环境中的单次观测。准备、协方差、eigen/tail 与输出的分项记录见[匿名报告](../benchmarks/torchstaar_cohort_compact_preflight_2026-10-09.json)，嵌套阶段不可相加。本次对照验证相对冻结 TF32 实现的精度，未完成全基因组或官方 R 验收。

计时累加锁修正后的实现 `0ab6976` 重新运行首次构建与命中两组，wall 分别为 282.503/216.347 s；44,600 个概率及 logP 比较的最大差仍为 0，原生结果完整性通过，GPU 组件为 201 项测试及 24 项子测试。此复测复用了上表已核验的旧版参考结果，未重新计算参考组；参考与两组候选属于不同轮次，时间不能解释为同轮配对加速。来源、复用证据、分项时间与校验见[线程版复测记录](../benchmarks/torchstaar_cohort_compact_thread_retest_2026-10-09.json)。

### 独立多 CPU 准备测试

独立 CPU 准备测试也绑定初轮实现 `6474f20`。在同一真实样本集上固定 64 个请求（Single 26、coding 5、noncoding 7、ncRNA 26），每个 CPU 配置使用全新的派生缓存，随后测命中读取。独立 spawn reader 只接收请求 ID，返回校验与计时；prepared 数组不跨进程复制。1/2/4 进程的六组 prepared 数据逐字段 bytehash 一致。

| CPU 进程数 | 首次准备及存储 wall | 命中读取 wall | 冷准备进程峰值 RSS 合计 | 人口 frame 读取 |
|---|---:|---:|---:|---:|
| 1 | 9.431 s | 3.640 s | 566 MiB | 33 |
| 2 | 6.048 s | 3.307 s | 1,088 MiB | 46 |
| 4 | 4.792 s | 3.020 s | 2,003 MiB | 65 |

墙钟包含进程启动、reader 初始化、读取/准备/存储或命中、digest 与退出；各进程初始化约 1.6–1.9 s。4 进程冷准备约为 1 进程的 1.97 倍，但独立 reader 重复读取同一个人口 frame，CSR 压缩读取量从 51.5 MB 增至 107.5 MB。每组派生文件约 114.2 MB；命中时完整 blob SHA 更新 CPU 合计约 0.09 s，这一小批请求的校验不是主要成本。实际磁盘冷读未受控。

这些是读取与 compact 准备的微 benchmark，未运行 GPU、统计和原生输出，也未重放完整 gene union fallback 或 Single 有效列打包。上述两个冻结线程版本均为每个 GPU worker 一个 CPU 预取线程；该微测试未启用生产多进程准备池，worker 准备后再由主进程重新校验整 blob 的额外成本也未在此测试中发生。完整匿名分项见[CPU 准备记录](../benchmarks/torchstaar_cpu_compact_pool_2026-10-09.json)。

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
