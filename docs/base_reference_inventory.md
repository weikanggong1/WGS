# Base STAARpipeline 的完整 mask 与原版对照

本页记录单个连续表型的 base STAARpipeline 范围和可执行原 R 对照。运行入口见 [全染色体流程](chromosome.md)。原 R 对照调用推荐环境中的 STAARpipeline 0.9.9、STAAR 0.9.9；其 pipeline 来源为 [作者维护的参考版本](https://github.com/yuxinyuanqt/STAARpipeline/tree/fbce778bf14cc4f9e892989a194c64bae2670311)。源码采用 GPL-3.0。

2026-10-05，chr21、一个连续表型的完整串行 GPU 流程通过原 R 严格验收。全部 795 项任务、18 个正式输出和 15 个 mask 均已覆盖：221 个 coding gene、221 个 noncoding gene、349 个 ncRNA gene，以及四个单变异区间。执行源码 SHA-256 为 `c6362c6d392dce8c29668a9563a88182714571ebc8aa68285e5988a7ad423300`；`executable_source` 包含 33 个执行及依赖定义文件（含参考 LAPACK 显式锁），不含文档。实际 33 个执行文件及 108 个快照文件的哈希、795 个任务的唯一名称、顺序和计时记录均已独立核对。

本轮单模式验收为 **18 项 `serial_vs_R` 关联文件比较、八份 Single 元数据原 R 读回（R/GPU 各四份）、一项 `serial_vs_R` 零模型比较**。18 个文件的对象结构、列类型及顺序一致，161,839 个数值字段、3,343,119 个数值单元格全部满足 `|GPU-R| <= 1e-10 + 1e-7*|R|`；结构差异和超容差单元格均为零。最大绝对差为 `1.0913936421275139e-10`，最大相对差为 `1.7739502638151633e-9`。原始表型经 RINT 和重新拟合写出的零模型，12 个数值字段、341,221 个数值单元格与 R 差异均为 0。最终独立审计只读原生文件元数据，不重新运行关联分析，审计 SHA-256 为 `165f6b661beeab443fdf703fc36b8ab0000b6c33e28850e76f49dc436cb044a9`。

四个 Single 文件均为 13 列，共 318,132 行，含 238,947 个双等位 SNV 和 79,185 个双等位 Indel；每行 N 为 42,652，无多等位结果、缺失 allele 或缺失 N。完整扫描的解码计数为 13,733,596 个位点，MAC 合格 318,132 个，REF_AF tie 实际为 1，半缺失和全部 allele 缺失计数为 0，最大 Bit2 层数为 2。八份原 R 元数据读回与正式比较报告及其文件哈希缓存完全一致。

按用户要求，本轮完整 GPU 实验只验收串行模式，`serial_completed` 退出码为 0；另一个批量模式在完成前取消，driver 的退出码 130 记录这次取消。取消的批量运行不提供完整性能结论。计时边界和完整串行性能见 [性能分析](performance.md)，此前版本的区域基准见 [benchmark](benchmark.md)。本冻结版本的 CPU 测试为 171 项通过、42 项跳过；A100 CUDA 四组专项为 67 项通过、0 项跳过。19 个真实集合的两模式局部回归各通过 1,336 个数值字段，仍单独保留其局部范围。

2026-10-05：一个敏感真实集合的 enhancer_DHS mask 含 1,216 个稀有变异，定位了 `5b0dfd83…` 的工作区差异。G、MAF、PHRED、U/V、权重和二次型与 R 逐位一致；MKL 查询得到 153,216 元素的工作区，原 Armadillo 固定使用 `66*1216=80,256`。前者产生 1,189 个谱舍入差，最大 `1.023e-12`，被原近均值 Saddle 放大。固定原工作区后，该 mask 的三个敏感谱与 R 逐位一致，全部 88 个数值字段通过严格对照；19 个集合两模式的最大绝对/相对差分别为 `8.899e-10` / `2.721e-9`。只修正工作区，未改统计公式、根搜索阈值或验收容差。

2026-10-05 的历史 `5b0dfd83…` 完整复跑在 325 项串行任务后被严格比较保护器中断，退出码 130；已完成 221 项 coding 和 104 项 noncoding，批量模式未开始。7 个正式关联文件中 6 个通过，Noncoding 367 的 19,140 个数值单元格有 3 个超出容差，结构差异为零，最大绝对差 `3.6844235752897525e-6`、最大相对差 `8.603064423370173e-6`。这三个字段来自同一 mask，已定位到上文的工作区差异，并在固定工作区的真实集合回归中通过；本轮 `c6362c6d…` 的完整串行结果见开头。该历史轮的原始表型经 RINT、拟合并写出的零模型也由原 R 回读严格通过：12 个数值字段、341,221 个数值单元格，结构差异和超容差单元格为零，最大绝对/相对差均为 0。该轮保留 interrupted / `strict_failed` 状态，源码、文件和日志独立保存，不用于完整速度结论。

2026-10-05 的历史冻结源码 `8520cf24…` 从原始表型重新拟合并写出零模型。独立原 R 回读的 `serial_vs_R` 比较严格通过：12 个数值字段、341,221 个数值单元格，结构差异和超出容差的数值单元格均为零，最大绝对差与相对差均为 0。该轮串行运行在完成 377 项任务后暂停，退出码为 130；8 个已比较的关联文件中 7 个通过，Noncoding 368 的 23,055 个数值单元格有 3 个超出容差，结构差异为零，最大绝对差为 `5.5230437395747956e-7`、相对差为 `1.496960491605094e-6`。批量模式未开始，完整验收状态保留 `strict_failed`；该历史轮未完成另外两个零模型及全套关联与单变异元数据检查。源码、文件和日志独立保存，不用于完整速度结论；本轮完整串行结果见开头。

同日，原 R 全染色体参考运行完成：18 个正式关联文件齐全，所有子进程退出码为 0。两个 worker、每个 BLAS 线程数为 1，原关联调度的并行 wall 为 `24170.53543522954` 秒，各关联函数累计 `46101.251` 秒。并行 wall 不包含原始表型准备和零模型拟合；函数累计时间包含并行重叠，二者分别报告。

历史冻结源码 `56162751f3d9cf29ff666bf43b75503b7981b6a89451b6ab666a600b3d0a7b8a` 的五个完整 Coding 文件严格通过；首个 Noncoding 文件的 20,358 个数值单元格中有两个超出 `abs=1e-10 + rel=1e-7`，结构差异为零，最大绝对差 `8.28e-8`、相对差 `2.088e-7`。该轮状态为 interrupted / `strict_failed`，完整计时未完成，产物独立保留。这里不报告完整加速结论。

更早的一处差异已在该真实基因的 528 个数值元素局部回读中通过，但局部通过不代替本轮全量结果。频率、内积和小谱求值的规则及局部证据见 [统计说明](statistics.md)。

8520cf24 冻结版本的历史回归记录：CPU 测试 142 项通过、41 项跳过（40 项需 CUDA、1 项需本地 SDK）；CUDA 专项在 RTX 3060 上 81 项通过，在共享 A100 上 83 项通过。15 个真实集合分别对照原 R，串行和批量各 984 个字段严格通过。单元测试和局部集合保留各自范围；这一历史源码的完整 795 项验收未完成，当前完整串行结果见开头。

## 全部 15 个基因 mask

共同条件是当前染色体、QC 为 `PASS`、指定的 variant type、最终模型样本中的稀有 MAF。基因分析默认 `SNV`、`mean` 缺失填补、MAF < 0.01、至少 2 个稀有变异。单变异默认 `variant`、MAC ≥ 20；单变异自身不使用这些 mask。

| 入口 | mask | 原选择条件 |
|---|---|---|
| `Gene_Centric_Coding` | `plof` | `stopgain`、`stoploss` 或 splicing 类别，包括两个 ncRNA splicing 类别 |
| 同上 | `plof_ds` | `plof` 加 MetaSVM 为 D 的 missense |
| 同上 | `missense` | EXONIC.Category 为 `nonsynonymous SNV`；输出另外汇总 disruptive 检验 |
| 同上 | `disruptive_missense` | missense 且 MetaSVM 为 D |
| 同上 | `synonymous` | EXONIC.Category 为 `synonymous SNV` |
| 同上 | `ptv` | SNV 模式：`stopgain`、`stoploss`、`splicing`、`exonic;splicing`；Indel 模式：frameshift deletion/insertion；variant 模式取二者并集 |
| 同上 | `ptv_ds` | `ptv` 加 disruptive missense |
| `Gene_Centric_Noncoding` | `upstream` | Category 为 upstream，Info 的任意逗号 token 匹配基因 |
| 同上 | `downstream` | Category 为 downstream，Info 的任意逗号 token 匹配基因 |
| 同上 | `UTR` | UTR3、UTR5、UTR5;UTR3；Info 的首个左括号前 token 匹配基因 |
| 同上 | `promoter_CAGE` | 原 TxDb promoter 区间并集内、有 CAGE 信号；Info 按左括号/逗号/分号/连字符拆分的首 token 匹配基因 |
| 同上 | `promoter_DHS` | 同 promoter_CAGE，使用 DHS 信号 |
| 同上 | `enhancer_CAGE` | GeneHancer 和 CAGE 非空；GeneHancer 第 4 个等号 token 的首分号 token 匹配基因 |
| 同上 | `enhancer_DHS` | 同 enhancer_CAGE，使用 DHS 信号 |
| 独立 `ncRNA` | `ncRNA` | ncRNA_exonic、ncRNA_exonic;splicing、ncRNA_splicing；Info 首分号段移除括号内容后，前三个逗号 token 之一匹配基因 |

**原 R 真实覆盖基线。** chr21、一个连续表型的正式参考覆盖 42,652 个模型样本；基因分析采用上述 SNV/mean 默认组合。18 份原 R 批次 coverage 汇总记录了 3,443 个 gene–mask 槽位，其中 1,849 个非空、1,594 个为 `NULL`，全部 15 种 mask 都有真实非空结果。目录身份、mask 库存及顺序、空对象类型的检查均无差异；`NULL` 保留为正式输出的一部分。

| 类别 | mask | 非空 | `NULL` | 槽位 |
|---|---|---:|---:|---:|
| Coding | `plof` | 115 | 106 | 221 |
| Coding | `plof_ds` | 128 | 93 | 221 |
| Coding | `missense` | 156 | 65 | 221 |
| Coding | `disruptive_missense` | 89 | 132 | 221 |
| Coding | `synonymous` | 154 | 67 | 221 |
| Coding | `ptv` | 112 | 109 | 221 |
| Coding | `ptv_ds` | 128 | 93 | 221 |
| Noncoding | `upstream` | 111 | 110 | 221 |
| Noncoding | `downstream` | 107 | 114 | 221 |
| Noncoding | `UTR` | 151 | 70 | 221 |
| Noncoding | `promoter_CAGE` | 95 | 126 | 221 |
| Noncoding | `promoter_DHS` | 107 | 114 | 221 |
| Noncoding | `enhancer_CAGE` | 123 | 98 | 221 |
| Noncoding | `enhancer_DHS` | 132 | 89 | 221 |
| ncRNA | `ncRNA` | 141 | 208 | 349 |

非空 gene 返回对象为 mixed list matrix：missense 为 `1×97`，其余为 `1×91`。四个 Single 正式文件分别有 45,829、104,342、95,219 和 72,742 行，共 318,132 行，MAC cutoff 为 20。原 R coverage 的 timing 哈希聚合为 `a69f730e7413db978c171385a40e358293fd4cc8b66e328633f0dd0e6f297415`，18 个正式文件的哈希聚合为 `5fbdacbec0d14f6b051dbe15cffd6e3139b1e6df6a612c6cfb71c9b1e9c8b463`。本轮串行 GPU 的 18 个正式文件已逐一通过这一原 R 基线的结构和数值比较，并完成上文的八份 Single 元数据读回及零模型检查。该结论对应 chr21、一个连续表型、基因 SNV/mean 与单变异 variant/MAC 20；其他染色体、Indel 基因分析、其他参数组合、多表型或二分类表型仍需各自验证。

Coding 使用原包 `genes_info` 的闭区间坐标。非编码和 ncRNA 从整条染色体的注释分派基因；远端 enhancer 不能按 coding 基因跨度裁掉。Promoter 使用同一个 TxDb 版本导出的真实区间并集，不能用基因上下游长度代替。重复位置按 GDS variant 身份保留，索引按 GDS 原顺序排列。

`all_categories` 只包含前五个 coding mask；全部七个必须使用 `all_categories_incl_ptv`。Noncoding 的 `all_categories` 包含全部七项；`ncRNA` 是独立函数。Long_Masks 教程改变任务调度和大集合处理，未增加新的生物学 mask。

每个 gene 入口还支持 `variant_type="Indel"` 和 `"variant"`、`geno_missing_imputation="minor"`、关闭注释权重、选择单个类别、集合大小上下限。本轮完整染色体基准使用上面的 SNV/mean 默认组合；其他参数组合需要对应的独立验证。

## Base 与 PheWAS 的频率和缺失填补

实际安装的原 R namespace 已核对：base 为上述 STAARpipeline 0.9.9，PheWAS 为 [STAARpipelinePheWAS 0.9.7.1](https://github.com/yuxinyuanqt/STAARpipelinePheWAS/tree/6b72cf9d1f5ef001887b37d1d77a0c5370c46be0)。它们共用基因型提取 helper，但后续计算 MAF 的顺序不同。`frequency_mode` 的两种语义据此区分为 `reference` 和 `count`；本轮完整串行验收对应 base 的 `reference` 路径，PheWAS 的多表型完整覆盖按其原 wrapper 另行验证。

两条路径的方向规则相同：原 helper 在 `ALT_AF <= 0.5` 时选 ALT，因此 AF tie 也选 ALT。PheWAS 导入 base helper，随后每 trait 保留并集方向；`frequency_mode` 不改变 tie 方向。历史 35 位点解码和 1,023 行单变异 pilot 的 AF tie 计数为 0；本轮完整串行解码实际记录 1 个 REF_AF tie，正式结果已纳入完整原 R 比较。两个 pilot 的计数保持其各自读取范围。

| 原入口 | 频率语义 | MAF 和 mean 填补值 |
|---|---|---|
| Base 单表型 | `reference` | 在模型样本的非缺失 allele 中先求 REF_AF，再计算 `ALT_AF = 1 - REF_AF`、`MAF = min(REF_AF, ALT_AF)`；沿用提取结果中的 MAF，以 `2 * MAF` 填补缺失 |
| PheWAS 每个表型 | `count` | 在样本并集确定 minor 方向后，按当前表型样本提取 dosage；重新求非缺失 MAC 和 missing count，使用 `MAF = MAC / (2 * (N - missing_count))`，再以 `2 * MAF` 填补 |

在样本和 minor 方向相同、没有半缺失 genotype 时，两种 MAF 公式在实数算术下等价，浮点求值的减法与除法顺序会留下不同舍入。真实子集已观察到约 `3.31e-17` 的 MAF 差别；mean 填补会把这种差别带入基因型、score 和协方差，MAF 本身也进入 Beta 权重。

两条单变异 wrapper 的并集初始计数都先求 `ALT_AC = 2*round(N*(1-missing_rate))-REF_AC`，再取 `MAC = min(REF_AC, ALT_AC)`。Base 用此 MAC 筛选，随后不加完整 dosage MAC 的第二次 cutoff；PheWAS 每 trait 则另按完整 dosage 列和筛选。`minor` 模式把缺失填零后重新确定频率：base 先恢复 `MAC_restore = round(((2*MAF)*(1-missing_rate))*N)`，PheWAS 用完整 dosage 列和，两者各自再除以 `2*N`。初始 ALT_AC 公式与 base minor 填补的 MAC_restore 公式用途不同；奇数已知 allele 和浮点舍入可以使两者得到不同整数。

半缺失 genotype 另有实际规则差别：SeqArray 的 AF/AC 和 missing rate 按 allele 统计，但任一 allele 缺失时整次 dosage 为 NA。Base 初始公式的 missing rate 是缺失 allele / `2*N`；PheWAS 每 trait 的 missing count 是缺失整次调用的样本数。6 样本、5 位点的原 SeqArray 1.48.0 合成规则单测确认了这一区别；它不证明真实 pilot 或全染色体含有半缺失调用。详见 [读取器的统计粒度](gds.md#半缺失-genotype-的两种统计粒度)。

因此，即使只有一个表型，也需按所复现的 R 入口选择频率路径；不能仅移除 PheWAS 的 trait 输出层就把它视为 base。具体源码见 base [Genotype_sp_extraction.R](https://github.com/yuxinyuanqt/STAARpipeline/blob/fbce778bf14cc4f9e892989a194c64bae2670311/R/Genotype_sp_extraction.R)、[coding_incl_ptv.R](https://github.com/yuxinyuanqt/STAARpipeline/blob/fbce778bf14cc4f9e892989a194c64bae2670311/R/coding_incl_ptv.R) 和 PheWAS [coding_PheWAS.R](https://github.com/yuxinyuanqt/STAARpipelinePheWAS/blob/6b72cf9d1f5ef001887b37d1d77a0c5370c46be0/R/coding_PheWAS.R)。

## 注释权重

教程顺序为 `CADD, LINSIGHT, FATHMM.XF, aPC.EpigeneticActive, aPC.EpigeneticRepressed, aPC.EpigeneticTranscription, aPC.Conservation, aPC.LocalDiversity, aPC.Mappability, aPC.TF, aPC.Protein`。LocalDiversity 后紧接其互补 PHRED 列 `aPC.LocalDiversity(-)`，因此产生 12 个注释列。CADD 的 NA 变为 0，其他列保留 NA。原函数跳过 catalog 中不存在的请求名；完整基准先确认 11 个请求名全部存在。

`Annotation_name_catalog` 将语义名称映射到 aGDS 字段，`Annotation_dir` 提供字段前缀。它同时需要 GENCODE.Category、GENCODE.Info、GENCODE.EXONIC.Category、MetaSVM、CAGE、DHS、GeneHancer。`Use_annotation_weights=TRUE` 且 variant type 为 SNV 时读取 PHRED 权重；其他 variant type 不加这些权重。每个 annotation 的权重与无注释权重分别进入 Beta(1,25)/Beta(1,1) 的 SKAT、Burden、ACAT-V，随后生成 STAAR-S/B/A、ACAT-O、STAAR-O。Missense 额外纳入 disruptive 的六个基础检验。

## 完整目录、批次与正式文件

基因目录直接从锁定原 R namespace 导出，不根据有效结果反推。`genes_info` 为 18,445×4 data.frame，列为 hgnc_symbol、chromosome_name、start_position、end_position。`ncRNA_gene` 为 21,104×2 data.frame，列为 chr、ncRNA。它们的原行序号进入 `catalog_index`，空 mask 也保留在调度与 coverage 中。

使用 [目录导出 oracle](../validation/oracles/export_base_manifest.R) 导出完整 manifest：

```bash
# 只在锁定的原 R 验证环境运行；输入和导出目录均为私有路径。
Rscript validation/oracles/export_base_manifest.R \
  21 private/catalog 50 100 private/jobs_num.Rdata private/chr21.gds
```

`root_manifest.json` 包含 chromosome、genes_info 列表、ncRNA_genes 列表、PASS start_loc/end_loc、四类 array_offsets、每批基因数和注释名。Coding/noncoding 每 50 个 gene，ncRNA 每 100 个 gene，Individual 每 10 Mb。全基因组前序批次数决定原 arrayid，不在单染色体重新从 1 编号。单变异边界来自全部 PASS 位点，与 gene 的 SNV 过滤独立。

| 类别 | saved object | 文件名 | base 对象结构 |
|---|---|---|---|
| Null | `obj_nullmodel` | `obj_nullmodel.Rdata` | 原 20-field glmmkin 对象，含稀疏 Matrix 与 call |
| Coding | `results_coding` | `<prefix>_Coding_<arrayid>.Rdata` | 每 gene 七个类别组成的 named list；按教程 append，保留重复类别名和 NULL |
| Noncoding | `results_noncoding` | `<prefix>_Noncoding_<arrayid>.Rdata` | 每 gene 七个类别组成的 named list；按教程 append |
| ncRNA | `results_ncRNA` | `<prefix>_ncRNA_<arrayid>.Rdata` | 原 mixed list matrix，按教程 rbind；全部为空时 NULL |
| Individual | `results_individual_analysis` | `<prefix>_Individual_Analysis_<arrayid>.Rdata` | data.frame，保留 factor levels、整数 N、原 row.names；无有效结果时 NULL |

base 没有 PheWAS 的外层 trait list。Gene 输出是 mixed list matrix，逐 cell 保留字符/整数/double 类型；原教程的 `which.max` 染色体参数为整数，部分非编码类别先形成字符向量。生产写出使用纯 Python 的 `layout="base"`，不启动 R。

## 原版调度与计时口径

[base_chromosome_reference.R](../validation/oracles/base_chromosome_reference.R) 调用 untouched 原函数并保存每 gene RDS、正式批次 Rdata 和逐 mask coverage。[run_base_reference.py](../validation/oracles/run_base_reference.py) 调度完整任务清单，默认两个 CPU worker，每个 BLAS 线程数为 1。它记录各进程 exit code、并行 wall、各脚本 wall、函数耗时之和；原函数捕获的空集合错误会保留在日志和 NULL 输出中。验收按退出状态、完整 coverage 和结果比较判断。

原关联调度从读取冻结 null 开始，它的并行 wall 不包含 raw 表型准备和 null 拟合。Raw→RINT/GRM→原 fit 另作独立步骤验证和计时；不能把两个测量相加称为连续原 CLI wall。完整 GPU driver 的 wall 则包含索引、读取、拟合、统计与正式写出。

推荐 0.9.9 与 [canonical 0.9.8.2](https://github.com/li-lab-genetics/STAARpipeline/tree/ed3e26f7fb4c5d70a765840d089e5863896ec081) 的外层 Gene_Centric_Coding/Noncoding wrapper 在忽略注释后相同，但内部实现存在实际差异：前者使用稀疏 extraction、显式频率筛选、空候选返回和 missing replacement，后者主要使用密集 dosage/flip；单变异 MAC 与 5000 分组的操作次序也不同。本次精度结论绑定推荐版本，不据此声称两个 R 版本输出相同。

新索引 parser 已用 198,265 条真实 GDS 注释中的 84,430 次有效分派和原 R 文法逐条对照，六种 parser 均无差异。base serializer 已对真实原七 coding 类别及重复名字 append 批次作原 R 回读：类型、属性、空项、顺序全部一致。全染色体的统计精度与耗时以正式覆盖验收报告为准。

参考：[STAARpipeline](https://github.com/li-lab-genetics/STAARpipeline)、[STAAR](https://github.com/li-lab-genetics/STAAR)、[教程](https://github.com/li-lab-genetics/STAARpipeline-Tutorial)、[Nature Methods 2022](https://doi.org/10.1038/s41592-022-01640-x)、[Nature Genetics 2020](https://doi.org/10.1038/s41588-020-0676-4)。
