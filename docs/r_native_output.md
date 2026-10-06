# 原生 R 输出

`staar_phewas.r_output` 将已经计算好的关联结果写成 R 可以直接读取的 `.Rdata` 或 `.rds`。生产写文件由 Python 完成，使用 [rdata](https://github.com/vnmabus/rdata) 的转换与 XDR 序列化接口；不启动 R。R 只用于开发时的原版对照和回读验收。当前连续单表型生产 CLI 使用 CUDA 串行强制 `tf32`；输出结构不随矩阵乘法模式改变。低级 API 保留 FP64 控制，CLI 控制需显式设置 `matmul_mode="fp64"`、JSON 布尔值 `precision_control=true`。当前 TF32 验收状态见 [TF32 benchmark](tf32_benchmark.md)。

## 文件及对象结构

教程的输出前缀和分批编号由运行配置确定。默认 `.Rdata` 中的对象名如下。

| 功能 | 保存对象名 | PheWAS 对象结构 |
| --- | --- | --- |
| Gaussian 零模型 | `obj_nullmodel` | `glmmkin` 命名列表，包含原类型的向量、矩阵和稀疏 Matrix 对象 |
| Coding | `results_coding` | 全类别时为类别命名列表，内层按表型排列；有效结果为混合类型 matrix，空结果为 `NULL` |
| Noncoding、ncRNA | `results_noncoding` | Noncoding 全类别使用类别命名列表；ncRNA 按表型排列；内层为混合类型 matrix 或 `NULL` |
| 单点 | `results_individual_analysis` | 表型列表，内层为 `data.frame` 或 `NULL` |

Coding 的染色体单元格为 double，`#SNV` 为 integer。Noncoding 的 upstream、downstream、UTR 以及 ncRNA 的前四列为 character，包括染色体和 `#SNV`；这来自原 R 函数构造元数据向量时的类型转换。Promoter、enhancer 的染色体仍为 double、`#SNV` 为 integer。其余计数与统计量为 double。

单点结果的 `CHR`、`POS` 是 double，`N` 是 integer，`REF`、`ALT` 是 factor。因子水平按原函数分组计算和拼接的顺序保留；最终按位置排序后仍保留原 integer `row.names`。Python 中这些信息由 `TraitRows` 保存，单纯将结果转为 JSON 再读回来不能保留这些属性。

矩阵的 `dim`、`dimnames`、混合单元格类型、命名列表顺序及空 `NULL` 都保留。普通矩阵行名是 `results_temp`；有效 disruptive missense 合并后的 missense 行名为 `results_m`，全类别结果中对应 disruptive 矩阵的行名为 `NULL`。

单个连续表型的 STAARpipeline 基础教程使用 `layout="base"`。Coding/Noncoding 为类别命名列表，各元素直接是 matrix 或 `NULL`；ncRNA 直接是 matrix 或 `NULL`，保存名为 `results_ncRNA`；单变异直接是 data.frame 或 `NULL`。教程由 `which.max` 得到整数染色体编号，因此 base coding、promoter 和 enhancer 的 `Chr` 单元格为 integer，单变异的 `CHR` 仍从 GDS 读取为 double。混合 matrix 的 `typeof` 为 list，统计单元格保留 double，不把数值列转为 character。完整文件计划见 [chromosome](chromosome.md)。

## Python 调用

```python
from staar_phewas.r_output import write_association_output

# result 是 pipeline.coding(...) 返回的类别→表型→记录结构。
write_association_output(
    "output/Brain_Coding_1.Rdata",  # 文件名决定 Rdata/RDS 格式
    result,
    kind="coding",
)

# RDS 直接保存相同的 R 对象，不使用保存变量名。
write_association_output("output/Brain_Coding_1.rds", result, kind="coding")
```

`write_association_output(path, result, kind, object_name=None, layout="phewas")` 的参数如下。

- `path`：输出路径，后缀可为 `.Rdata`、`.rda` 或 `.rds`。父目录不存在时会创建。
- `result`：pipeline 返回的有序记录及原类型元数据。空表型条目保留为 R `NULL`。
- `kind`：`coding`、`noncoding`、`ncrna`、`individual` 或 `singlevariant`。
- `object_name`：`.Rdata` 内的变量名；默认使用上表中的教程变量名。
- `layout`：`phewas` 保持 PheWAS 的原表型列表层次；`base` 要求单个表型并写基础教程的直接对象结构。

多个基因共用一个教程输出文件时，使用 `write_association_batch(path, results, kind, object_name=None, layout="phewas")`。`results` 是按任务顺序排列的 pipeline 结果序列；其余参数与单任务相同。Coding、Noncoding 按 R `append` 追加，重复类别名全部保留。每个单点任务单独输出，以保留其完整分组、factor 和行号信息。单任务批调用与 `write_association_output` 的对象结构相同。

`base` 的 ncRNA 批次按原 `rbind` 合并矩阵并忽略空 `NULL`，不会保留 PheWAS 的外层表型列表。Coding/Noncoding 原 `append` 产生的重复类别名完整保留；结果中不增加基因名外层。CLI 在当前原批次的最后一个任务完成后立即写文件并释放批次记录。

零模型及低层转换使用 `write_r_object(path, value, object_name=None, compression=True)`。默认写 gzip level 1 压缩的 XDR v3 文件；压缩级别只改变文件字节和写出成本，不改变 R 对象结构或数值。`.Rdata` 必须提供变量名；`.rds` 直接保存 `value`。`None` 是 R `NULL`，空列表是长度为零的 R list，两者有不同含义。

| Python 承载类型 | 参数及输出 |
| --- | --- |
| `RAttributed` | `value` 与有序 `attributes`；合并原属性，`object_flag` 控制 R 对象标志 |
| `RMatrix` | 二维 `values`、`columns`、`row_names`；`mode` 为 `double`、`integer`、`character` 或 `list` |
| `RDataFrame` | 有序 `columns` 映射和 `row_names`，写出原 data.frame 属性 |
| `RFactor` | 字符串 `values` 与有序且不重复的 `levels`；写 integer 编码和 factor 属性 |
| `RS4` | `class_name`、有序 `slots`、`package`；默认包为 `Matrix` |
| `RSymbol` | `name`，写出 R 符号 |
| `RCall` | 函数 `name`、有序 `(参数名或 None, 值)` 序列，写出 R language call |

`sparse_matrix(matrix, class_name="dgCMatrix", uplo="U", dimnames=None)` 将 SciPy 稀疏矩阵写为 Matrix 的 CSC slots。`dsCMatrix` 保留所选上三角或下三角；不展开为样本数平方的 dense 矩阵。`dense_s4_matrix(matrix, dimnames=None)` 写出 `dgeMatrix`，用于零模型的有限列精度设计矩阵。

## R 读取及原版写法

```r
# PheWAS：类别内还有表型列表层。
load("output/Brain_Coding_1.Rdata")
results_coding$missense[[1]]

results_from_rds <- readRDS("output/Brain_Coding_1.rds")
stopifnot(identical(results_coding, results_from_rds))

# 单表型 base（写出时 layout="base"）：类别内直接是 matrix 或 NULL。
load("output/Brain_Coding_base_1.Rdata")
results_coding$missense
```

原教程的对应写法是 `save(results_coding, file=...)`、`save(results_noncoding, file=...)`、`save(results_individual_analysis, file=...)` 和 `save(obj_nullmodel, file="obj_nullmodel.Rdata")`。关联文件保留配置的前缀与数组编号，例如 `Brain_Coding_1.Rdata`。

## 本版原生 TF32 / FP32 完整文件验收

本版 F 在真实 chr21 的全部 795 项任务中生成 18 份关联文件与一份零模型，共 19 份原生文件，结构检查全部通过。15 份包含 P 的文件共 478,082 个有效、可比较 P；三个合法空关联文件保留原 NULL，仍通过结构及原严格空结果检查。原 R 或本版任一 `P<0.05` 的联合范围包含 24,713 个 P，logP 误差无超限，最大为 `0.0003579714`。全部 P 的差异继续保存作诊断，关联非 P 浮点数不要求逼近 FP64。

零模型另按原严格数值容差验收。本版从已经验证样本 ID、顺序及输入字段的 cache 载入模型；写出 `residuals` 时按原 NPZ 的 dtype 计算 `phenotype-fitted_values`，保留原字段名、class 和属性。模型 phenotype、fitted、scaled residual、Score、协方差及完整谱的关联计算仍为 TF32/FP32，序列化不重新计算 FP64 稠密矩阵，也不读取原 R 的数值作为输入。先前 B/C2/E 的 332 个 residuals 超容差保持为旧失败记录，本版修复后严格零模型通过。

完整缓存流程进程墙钟 `429.097 s`，首次转存与新拟合零模型另计，300 秒目标尚未达到。详见 [本版 benchmark](tf32_benchmark.md)与[验收规则](../validation/README_logp.md)。

## v0.1 区域样例的已完成回读验证

以下窗口结果属于 v0.1 历史回读记录；当前 pipeline 已移除固定窗口和滑动窗口入口。

v0.1 的区域样例 CLI 从原始表型准备输入并完成真实 GPU 计算，包含三个 Coding 基因、一个单滑窗、Noncoding 全部七类、ncRNA、五个半重叠滑窗、18 个单点结果，以及单独选择的 missense、disruptive missense 和 UTR，共 11 项正式 `.Rdata`。这些是指定区域及基因的 PheWAS 样例。原版 R 3.6.1 与 Matrix 1.2-17 递归读取后，`typeof`、全部属性、列名及顺序、列表层次、空条目、factor 水平和 `row.names` 均为零结构差异。全部数值满足 `abs(error) <= 1e-10 + 1e-7 * abs(reference)`；最大绝对差为 `1.12218e-8`，最大相对差为 `3.59737e-8`。

可选 `.rds` 已另行对八种真实结果对象回读，均与对应 `.Rdata` 命名对象 `identical`。两个基因的 Coding 批列表和两个滑窗批矩阵也与原 R 的 `append`、`rbind` 结构一致。

从原始表型重新准备输入并拟合的 42,652 样本 Gaussian 零模型，完成原版 20 字段回读，包括 `call`、`names`、`assign`、稀疏 S4 slots 及矩阵维度名。结构差异为零，全部数值字段最大绝对差为 `1.78e-14`。原版 `STAAR_sp` 可以直接使用 Python 正式输出的零模型；真实三变异集合的统计结果最大绝对差为 `1.92e-15`，score 最大差为 `5.51e-14`，协方差差为零。

SPA 文件也用原版完整 Coding 函数回读对照。在 288,554 样本的原 R 拟合状态验证模式下，两个基因分别包含 11 和 19 个稀有变异；正式 GPU CLI `.Rdata` 的结构均匹配原版，同一 `SPA_p_filter=TRUE` 分支的全部统计最大绝对差分别为 `1.34e-15`、`4.45e-16`。这一项验证原拟合状态输入后的关联与写出，不代表原生二元混合零模型拟合或其正式文件导出已经完成。

独立比较器同时区分 R 的 `NA_real_` 与 `NaN`，并检查 `Inf` 的符号。固定特殊值的原 R 读回控制已确认：`NA`/`NaN` 互换及正负无穷互换均拒绝；相同 `NA`、相同 `NaN` 通过。冻结版本的 Python writer 将 `np.nan` 保存为原生 R `NaN`，没有改写为 `NA`。这些是文件规则控制，不作为真实数据 benchmark。

这一检查验证文件及对象兼容性。统计数值的原版对照由完整 pipeline 验收记录独立报告；文件字节无需相同，数值误差仍按统计验收标准检查。

## 0.2.0 FP64 基线的 chr21 串行完整文件验收

2026-10-05，冻结执行源码 `c6362c6d…` 在一个连续连续表型、42,652 个模型样本的 chr21 上完成 GPU 串行验收。Base STAARpipeline 原目录的 221 个 coding gene、221 个 noncoding gene、349 个 ncRNA 及四个单变异区段全部运行，共 795 项任务、15 个基因 mask。18 份正式关联 `.Rdata` 的 saved object、对象结构、NULL、factor、row.names、批次顺序及数值全部通过原 R 读回。

18 个 `serial_vs_R` 比较共检查 161,839 个数值字段、3,343,119 个数值单元格；结构差异与超容差单元格为零，最大绝对差 `1.0913936421275139e-10`、最大相对差 `1.7739502638151633e-9`，均满足 `1e-10 + 1e-7 * abs(R)` 的组合容差。四个 Single 文件共 318,132 行、13 列，其中 SNV 238,947 行、Indel 79,185 行，N 均为 42,652；原 R 和 GPU 各四次原生元数据读回一致。新拟合并写出的 `obj_nullmodel.Rdata` 在 12 个数值字段、341,221 个单元格上与原 R 的最大差为 0。最终验收为 **18 项关联比较、8 次 Single 元数据检查和一次零模型比较全部通过**，详见 [完整原版对照记录](base_reference_inventory.md) 与 [公开聚合结果](../benchmarks/staar_chr21_serial_2026-10-05.json)。

历史 `56162751…`、`8520cf24…` 和 `5b0dfd83…` 轮次因敏感尾概率超容差而中断，各轮文件独立保留。最新版本固定原 Armadillo 的 `66×阶数` LAPACK 工作区后完成上述完整串行对照；具体定位与版本记录见 [全量记录](base_reference_inventory.md)。

另一个独立真实 100 kb 单变异 pilot 已完成原 R/GPU 正式文件对照：1,023 行、13 列、10,230 个 double 单元格，结构及超出严格容差的数值差均为零。最大绝对差 `2.956e-12`、最大相对差 `3.022e-13`；REF/ALT 的 factor 水平和编码、整数 row.names、列顺序及全部属性一致，RDS 与各自的 Rdata 对象 `identical`。最终官方 SDK `native_auto` 读取器，以及进一步加入 CUDA 整数解码和 MAC 预筛的版本，分别重新生成正式文件并通过同一整套原 R 回读。该区段包含 324 个 common、699 个 rare 位点，实际 REFminor 与 AF tie 均为零；新解码覆盖的半缺失和全部 allele 缺失也为零。这些方向和边界情况尚未由该真实 pilot 覆盖。此项只记录对应候选版本的局部正确性，不纳入完整染色体 benchmark；原 100 kb 调用未单独记录函数耗时，不能用多次选区试探的总耗时作 R 对照。

## 原实现与许可

- [STAARpipelinePheWAS](https://github.com/li-lab-genetics/STAARpipelinePheWAS)：PheWAS 函数、结果列表和列构造；GPL v3。
- [STAARpipeline Tutorial](https://github.com/li-lab-genetics/STAARpipeline-Tutorial)：分步输出变量及文件约定。
- [rdata](https://github.com/vnmabus/rdata)：Python R 序列化依赖；MIT。
- [R 序列化说明](https://cran.r-project.org/doc/manuals/r-release/R-ints.html#Serialization-Formats)：XDR 和对象属性格式。
