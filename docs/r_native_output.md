# 原生 R 输出

`torchstaar.r_output` 将已经计算好的关联结果写成 R 可以直接读取的 `.Rdata` 或 `.rds`。生产写文件由 Python 完成，使用 [rdata](https://github.com/vnmabus/rdata) 的转换与 XDR 序列化接口；不启动 R。R 只用于开发时的原版对照和回读验收。当前连续单表型生产 CLI 使用 CUDA 串行强制 `tf32`；输出结构不随矩阵乘法模式改变。低级 API 保留 FP64 控制，CLI 控制需显式设置 `matmul_mode="fp64"`、JSON 布尔值 `precision_control=true`。当前 TF32 验收状态见 [TF32 benchmark](torchstaar.md)。

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

单个连续表型的 STAARpipeline 基础教程使用 `layout="base"`。Coding/Noncoding 为类别命名列表，各元素直接是 matrix 或 `NULL`；ncRNA 直接是 matrix 或 `NULL`，保存名为 `results_ncRNA`；单变异直接是 data.frame 或 `NULL`。教程由 `which.max` 得到整数染色体编号，因此 base coding、promoter 和 enhancer 的 `Chr` 单元格为 integer，单变异的 `CHR` 仍从 GDS 读取为 double。混合 matrix 的 `typeof` 为 list，统计单元格保留 double，不把数值列转为 character。完整文件计划见 [chromosome](torchstaar.md)。

## Python 调用

```python
from torchstaar.r_output import write_association_output

# result 是 pipeline.coding(...) 返回的类别→表型→记录结构。
write_association_output(
    "output/Study_Coding_1.Rdata",  # 文件名决定 Rdata/RDS 格式
    result,
    kind="coding",
)

# RDS 直接保存相同的 R 对象，不使用保存变量名。
write_association_output("output/Study_Coding_1.rds", result, kind="coding")
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
load("output/Study_Coding_1.Rdata")
results_coding$missense[[1]]

results_from_rds <- readRDS("output/Study_Coding_1.rds")
stopifnot(identical(results_coding, results_from_rds))

# 单表型 base（写出时 layout="base"）：类别内直接是 matrix 或 NULL。
load("output/Study_Coding_base_1.Rdata")
results_coding$missense
```

原教程的对应写法是 `save(results_coding, file=...)`、`save(results_noncoding, file=...)`、`save(results_individual_analysis, file=...)` 和 `save(obj_nullmodel, file="obj_nullmodel.Rdata")`。关联文件保留配置的前缀与数组编号，例如 `Brain_Coding_1.Rdata`。

## 当前验证与版本

原生文件验收要求全部原结构、保存对象、NULL、factor/row.names和严格null一致；P有效性与显著联合精度分别检查。最新完整R验收、进程墙钟与近期版本记录只见 [主指南](torchstaar.md#5-最新真实精度与耗时)，不以文件写出成功代替验收通过。

当前writer使用gzip level1的原XDR v3结构，cached null residuals按原缓存dtype序列化；关联张量精度不改变。Python写出不运行R，独立回读命令见 [验证说明](../validation/README.md)。

## 原实现与文献

[rdata转换/序列化](https://github.com/vnmabus/rdata)，原STAARpipeline/PheWAS结构与方法文献见 [主指南参考](torchstaar.md#7-原实现许可与参考文献)。
