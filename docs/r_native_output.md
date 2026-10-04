# 原生 R 输出

`staar_phewas.r_output` 将已经计算好的关联结果写成 R 可以直接读取的 `.Rdata` 或 `.rds`。生产写文件由 Python 完成，使用 [rdata](https://github.com/vnmabus/rdata) 的转换与 XDR 序列化接口；不启动 R。R 只用于开发时的原版对照和回读验收。

## 文件及对象结构

教程的输出前缀和分批编号由运行配置确定。默认 `.Rdata` 中的对象名如下。

| 功能 | 保存对象名 | PheWAS 对象结构 |
| --- | --- | --- |
| Gaussian 零模型 | `obj_nullmodel` | `glmmkin` 命名列表，包含原类型的向量、矩阵和稀疏 Matrix 对象 |
| Coding | `results_coding` | 全类别时为类别命名列表，内层按表型排列；有效结果为混合类型 matrix，空结果为 `NULL` |
| Noncoding、ncRNA | `results_noncoding` | Noncoding 全类别使用类别命名列表；ncRNA 按表型排列；内层为混合类型 matrix 或 `NULL` |
| 滑窗 | `results_sliding_window` | 表型列表，内层为混合类型 matrix 或 `NULL` |
| 单点 | `results_individual_analysis` | 表型列表，内层为 `data.frame` 或 `NULL` |

Coding 的染色体单元格为 double，`#SNV` 为 integer。Noncoding 的 upstream、downstream、UTR 以及 ncRNA 的前四列为 character，包括染色体和 `#SNV`；这来自原 R 函数构造元数据向量时的类型转换。Promoter、enhancer 的染色体仍为 double、`#SNV` 为 integer。滑窗坐标为 double，`#SNV` 为 integer。其余计数与统计量为 double。

单点结果的 `CHR`、`POS` 是 double，`N` 是 integer，`REF`、`ALT` 是 factor。因子水平按原函数分组计算和拼接的顺序保留；最终按位置排序后仍保留原 integer `row.names`。Python 中这些信息由 `TraitRows` 保存，单纯将结果转为 JSON 再读回来不能保留这些属性。

矩阵的 `dim`、`dimnames`、混合单元格类型、命名列表顺序及空 `NULL` 都保留。普通矩阵行名是 `results_temp`；有效 disruptive missense 合并后的 missense 行名为 `results_m`，全类别结果中对应 disruptive 矩阵的行名为 `NULL`。

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
- `kind`：`coding`、`noncoding`、`ncrna`、`sliding`、`individual` 或 `singlevariant`。
- `object_name`：`.Rdata` 内的变量名；默认使用上表中的教程变量名。
- `layout`：当前为 `phewas`，保持 PheWAS 的原列表层次。

多个基因或窗口共用一个教程输出文件时，使用 `write_association_batch(path, results, kind, object_name=None, layout="phewas")`。`results` 是按任务顺序排列的 pipeline 结果序列；其余参数与单任务相同。Coding、Noncoding 按 R `append` 追加，重复类别名全部保留；滑窗按同一表型分别执行矩阵 `rbind`，保留行名。每个单点任务单独输出，以保留其完整分组、factor 和行号信息。单任务批调用与 `write_association_output` 的对象结构相同。

零模型及低层转换使用 `write_r_object(path, value, object_name=None, compression=True)`。默认写 gzip 压缩的 XDR v3 文件。`.Rdata` 必须提供变量名；`.rds` 直接保存 `value`。`None` 是 R `NULL`，空列表是长度为零的 R list，两者有不同含义。

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
load("output/Brain_Coding_1.Rdata")
results_coding$missense[[1]]

results_from_rds <- readRDS("output/Brain_Coding_1.rds")
stopifnot(identical(results_coding, results_from_rds))
```

原教程的对应写法是 `save(results_coding, file=...)`、`save(results_noncoding, file=...)`、`save(results_sliding_window, file=...)`、`save(results_individual_analysis, file=...)` 和 `save(obj_nullmodel, file="obj_nullmodel.Rdata")`。关联文件保留配置的前缀与数组编号，例如 `Brain_Coding_1.Rdata`。

## 已完成的回读验证

最后一次完整 CLI 从原始表型准备输入并完成真实 GPU 计算，包含三个 Coding 基因、一个单滑窗、Noncoding 全部七类、ncRNA、五个半重叠滑窗、18 个单点结果，以及单独选择的 missense、disruptive missense 和 UTR，共 11 项正式 `.Rdata`。原版 R 3.6.1 与 Matrix 1.2-17 递归读取后，`typeof`、全部属性、列名及顺序、列表层次、空条目、factor 水平和 `row.names` 均为零结构差异。全部数值满足 `abs(error) <= 1e-10 + 1e-7 * abs(reference)`；最大绝对差为 `1.12218e-8`，最大相对差为 `3.59737e-8`。

可选 `.rds` 已另行对八种真实结果对象回读，均与对应 `.Rdata` 命名对象 `identical`。两个基因的 Coding 批列表和两个滑窗批矩阵也与原 R 的 `append`、`rbind` 结构一致。

从原始表型重新准备输入并拟合的 42,652 样本 Gaussian 零模型，完成原版 20 字段回读，包括 `call`、`names`、`assign`、稀疏 S4 slots 及矩阵维度名。结构差异为零，全部数值字段最大绝对差为 `1.78e-14`。原版 `STAAR_sp` 可以直接使用 Python 正式输出的零模型；真实三变异集合的统计结果最大绝对差为 `1.92e-15`，score 最大差为 `5.51e-14`，协方差差为零。

SPA 文件也用原版完整 Coding 函数回读对照。在 288,554 样本的原 R 拟合状态验证模式下，两个基因分别包含 11 和 19 个稀有变异；正式 GPU CLI `.Rdata` 的结构均匹配原版，同一 `SPA_p_filter=TRUE` 分支的全部统计最大绝对差分别为 `1.34e-15`、`4.45e-16`。这一项验证原拟合状态输入后的关联与写出，不代表原生二元混合零模型拟合或其正式文件导出已经完成。

这一检查验证文件及对象兼容性。统计数值的原版对照由完整 pipeline 验收记录独立报告；文件字节无需相同，数值误差仍按统计验收标准检查。

## 原实现与许可

- [STAARpipelinePheWAS](https://github.com/li-lab-genetics/STAARpipelinePheWAS)：PheWAS 函数、结果列表和列构造；GPL v3。
- [STAARpipeline Tutorial](https://github.com/li-lab-genetics/STAARpipeline-Tutorial)：分步输出变量及文件约定。
- [rdata](https://github.com/vnmabus/rdata)：Python R 序列化依赖；MIT。
- [R 序列化说明](https://cran.r-project.org/doc/manuals/r-release/R-ints.html#Serialization-Formats)：XDR 和对象属性格式。
