# 表型、GDS 与亲缘矩阵对齐

`staar_phewas.prepare` 从表型表、原生 SeqArray GDS 和 R 格式的稀疏 GRM 生成私有 NPZ。工具保留表型表顺序，剔除缺失、排除名单及无法同时匹配的样本。GDS 和 Matrix 文件由 Python 读取，不启动 R。

```bash
staar-phewas-prepare \
    --gds private/chr22.gds \
    --phenotypes private/phenotypes.tsv \
    --phenotype-columns trait_A \
    --id-column IID \
    --grm private/relationship.Rdata \
    --id-pattern '([^_]+)$' \
    --exclude private/excluded_ids.txt \
    --output private/aligned_trait.npz
```

`--phenotype-columns trait_A trait_B` 生成联合完整样本输入。多个独立 PheWAS 表型分别准备各自 NPZ，保留不同的有效样本数。

## Python 调用和参数

```python
from staar_phewas.prepare import prepare_input

alignment_summary = prepare_input(
    gds="private/chr22.gds",
    phenotypes="private/phenotypes.tsv",
    phenotype_columns=["trait_A"],
    id_column="IID",
    grm="private/relationship.Rdata",
    id_pattern=r"([^_]+)$",
    exclude="private/excluded_ids.txt",
    output="private/aligned_trait.npz",
)
```

| 参数 | 格式、默认值及用途 |
| --- | --- |
| `gds` | 原生 SeqArray 文件，读取其 `sample.id`。 |
| `phenotypes` | 有表头的表型文件，ID 保持字符串。 |
| `phenotype_columns` | 一个或多个数值列名；共同缺失删除，不能重复。 |
| `id_column` | 表型 ID 列，默认 `IID`。 |
| `covariate_columns` | 可选数值协变量列名；工具在前面加入截距，同步剔除缺失。默认空。 |
| `grm` | 可选 `.Rdata`，含一个命名的 `dsCMatrix` 或对称 `dgCMatrix`。行列 ID 必须相同。省略时准备普通模型输入。 |
| `grm_object` | 工作空间有多个稀疏矩阵时指定对象名；CLI 为 `--grm-object`。 |
| `id_pattern` | 默认 None，严格匹配原 ID；可给恰好一个捕获组的正则式，显式归一化 GDS/GRM ID。不变换表型表 ID。示例提取最后一个下划线后的部分。任何归一化重复 ID 都报错。 |
| `exclude` | 可选无表头单 ID 列，或 FID/IID 空白分隔表；常见 ID 表头跳过。 |
| `exclude_id_index` | 多列表的 ID 列位置，从零开始，默认 1；单列自动使用第 0 列。 |
| `delimiter` | `whitespace`（默认）、`tab` 或 `comma`。 |
| `output` | 私有 NPZ 输出路径；应使用 `.npz` 后缀。 |

返回汇总包含表型行数、完整行数、排除 ID 数、最终样本数、性状数和保留 GRM 边数。输出 NPZ 包含 `ids`、原始 `gds_sample_ids`、零起始 `sample_indices`、`y_raw`、`phenotype_names`；GRM 输入时还包含 `grm_indices`、`grm_diagonal` 和非对角边的三个数组。单性状 `y_raw` 为 N 向量，联合为 N×T。协变量存在时另存 N×P `covariates` 与 `covariate_names`。

本工具不变换表型，不再次阈值化 GRM，不把其对角替换成 1。秩变换在拟合配置的 `transform="rint"` 阶段执行。正式零模型及关联输出见 [pipeline](pipeline.md)。

2026-10-04：真实原始连续表型 64,840 行、60,709 个有限值、353 个排除 ID，最终 42,652 个 GDS/GRM 对齐样本。Python 原生读取 GRM 后的 8 个关键数组与独立 R 导出的准备结果逐项一致，保留真实非单位对角项。此记录属于输入一致性检查；关联精度和耗时见 [benchmark](benchmark.md)。

原 R 对应通过 `load()` 读取 Matrix GRM、`seqGetData(genofile,"sample.id")` 读取样本，并按照匹配索引子集化；正式拟合调用见 [零模型](null_model.md)。参考：[SeqArray](https://github.com/zhengxwen/SeqArray)、[CoreArray pygds](https://github.com/CoreArray/pygds)、[rdata](https://github.com/vnmabus/rdata)、[R Matrix](https://cran.r-project.org/package=Matrix)。
