# PheWAS 变异分组与注释权重

`staar_phewas.masks` 按 STAARpipelinePheWAS 的注释规则生成 coding、noncoding 和 ncRNA 分组。每个分组返回输入变异中的零起始索引，顺序与 GDS 中的变异顺序一致。该步骤只确定功能分组；表型缺失、等位基因频率、剂量填补和稀有变异筛选由后续统计流程处理。

参考源码冻结为 STAARpipelinePheWAS `7a2c49617b3791c35a20504e260c038c02a4c643`，包版本 `0.9.7.1`。作者推荐容器对应的 `yuxinyuanqt/STAARpipelinePheWAS` 版本为 `6b72cf9d1f5ef001887b37d1d77a0c5370c46be0`，两者的整个 `R/` 目录逐文件一致。源码按 GPLv3 改写，原作者为 Xihao Li、Zilin Li 和 Yuxin Yuan。

## 输入与 Python 调用

`VariantAnnotations` 接收已经对齐的变异元数据：

| 参数 | 格式与含义 |
|---|---|
| `position` | 长度为变异数的一维整数数组；一开始计数的基因组坐标 |
| `qc` | 同长度字符串数组；只有 `PASS` 进入分析 |
| `annotations` | 从语义注释名到同长度数组的映射；键与原 R 注释目录中的 `name` 一致 |
| `ref`、`alt` | 参考与替代等位基因字符串；未提供 `snv` 时用于分类 |
| `chromosome` | 可选染色体数组；跨染色体筛选时必需 |
| `variant_id` | 可选原始变异编号，便于记录所选变异身份 |
| `snv` | 可选布尔数组；可直接传入 R `isSNV()` 导出的分类 |

以下示例中的数组均来自数据读取步骤。

```python
from staar_phewas.masks import VariantAnnotations, coding_masks

# 所有数组必须采用同一个变异顺序。
variant_annotations = VariantAnnotations(
    position=variant_positions,
    qc=variant_quality_labels,
    annotations=functional_annotations,
    ref=reference_alleles,
    alt=alternate_alleles,
    chromosome=variant_chromosomes,
    variant_id=variant_identifiers,
)

# 基因起止坐标必须来自与原版一致的 genes_info 表。
coding_variant_indices = coding_masks(
    variants=variant_annotations,
    gene_name=selected_gene_name,
    gene_start=selected_gene_start,
    gene_end=selected_gene_end,
    chromosome=selected_chromosome,
    variant_type="SNV",
    include_ptv=False,
)
loss_of_function_variant_ids = variant_identifiers[coding_variant_indices["plof"]]
```

`coding_masks` 的 `gene_name` 是记录中的基因名；`gene_start`、`gene_end` 为包含端点的基因范围。原版 coding 选择使用基因范围，不再根据 `GENCODE.Info` 重新分配重叠基因。`variant_type` 接受 `SNV`、`Indel` 或 `variant`；默认 `SNV`。`include_ptv=True` 另返回 `ptv` 和 `ptv_ds`。`chromosome` 可选，但读取多个染色体时应明确给出。

原版 `isSNV()` 默认要求二等位位点的 REF、ALT 都是一碱基；多等位单碱基位点也会归入 `Indel` 分支，因为该分支实际采用 `!isSNV()`。

## 分组规则与输出

| 分组 | 原版选择规则 |
|---|---|
| `plof` | `stopgain`、`stoploss`，或 `splicing`、`exonic;splicing`、`ncRNA_splicing`、`ncRNA_exonic;splicing` |
| `plof_ds` | `plof` 与 disruptive missense 的并集 |
| `missense` | `GENCODE.EXONIC.Category == "nonsynonymous SNV"` |
| `disruptive_missense` | missense 且 `MetaSVM == "D"` |
| `synonymous` | `GENCODE.EXONIC.Category == "synonymous SNV"` |
| `ptv` | SNV 分支为 stopgain/stoploss 与 coding splice；Indel 分支为 frameshift deletion/insertion；`variant` 分支取两者并集 |
| `ptv_ds` | `ptv` 与 disruptive missense 的并集 |

`noncoding_masks` 返回 `upstream`、`downstream`、`UTR`、`promoter_CAGE`、`promoter_DHS`、`enhancer_CAGE`、`enhancer_DHS`，顺序与原版 `all_categories` 一致。`include_ncrna=True` 另返回 `ncRNA`。所需注释包括 `GENCODE.Category`、`GENCODE.Info`、`GeneHancer`、`CAGE` 和 `DHS`。

- upstream/downstream 使用对应 GENCODE 类别，按 `GENCODE.Info` 的逗号分隔完整基因名匹配。
- UTR 使用 `UTR3`、`UTR5` 或 `UTR5;UTR3`，匹配 `GENCODE.Info` 第一个左括号前的整个字符串。
- promoter 先与原版 TxDb 全部基因的 promoter 区间并集相交，再要求 CAGE 或 DHS 注释非空，并按原版分隔符提取 `GENCODE.Info` 第一个基因名。
- enhancer 要求 GeneHancer 与 CAGE/DHS 同时非空；基因名取 GeneHancer 按 `=` 分隔的第四段，再取 `;` 前第一段。
- ncRNA 使用 `ncRNA_exonic`、`ncRNA_exonic;splicing`、`ncRNA_splicing`，仅检查第一个分号段去除括号内容后的前三个逗号分隔基因名。

promoter 的 `promoter_overlap` 必须是与每个变异对齐的布尔数组。也可通过 `promoter_overlaps(positions, chromosomes, promoter_intervals)` 生成；`promoter_intervals` 为 `(染色体, 起点, 终点)` 列表，采用一开始计数且包含端点的坐标。必须从同一版 TxDb 的 `promoters(genes(txdb), upstream=3000, downstream=3000)` 导出区间，不用 coding 基因起点附近的近似区间替代。默认计算所有类别时，缺少该输入会报错。可用 `requested_categories=["UTR"]` 等明确选择类别；仅请求非 promoter 类别时不需要该输入。

## 注释矩阵和权重

`annotation_phred_matrix(annotations, annotation_names, indices=None, variant_type="SNV", use_annotation_weights=True, number_variants=None)` 返回 `(PHRED矩阵, 列名列表)`。

- `annotation_names` 决定列顺序；目录中不存在的名字按原版规则跳过。
- `indices` 可限定为某个分组的索引；省略时取所有输入变异。
- `number_variants` 在注释映射为空时指定输出行数。
- CADD 的缺失分数填为零，其他缺失分数保留。
- `aPC.LocalDiversity` 后紧接一列 `aPC.LocalDiversity(-)`，值为 `-10 log10(1-10^(-PHRED/10))`。
- `variant_type` 不为 `SNV`，或 `use_annotation_weights=False` 时，返回零列矩阵，符合原版分支。

`staar_weights(maf, annotation_phred=None)` 接收已经按当前表型筛选的稀有变异频率，返回 `B`、`S`、`A` 三个矩阵。每个矩阵先排列 Beta(1,25) 的基础权重与注释权重，再排列 Beta(1,1) 对应权重。注释变换为 `r=1-10^(-PHRED/10)`；Burden 乘 `r`，SKAT 乘 `sqrt(r)`，ACAT-V 乘 `r` 并使用原版 Beta 密度比例。所有计算采用 float64。

## 多表型的数据关系

`sample_union(sample_lists)` 对多个 null model 的样本编号取保序并集。`sample_indices(samples, union_samples)` 返回单个模型样本在并集中的行位置，保持该模型的原顺序。各模型样本缺失可以不同；每次统计分析必须在自己的样本子集重新计算频率和缺失填补。

PheWAS 的 `obj_nullmodel_list` 是多个分析对象的列表。单个对象 `n.pheno == 1` 时执行逐表型 STAAR；`n.pheno > 1` 时执行利用表型相关结构的 MultiSTAAR 联合检验。多个单表型模型的列表不等于 MultiSTAAR 联合模型。

原版单个 category、single-variant 和 sliding-window 返回按模型顺序的列表；`all_categories` coding/noncoding 返回以类别为外层的命名列表，每个类别内再按模型排列结果。Coding/noncoding 表首五列为 `Gene name`、`Chr`、`Category`、`#SNV`、`cMAC`；滑窗表首五列为 `Chr`、`Start Loc`、`End Loc`、`#SNV`、`cMAC`。普通分支包含六类检验、`ACAT-O` 和 `STAAR-O`，SPA 分支采用 Burden 与 `STAAR-B`。

原版 missense 汇总还纳入 disruptive missense 的六个基础 p 值，并添加 `-Disruptive` 列。`staar_phewas.results` 实现结果组装：

```python
from staar_phewas.results import coding_record, assemble_phewas_results

# 每个 trait 的统计结果按 null model 列表顺序排列。
records_by_trait = [[] for fitted_model in fitted_null_models]
for trait_index, mask_statistics in enumerate(missense_statistics):
    if mask_statistics is not None:
        records_by_trait[trait_index].append(coding_record(
            selected_chromosome, selected_gene_name, "missense", mask_statistics,
        ))
    disruptive_statistics = disruptive_missense_statistics[trait_index]
    if disruptive_statistics is not None:
        records_by_trait[trait_index].append(coding_record(
            selected_chromosome, selected_gene_name, "disruptive_missense", disruptive_statistics,
        ))

coding_results = assemble_phewas_results(
    records_by_trait, kind="coding", category="all_categories",
)
first_trait_missense_rows = coding_results["missense"][0]
```

`coding_record(chromosome, gene_name, category, statistics)` 将统计字典的 `num_variant` 改为 `#SNV`，并放在原版五个元数据列后；剩余统计列按输入顺序保留。`window_record(chromosome, start, end, statistics)` 生成滑窗表的五个元数据列。`single_variant_record(chromosome, position, ref, alt, alt_af, maf, number_samples, statistics, number_phenotypes=1, use_spa=False)` 接受原版 score 结果；普通单表型输出 `Score`、`Score_se`、`Est`、`Est_se`，联合模型输出 `Score1` 等列，SPA 仅输出 p 值。`pvalue_log10` 为正的 `−log10(p)`。

`assemble_phewas_results` 的 `kind` 接受 `coding`、`noncoding`、`ncrna`、`singlevariant`、`sliding`；`category` 默认 `all_categories`，也可指定单个类别。`include_ptv=True` 添加两个 PTV 类别；原 R 的 `all_categories_incl_ptv` 别名也受支持。`include_ncrna=True` 可将 ncRNA 添加到 noncoding 集合。`use_spa` 决定补充列和组合规则；`cauchy_combiner` 可传入 Cauchy 组合函数，默认使用统计模块的 `cct`。结果仅含普通字典和列表，可以写为 JSON；缺少有效统计结果的 mask 保留空列表。singlevariant 按位置排序。

对于非空 disruptive mask，组装器重算 missense 的六个 STAAR 分组 p 值和 `STAAR-O`，保留 `ACAT-O`；若 disruptive mask 没有有效统计结果，附加的六列填为 1，并保留原来的汇总值。SPA 分支附加两个 Burden 值，按原 R 规则处理缺失值和等于 1 的值，并更新 `STAAR-B`。这项结果组装支持不代表 SPA 统计或 MultiSTAAR 联合 null model 已实现。

## 原版调用与验证状态

对应 R 接口为 `Gene_Centric_Coding_PheWAS(chr, gene_name, genofile, obj_nullmodel_list, category="all_categories", ...)`、`Gene_Centric_Noncoding_PheWAS(...)` 和 `ncRNA_PheWAS(...)`。Python masks 模块由 pipeline 调用，没有单独的命令行统计入口。

当前已核对官方选择规则，并通过 coding 边界、保序样本映射、重叠区间、CADD 缺失、LocalDiversity 补列及 B/S/A 权重形状的功能检查。真实 chr22 注释对照覆盖 198,265 个变异、六个 coding 基因、五个有非空结果的 ncRNA 基因，以及三种 variant_type，共 285 次原 R 选择函数调用；Python 所选变异编号及顺序全部一致，77 个非空 SNV 注释矩阵在 `rtol=atol=2e-14` 范围内一致。

promoter 区间来自原版环境的 `TxDb.Hsapiens.UCSC.hg38.knownGene 3.10.0`，共 25,750 个区间；Python 对全部 198,265 个实际变异的重叠判定与原 `GRanges` 一致，其中 18,864 个变异有重叠。此对照执行官方 R selector，数据读取替换为真实 GDS 节点导出，在基因型提取之前截断；统计流程另行验收。

结果组装的功能检查覆盖列顺序、category 外层结构、空 mask、disruptive 补充与 SPA 缺失处理。文件输出通过 `staar_phewas.r_output.write_association_output` 生成原生 `.Rdata` 或 `.rds`，保留原对象名、命名列表、混合类型 matrix、NULL、data.frame、factor 和行号。完整真实数据验收需对比 null model、score、协方差、全部统计结果以及耗时，并分别记录元数据读取、mask 选择与统计计算。

## 来源与参考文献

- [STAARpipelinePheWAS 源码](https://github.com/li-lab-genetics/STAARpipelinePheWAS/tree/7a2c49617b3791c35a20504e260c038c02a4c643)、[GPLv3 许可](https://github.com/li-lab-genetics/STAARpipelinePheWAS/blob/7a2c49617b3791c35a20504e260c038c02a4c643/LICENSE.md)。
- [SeqVarTools isSNV 实现](https://github.com/smgogarten/SeqVarTools/blob/devel/R/Methods-SeqVarGDSClass.R)。
- Li, Z., Li, X., et al. A framework for detecting noncoding rare variant associations of large-scale whole-genome sequencing studies. *Nature Methods* **19**, 1599–1611 (2022). [DOI](https://doi.org/10.1038/s41592-022-01640-x)。
- Li, X., Li, Z., et al. Dynamic incorporation of multiple in silico functional annotations empowers rare variant association analysis of large whole-genome sequencing studies at scale. *Nature Genetics* **52**, 969–983 (2020). [DOI](https://doi.org/10.1038/s41588-020-0676-4)。
