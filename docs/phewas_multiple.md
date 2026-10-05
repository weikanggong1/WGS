# 多个独立表型的 PheWAS

`PheWASPipeline` 支持多个独立 Gaussian 零模型。每个表型保留自己的完整案例、协变量和 GRM 拟合；读取所有模型的样本并集后，在每个模型样本中重新计算频率和插补值。这与一个模型同时估计相关表型协方差的 [联合 MultiSTAAR](multi.md) 有不同统计含义。

```mermaid
flowchart LR
    A[各表型完整案例与 GRM] --> B[分别拟合 Gaussian null]
    B --> C[以实际 GDS ID 构造样本并集]
    C --> D[共享分块基因型与全局 minor 方向]
    D --> E[各表型重新计算 MAF 与缺失插补]
    E --> F[各自 STAAR 与单变异统计]
    F --> G[按表型顺序保存原生 Rdata]
```

每个输入 NPZ 至少包含 `y_raw[n]`、`ids[n]`，可包含 `covariates[n,p]`、`grm_diagonal[n]` 和零基 `sample_indices[n]`。不同模型允许不同 `n`，个体顺序须与设计矩阵和 GRM 一致。共同个体只在 GDS 并集中读取一次。跨染色体使用实际 GDS ID 重新匹配，不能把上一条染色体的物理行号直接套用。

```python
import json
from pathlib import Path
from staar_phewas.gds import SeqArrayGDS
from staar_phewas.io import fit_prepared_input
from staar_phewas.pipeline import PheWASPipeline

# 所有个体数据和产物存放在私有目录。
annotation_catalog = json.loads(Path("annotation_catalog.json").read_text())
gene_start, gene_end = 200000, 230000  # 对照版本的 GENE_B 一基 inclusive 坐标
model_one, rows_one = fit_prepared_input("trait_one.npz", device="cuda", transform="rint")
model_two, rows_two = fit_prepared_input("trait_two.npz", device="cuda", transform="rint")
with SeqArrayGDS("chromosome.gds") as gds:
    pipeline = PheWASPipeline(
        gds, [model_one, model_two],
        gds_sample_indices=[rows_one, rows_two],
        qc_path="annotation/info/QC_label",
        annotation_catalog=annotation_catalog,  # 功能注释名 -> 原 GDS 节点
        annotation_names=["CADD", "aPC.LocalDiversity"],
    )
    results = pipeline.coding(
        chromosome="22", gene_name="GENE_B", start=gene_start, end=gene_end,
        category="all_categories",
    )
```

`results` 的各类别包含按 `[model_one, model_two]` 排列的结果；某个模型未达到稀有变异数下限时保留原版空项。所有模型共用 union 的 minor allele 方向，随后按各自完整案例计算 MAF；不会为每个模型再次翻转等位基因。参数、各分析输入和正式输出详见 [pipeline](pipeline.md)。

命令行是 `staar-phewas-torch analysis.json --device cuda`。`phenotypes` 数组中每一项对应一个独立模型：

```json
{
  "phenotypes": [
    {"name": "trait_one", "input": "trait_one.npz", "transform": "rint", "save_model": "null_one.npz"},
    {"name": "trait_two", "input": "trait_two.npz", "transform": "rint", "save_model": "null_two.npz"}
  ],
  "qc_path": "annotation/info/QC_label",
  "annotation_catalog": "annotation_catalog.json",
  "annotation_names": ["CADD", "aPC.LocalDiversity"],
  "chromosomes": [{"name": "22", "gds": "chromosome.gds", "jobs": [{
    "kind": "coding", "arguments": {"gene_name": "GENE_B", "start": 200000, "end": 230000},
    "output": "coding_results.Rdata"
  }]}]
}
```

无需 `joint_mode`；多个独立模型的 `n_pheno` 各为 1。每个 model cache 保存自己的样本顺序。正式关联输出使用 Rdata/RDS；多个 genomic job 指向同一输出文件时，coding/noncoding 按原版 append 保留重复类别名，sliding 按每个表型 rbind。`debug_output` 仅在 `debug_json=true` 时额外保存同一次计算的私有 JSON。

原版 R 对应调用为 `Gene_Centric_Coding_PheWAS(..., obj_nullmodel_list=list(null_one, null_two))`、`Sliding_Window_Single_PheWAS` 和 `Individual_Analysis_PheWAS`。原模型分别由 `STAARpipeline::fit_nullmodel(y~1, data=..., kins=..., id="id")` 拟合。

2026-10-04 用两个真实连续连续表型分别保留 42,652、42,418 名完整案例，均保留真实 GRM。第一个表型从 raw 输入采用与 R 逐元素相同的 RINT；第二个使用明确冻结的变换值，R 和 GPU 读取相同值，不将这一项称为新的 raw-transform 验收。原作者推荐环境中真正 STAARpipelinePheWAS wrapper 运行两模型列表，比较 GENE_B 全部 coding 类别、GENE_A 单窗口及单变异。

| 分析 | 核对数值字段 | 最大绝对差 | 原 R / GPU job 秒 |
|---|---:|---:|---:|
| GENE_B 全部 coding | 362 | 1.12e-8 | 32.811 / 39.016 |
| GENE_A 单窗口 | 74 | 4.10e-12 | 15.903 / 17.792 |
| GENE_A 单变异 | 396 | 5.41e-12 | 8.167 / 21.165 |

所有字段通过 `abs_error <= 1e-10 + 1e-7*abs(R_value)`；正式 Rdata 的对象名、class、typeof、属性顺序、dimnames、factor levels 和 row.names 与原版比较均无差异。第二个模型的 theta、精度、Sigma_iX 和固定效应 covariance 逐位一致，scaled residual 最大差 `1.39e-17`。

GPU 整体 110.081 秒，包含第一个 raw 表型 null 拟合 4.337 秒、模型加载、GDS 读取、三个 job 和 native 文件写入；观测峰值分配 0.250 GiB。原 R 表中是各 wrapper job 墙钟，不包含 null 拟合和外层文件保存，不能由这两类时间计算整体加速比。此前旧输入缓存引起的差分已换成当前冻结输入重新核验，旧缓存结果不属于该版本记录。

参考：[STAARpipelinePheWAS 原代码](https://github.com/li-lab-genetics/STAARpipelinePheWAS)、[STAARpipeline 论文](https://doi.org/10.1038/s41592-022-01640-x)、[STAAR](https://doi.org/10.1038/s41588-020-0676-4)。
