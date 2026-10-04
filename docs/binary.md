# 二分类 STAAR SPA

`staar_phewas.binary` 用 float64 PyTorch 复现原版二分类 saddlepoint approximation（SPA）。输入已经拟合并按同一样本排序的二分类零模型状态，以及 minor allele 定向、均值填补后的基因型。GPU 输入保留 GPU 计算；生产调用不调用 R。当前 API 提供个体变异 SPA 和原 `STAAR_Binary_SPA_sp` 的 burden 检验。

SPA 从残差化基因型 `G_tilde = G - projection_left @ (xw @ G)` 和 score 出发，先作 Newton 迭代；失败时按原版使用 golden-section 搜索及二分搜索，然后把正负两个单侧概率相加。该原函数返回两组 Beta burden、各组 STAAR-B 以及总 STAAR-B。

```python
import torch
from staar_phewas.binary import individual_score_test_spa, staar_binary_spa

# 所有矩阵按同一完整样本顺序排列，计算使用 float64。
genotype_matrix = genotype_matrix.to(device="cuda", dtype=torch.float64)
# genotype_matrix: [样本数, 变异数]，无缺失的 minor allele dosage。
# fitted_probability: [样本数]，二分类零模型拟合出的患病概率。
# phenotype_residual: [样本数]，普通模型为 y-mu；相关样本为 scaled.residuals。
# weighted_design_transpose: [固定效应数, 样本数]，原 XW 或稀疏模型 XSigma_i。
# projection_left_matrix: [样本数, 固定效应数]，原 XXWX_inv 或 XXSigma_iX_inv。

individual_results = individual_score_test_spa(
    genotype_matrix, phenotype_residual, fitted_probability,
    weighted_design_transpose, projection_left_matrix,
    return_diagnostics=True,
)
individual_pvalues = individual_results.pvalues  # [变异数]，保留 GPU 设备。

burden_results, burden_diagnostics = staar_binary_spa(
    genotype_matrix, minor_allele_frequency,
    phenotype_residual, fitted_probability,
    weighted_design_transpose, projection_left_matrix,
    annotations=annotation_phred,  # [变异数, 注释数]，PHRED 数值。
    names=annotation_names,
    return_diagnostics=True,
)
```

| 参数 | 含义 |
|---|---|
| `genotype` | 样本×变异 float64 dosage，已定向、已填补。函数不会 flip。 |
| `maf` | 变异的 minor allele frequency。按严格 `0<MAF<rare_maf_cutoff` 筛选。 |
| `residual` | 与原零模型分支对应的残差向量。 |
| `fitted_probability` | 长度为样本数的拟合概率，必须严格在 0 和 1 之间。 |
| `xw`、`projection_left` | 原 SPA 设计投影的两个矩阵，维度见示例；须来自同一零模型。 |
| `annotations`、`names` | PHRED 注释及唯一列名；不传时只使用基础 Beta 权重。 |
| `tol` | 默认 `2**-13`，与原 `.Machine$double.eps**0.25` 相同。 |
| `max_iter` | 默认 1000，原 Newton 与 golden-section 最大迭代数。 |
| `rare_maf_cutoff` | burden 默认 0.01。 |
| `rv_num_cutoff`、`rv_num_cutoff_max` | burden 集合最少 2 个变异，必须严格少于 10⁹ 个变异。 |
| `spa_p_filter` | burden 默认 False。设 True 时先计算原 score P 值，只对严格小于阈值的项作 SPA。 |
| `covariance` | 启用上述过滤时必须传入原始变异×变异 score 协方差，已经含拟合尺度。 |
| `normal_pvalues` | 个体变异 API 的预先计算 P 值；传入时仅对严格小于 `p_filter_cutoff` 的项重算。 |
| `p_filter_cutoff` | 默认 0.05，按原严格小于规则。 |
| `return_diagnostics` | 默认 False；True 时同时返回每项分支与失败信息。 |

`individual_score_test_spa` 默认返回变异长度的 Tensor。`staar_binary_spa` 默认返回字典：`num_variant`、`cMAC`，随后两组 `Burden(1,25)/(1,1)`，对应各注释列和 `STAAR-B(1,25)/(1,1)`，最后总 `STAAR-B`。每组包含基础权重和每个注释权重。

诊断 `SPAResult` 包含 `pvalues`、`used_bisection`、`failed`、两个单侧 Newton 的累计 `iterations`、`iteration_limit`。它们保持在输入设备。原算法遇到失败返回 P=1；本实现保留该规则并发出警告，调用者也可检查明确的失败标志。原 burden wrapper 合并时移除失败的 1 与 NA，所有剩余值之和为零则返回 1；这一边界同样保留。达到 Newton 迭代上限也单独报告。

当前实现保留冻结源码的 alternate K/K2 表达式与 max-iteration 分支求值顺序，包括其数值局限；未替换为另一个 SPA 实现。拟合零模型和准备完整样本是上层流程的责任，不能将普通零模型拟合结果冒充相关样本模型。

原 R 调用为：

```r
# 使用已经 minor allele 定向的稀疏矩阵，明确给出 MAF。
reference_burden <- STAAR::STAAR_Binary_SPA_sp(
  genotype_sp, MAF, obj_nullmodel, annotation_phred,
  SPA_p_filter=FALSE
)
reference_individual <- STAARpipeline:::Individual_Score_Test_SPA(
  G, XSigma_i, XXSigma_iX_inv, scaled.residuals, fitted.values,
  .Machine$double.eps^0.25, 1000L
)
```

本模块是低层 Python 数值 API；完整命令行入口与零模型配置遵循 pipeline 的支持范围。以下真实二分类对照使用原教程已有的 AD 零模型、288,554 个样本及真实 chr22 基因型。参考为原作者镜像的未经修改 R 包，包含无过滤 burden SPA、过滤 burden SPA、个体变异 SPA 三种调用；原 null 仅作为测试输入，用来隔离关联分析算法。

| 真实集合 | 变异数 | 无过滤 burden 最大 P 值绝对差 | 过滤 burden 最大绝对差 | 个体 SPA 最大绝对差 | R 三种调用合计秒 | GPU 三种调用合计秒 |
|---|---:|---:|---:|---:|---:|---:|
| OR11H1 pLoF | 3 | 1.87×10⁻⁹ | 6.67×10⁻¹⁶ | 9.07×10⁻¹³ | 3.901 | 1.866 |
| COMT pLoF | 5 | 3.41×10⁻¹⁴ | 7.22×10⁻¹⁶ | 2.50×10⁻¹⁵ | 167.117 | 6.436 |
| APOBEC3A pLoF | 9 | 2.31×10⁻¹³ | 7.22×10⁻¹⁶ | 3.74×10⁻¹³ | 40.781 | 4.104 |

所有输出通过 `abs=10⁻⁸ + rel=10⁻⁷` 的逐项对照标准。部分困难的个体变异达到原 Newton 1000 次上限，或按原算法失败返回 P=1；这些边界的输出也一致。表内时间包括三种关联分析调用，不包含 GDS 读取、样本对齐或拟合零模型；GPU 首组包含首次 CUDA 初始化，R 在用户空间容器中运行。因此这是 association core 验证，不能据此宣称完整二分类零模型流程已经验收，也不能概括端到端加速比。固定矩阵单元检查仅验证边界和错误规则，不作为科学 benchmark。

另以完整 coding wrapper 验证了真实二分类样本中的 OR11H1 pLoF（11 个变异）和 COMT pLoF（19 个变异）。这两个集合包含在较小样本集合中为单态、在二分类完整样本中重新进入筛选的变异。命令行生成的原生 `.Rdata` 由原 R 回读，列表、矩阵、列名及属性均一致；与未经修改的 PheWAS wrapper 使用相同的默认 `SPA_p_filter=TRUE` 比较，全部统计量的最大绝对差分别为 1.33×10⁻¹⁵ 和 4.44×10⁻¹⁶。该对照使用已有原二分类相关样本零模型的验证缓存；它验证 GDS 筛选、注释权重、关联计算和文件输出，未把原 R 拟合状态计作 PyTorch 零模型拟合。

代码按 GPL-3.0-only 提供，依据 [STAAR 0.9.9](https://github.com/yuxinyuanqt/STAAR/tree/4bbf77ba8a90894a434f5eb4473d540e172dad05) 与 [STAARpipeline 0.9.9](https://github.com/yuxinyuanqt/STAARpipeline/tree/fbce778bf14cc4f9e892989a194c64bae2670311) 的对应 R/C++ 函数。参考 Li X, Li Z, et al., *Nature Genetics* 52, 969–983 (2020), [DOI](https://doi.org/10.1038/s41588-020-0676-4)；Li Z, Li X, et al., *Nature Methods* 19, 1599–1611 (2022), [DOI](https://doi.org/10.1038/s41592-022-01640-x)；Cauchy 合并文献见统计内核文档。
