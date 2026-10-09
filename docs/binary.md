# 二分类 STAAR SPA

`torchstaar.binary` 用 float64 PyTorch 复现原版二分类 saddlepoint approximation（SPA）。输入已经拟合并按同一样本排序的二分类零模型状态，以及 minor allele 定向、均值填补后的基因型。GPU 输入保留 GPU 计算；生产调用不调用 R。当前 API 提供个体变异 SPA 和原 `STAAR_Binary_SPA_sp` 的 burden 检验。

SPA 从残差化基因型 `G_tilde = G - projection_left @ (xw @ G)` 和 score 出发，先作 Newton 迭代；失败时按原版使用 golden-section 搜索及二分搜索，然后把正负两个单侧概率相加。该原函数返回两组 Beta burden、各组 STAAR-B 以及总 STAAR-B。

```python
import torch
from torchstaar.binary import individual_score_test_spa, staar_binary_spa

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

本模块是显式 FP64 的低层 Python API，完整 TF32 染色体入口不接受二分类。二项拟合与原生状态说明见 [二项零模型](binary_null.md)。本版 Single benchmark 不覆盖 SPA 或完整二项流程；当前实测范围见 [主指南](torchstaar.md#真实验证与计时范围)。

代码按 GPL-3.0-only 提供，依据 [STAAR 0.9.9](https://github.com/yuxinyuanqt/STAAR/tree/4bbf77ba8a90894a434f5eb4473d540e172dad05) 与 [STAARpipeline 0.9.9](https://github.com/yuxinyuanqt/STAARpipeline/tree/fbce778bf14cc4f9e892989a194c64bae2670311) 的对应 R/C++ 函数。参考 Li X, Li Z, et al., *Nature Genetics* 52, 969–983 (2020), [DOI](https://doi.org/10.1038/s41588-020-0676-4)；Li Z, Li X, et al., *Nature Methods* 19, 1599–1611 (2022), [DOI](https://doi.org/10.1038/s41592-022-01640-x)；Cauchy 合并文献见统计内核文档。
