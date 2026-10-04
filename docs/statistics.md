# STAAR 数值内核

`staar_phewas.statistics` 从已经拟合好的零模型残差和基因型矩阵计算 score、协方差及 STAAR 检验。矩阵乘法、注释权重、burden、特征值分解、Cauchy 合并、saddlepoint 根搜索及分布尾概率均使用 PyTorch；输入在 GPU 时，数值运算保留在 GPU。所有统计计算使用 float64，无需 R 运行时。Python 负责检查收敛条件和将最终标量转换为输出字典。

本模块实现普通单表型分支的两组 Beta 权重、SKAT、burden、ACAT-V、六组注释合并和 STAAR-O。零模型拟合、GDS 读取、样本排列、基因型缺失填补、等位基因方向调整由 pipeline 在调用前完成。二分类 SPA 与多性状 MultiSTAAR 需要各自的实现，不会自动改用普通检验。

## Python 调用及输入输出

```python
import torch
from staar_phewas.statistics import score_covariance, staar_test

# genotype_matrix: [样本数, 变异数]，已按 minor allele 定向及填补缺失。
# phenotype_residual: [样本数]，普通 Gaussian 零模型的未缩放残差。
# covariate_matrix: [样本数, 固定效应数]，包含截距且列线性独立。
# minor_allele_frequency、minor_allele_count: [变异数]，顺序与基因型列一致。
# annotation_phred: [变异数, 注释数]，PHRED 数值，非预先转换的 rank。
genotype_matrix = genotype_matrix.to(device="cuda", dtype=torch.float64)
phenotype_residual = phenotype_residual.to(device="cuda", dtype=torch.float64)
covariate_matrix = covariate_matrix.to(device="cuda", dtype=torch.float64)

variant_score, variant_covariance = score_covariance(
    genotype_matrix,
    phenotype_residual,
    covariates=covariate_matrix,
    dispersion=residual_variance,  # sigma**2，已拟合的残差方差。
)
residual_degrees_of_freedom = sample_count - covariate_matrix.shape[1]
results = staar_test(
    variant_score,
    variant_covariance,
    minor_allele_frequency,
    minor_allele_count,
    annotations=annotation_phred,
    names=annotation_names,
    acat_calibration="gaussian_glm",
    dof=residual_degrees_of_freedom,
    cmac=rare_genotype_sum,  # 筛选后基因型的实际和，保留填补造成的小数。
)
```

`score_covariance` 输出长度为变异数的 `score` 和变异数×变异数的 `covariance`，保持基因型所在设备。普通模型计算 `U = Gᵀr`，`V = dispersion × [GᵀWG − GᵀWX(XᵀWX)⁻¹XᵀWG]`。Gaussian 默认 `W=I`；二分类普通模型传入已经拟合的 `working_weights`。

| 参数 | 含义 |
|---|---|
| `genotype`、`residual` | 已排列好的基因型与残差，无缺失、无非有限值。 |
| `covariates` | 普通模型的固定效应设计矩阵，截距需由调用者提供。 |
| `working_weights` | 普通模型的每样本方差权重，默认全 1。 |
| `dispersion` | 普通模型拟合出的方差尺度，必须为正。 |
| `projector` | 混合模型已拟合的 `P`，计算 `GᵀPG`。支持稀疏矩阵。 |
| `precision` | 稀疏混合模型的 `Sigma_i`，支持 COO/CSR，避免生成密集样本×样本矩阵。 |
| `precision_covariates` | 与上述 precision 对应的 `Sigma_iX`。 |
| `fixed_effect_covariance` | 原零模型的 `cov`，用于 `GᵀSigma_iG − (Sigma_iXᵀG)ᵀ cov (Sigma_iXᵀG)`。 |

必须在 `covariates`、`projector`、`precision` 三种入口中选择一种。混合模型入口传入已缩放残差，`dispersion=1`；拟合尺度已经包含在 `P` 或 `Sigma_i` 中。

`staar_test` 返回按原版顺序排列的字典。首先是 `num_variant`、`cMAC`，随后是六组结果：`SKAT(1,25)`、`SKAT(1,1)`、`Burden(1,25)`、`Burden(1,1)`、`ACAT-V(1,25)`、`ACAT-V(1,1)`；每组包含基础结果、名称为 `基础列名-注释名` 的注释结果和对应的 `STAAR-S/B/A` 合并结果。最后是 `ACAT-O`、`STAAR-O`。没有注释时共 16 列；每新增一列注释增加 6 列。

| 参数 | 含义及默认值 |
|---|---|
| `score`、`covariance` | 上述 U 和已经包含 dispersion 的 V；协方差必须对称。 |
| `maf`、`mac` | 对应变异的 minor allele frequency/count，函数不会再 flip。 |
| `annotations` | PHRED 注释矩阵，可省略，或传入零列矩阵。 |
| `names` | 注释列名，必须唯一；省略时生成 `annotation_1` 等名称。 |
| `acat_calibration` | `chi2`（默认）对应 SMMAT/relatedness 或普通二分类；`gaussian_glm` 对应普通 Gaussian。 |
| `dof` | Gaussian 零模型残差自由度 `n-rank(X)`；ACAT-V 的单变异 t 检验使用 `dof-1`。 |
| `n`、`covariate_count` | 未传 `dof` 时可用两者推算；`covariate_count` 应为设计矩阵秩。 |
| `mac_threshold` | ACAT-V 的低 MAC 合并阈值，默认 10；`MAC<=10` 一起做 burden。 |
| `rare_maf_cutoff` | 保留 `0<MAF<cutoff` 的变异，默认 0.01，使用严格小于。 |
| `rv_num_cutoff` | 最少变异数，默认 2；不足时抛出异常。 |
| `rv_num_cutoff_max` | 变异数必须严格小于此值，默认与原函数相同为 10⁹。实际批次大小还应按显存设置。 |
| `cmac` | 筛选后基因型的实际和；省略时返回筛选后的 MAC 和。缺失填补后建议显式传入。 |

GPU 上的 `V` 仍是变异×变异矩阵，单个 float64 矩阵约占 `8p²` 字节。函数逐列注释计算加权矩阵，不同时生成全部注释的矩阵。按区域限制变异数可控制显存。

## 与 R 数值规则的关系

- 注释变为 `rank = 1−10^(−PHRED/10)`。burden 权重是 `Beta × rank`，SKAT 是 `Beta × sqrt(rank)`，ACAT-V 是 `Beta² × rank / dbeta(MAF,0.5,0.5)²`。每组都保留不使用注释的基础权重。
- SKAT 先将小于 `10⁻⁸` 的特征值设为零，再运行上游 Saddle 的同一二分搜索：区间上界 0.499995、终止宽度 `10⁻⁸`。根的绝对值小于 `10⁻⁴` 时，采用上游二阶和四阶矩的 chi-square fallback；这里保留完整谱计算矩，未换成 Davies 或其他分布近似。
- burden 始终使用单自由度 chi-square。普通 Gaussian 的 ACAT-V 单变异部分使用原版 Student t；低 MAC 的合并 burden 仍使用 chi-square。SMMAT 的单变异 ACAT-V 使用 chi-square。
- ACAT-V 的低 MAC 合并权重是这些变异 ACAT 权重的**均值**，不是总和。
- 公开 `cct` 对应 `R/CCT.R`：出现精确 0 返回 0；出现精确 1 警告并返回 1；同时出现 0 和 1 抛出异常。`internal=True` 对应 ACAT-V 内部的 `CCT_pval.cpp`，保留其不同的精确 1 行为。`p<10⁻¹⁶` 与合并统计量大于 `10¹⁵` 的分支也保留。
- 原版 CCT 对全部零权重产生 NaN，本实现抛出 `DegenerateTestError`。原版 Saddle 在非退化谱但零统计量时具有无穷搜索区间，本实现返回数学上对应的 1。退化协方差、缺失注释或不合法自由度均明确报错。

低层 R 对照应使用与零模型分支一致的原始函数，例如：

```r
# 普通模型：weights_B/S/A 依照 STAAR 的 R 包函数构造。
reference <- STAAR:::STAAR_O(
  G, X, working, sigma, fam, residuals,
  weights_B, weights_S, weights_A, mac
)
# relatedness 稀疏分支：residuals 是 scaled.residuals。
reference_sparse <- STAAR:::STAAR_O_SMMAT_sparse(
  G, Sigma_i, Sigma_iX, cov, residuals,
  weights_B, weights_S, weights_A, mac
)
```

这些是数值内核的对照入口，完整 pipeline 的区域选择和最终结果还需要各自的 R wrapper 对照。

## 版本与验证记录

2026-10-04：从作者推荐的 `zilinli/staarpipeline:0.9.9` 镜像的已安装包 DESCRIPTION/NAMESPACE 锁定依赖：作者 [STAAR 0.9.9 源码](https://github.com/yuxinyuanqt/STAAR/tree/4bbf77ba8a90894a434f5eb4473d540e172dad05)、[STAARpipeline 0.9.9 源码](https://github.com/yuxinyuanqt/STAARpipeline/tree/fbce778bf14cc4f9e892989a194c64bae2670311)、[STAARpipelinePheWAS 0.9.7.1 源码](https://github.com/yuxinyuanqt/STAARpipelinePheWAS/tree/6b72cf9d1f5ef001887b37d1d77a0c5370c46be0)。其中 `STAAR_sp` 确实由该作者 STAAR 版本导出。

本次核对 `STAAR_sp` 的区域筛选、注释权重、分支选择和输出列；`STAAR_O`、`STAAR_O_SMMAT`、`STAAR_O_SMMAT_sparse`、Saddle、K/K1/K2、Bisection 和 CCT 内核与最初参照的主仓库源码相同。完整 R oracle 使用上述冻结依赖，保留 `STAAR_sp` 的原始实现。

数值单元检查包含显式矩阵投影与稀疏 precision 的一致性、Gaussian 方差尺度、注释权重公式、CCT 精确边界和极小尾概率、SKAT moment fallback、低 MAC 合并、Gaussian Student t，以及 float64 CPU/CUDA 结果对照。这些小型固定矩阵用于验证数值规则，**不作为真实数据 benchmark**。

真实单表型核心对照使用同一脑影像性状的 42,652 个完整样本，OR11H1、COMT、APOBEC3A 各自的 pLoF 和 missense 集合，另加 COMT synonymous，共七集，每集四列 PHRED 注释、40 个输出字段。参考运行使用上述镜像中的原始 `fit_nullmodel` 和 `STAAR_sp`，未修改原包源码。使用完全相同的 R score、协方差输入时，七集的全部 280 个输出字段通过 `abs=10⁻¹⁰ + rel=10⁻⁷` 的对照标准，P 值最大绝对差为 2.06×10⁻⁹。

修正零模型的求值顺序后，从同一冻结表型向量开始、由 GPU 原生拟合并生成 score/协方差的七集完整核心链也全部通过上述标准。这轮固定输入对照中，42,652 个样本的五次 AI 更新和最终方差参数、缩放残差、固定效应协方差与原版 R 逐位相同。下表记录这轮核心验证；基因型矩阵由真实数据按相同样本顺序和变异方向提取。

| 真实集合 | 变异数 | 最大 P 值绝对差 | GPU 核心耗时 | R 核心耗时 |
|---|---:|---:|---:|---:|
| OR11H1 pLoF | 3 | 5.97×10⁻¹⁶ | 2.056 s，首次 CUDA 初始化 | 1.286 s |
| OR11H1 missense | 93 | 1.76×10⁻⁹ | 0.220 s | 0.965 s |
| COMT pLoF | 5 | 5.55×10⁻¹⁶ | 0.203 s | 0.953 s |
| COMT missense | 74 | 3.89×10⁻¹⁵ | 0.157 s | 0.974 s |
| APOBEC3A pLoF | 9 | 8.22×10⁻¹⁵ | 0.177 s | 0.944 s |
| APOBEC3A missense | 166 | 2.63×10⁻¹³ | 0.245 s | 1.014 s |
| COMT synonymous | 44 | 2.05×10⁻⁹ | 0.176 s | 1.060 s |

原 Saddle 在接近均值处具有数值敏感性：早期拟合实现的约 10⁻¹¹ 二次型相对差，会被 `log(v/w)/w` 放大到约 10⁻⁵ 的 P 值差。修复没有调整 Saddle 搜索容差，而是复现原版投影点积与矩阵求解的舍入顺序。`numerics.reference_crossprod` 对单列固定效应使用 16 个 float64 FMA 累加器，保留 8 元素尾块、剩余尾块及水平合并次序；计算均为 PyTorch GPU。TorchScript 对每种新布局先预热，使首次拟合与随后拟合都使用融合的 FMA 运算。AI 矩阵保留 R 的列主序后交给 `torch.linalg.solve`；未使用私有 C 诊断或原 BLAS 作为生产后端。

零模型初始化的样本方差也保持 R `cov.c` 的扩展精度求值：`numerics.extended_variance` 用 float64 高、低两部分表示累加误差，在中心化、平方和除以 `n-1` 时保留误差项，最后才舍入成一个 float64 数值。这些步骤均由 Torch Tensor 在输入设备计算。生产 RINT 则使用原 R AS241 的系数和求值次序；真实 42,652 元素输入的转换结果已与原 `qnorm` 逐位对照。完整 pipeline 文档记录从这组原始表型重新转换并拟合的最终结果，不将冻结向量验证的逐位结论扩展到其他输入和运行库。

这一舍入对照锁定 GMMAT 1.3.2、Matrix 1.2-17、MKL 2019 Update 2 AVX512 的原版参考，以及验证环境的 PyTorch 2.5.1 / CUDA 11.8。多列固定效应使用通常的 float64 Torch 矩阵乘法，尚未据此声明多列模型逐位等价。矩阵点积单元检查包含不同尾块长度和非连续输入，均为单独生成的数值规则检查，不作为真实数据 benchmark。

表中时间包含核心 score 和检验，不包含 GDS 读取和零模型拟合；原版运行在用户空间容器中，不能据此概括端到端 GPU 加速比。完整 pipeline 的区域选择、输出文件及端到端结果由对应文档记录。

同一真实表型和对角 GRM 的零模型拟合：原版 R 为 4.525 s，GPU 为 2.530 s，均收敛于五次 AI 更新。GPU 时间排除输入读取和 CUDA context 初始化，包含点积的首次 TorchScript 预热。零模型与基因集合计时分开记录，避免将核心算子的耗时解释为完整 pipeline 加速比。

## 原实现、许可及文献

本模块按 GPL-3.0-only 提供，保留 [STAAR 作者与 GPL-3 许可出处](https://github.com/yuxinyuanqt/STAAR/blob/4bbf77ba8a90894a434f5eb4473d540e172dad05/DESCRIPTION)。源码对应 [R/STAAR_sp.R](https://github.com/yuxinyuanqt/STAAR/blob/4bbf77ba8a90894a434f5eb4473d540e172dad05/R/STAAR_sp.R)、[Saddle.cpp](https://github.com/yuxinyuanqt/STAAR/blob/4bbf77ba8a90894a434f5eb4473d540e172dad05/src/Saddle.cpp)、[CCT.R](https://github.com/yuxinyuanqt/STAAR/blob/4bbf77ba8a90894a434f5eb4473d540e172dad05/R/CCT.R) 和相关 `STAAR_O` 内核。

1. Li X, Li Z, et al. Dynamic incorporation of multiple in silico functional annotations empowers rare variant association analysis of large whole-genome sequencing studies at scale. *Nature Genetics* 52, 969–983 (2020). [DOI](https://doi.org/10.1038/s41588-020-0676-4).
2. Li Z, Li X, et al. A framework for detecting noncoding rare-variant associations of large-scale whole-genome sequencing studies. *Nature Methods* 19, 1599–1611 (2022). [DOI](https://doi.org/10.1038/s41592-022-01640-x).
3. Liu Y, et al. ACAT: A fast and powerful p value combination method for rare-variant analysis in sequencing studies. *American Journal of Human Genetics* 104, 410–421 (2019). [DOI](https://doi.org/10.1016/j.ajhg.2019.01.002).
4. Liu Y, Xie J. Cauchy combination test: a powerful test with analytic p-value calculation under arbitrary dependency structures. *Journal of the American Statistical Association* 115, 393–402 (2020). [DOI](https://doi.org/10.1080/01621459.2018.1554485).
