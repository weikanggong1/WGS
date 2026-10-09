> 0.5.0 的长 mask 使用 FastSKAT 混合谱和 4096 缓存协方差；完整谱/近似边界与真实验证见[版本说明](hybrid_validation.md)，三输入调用见[运行指南](cache_only_run.md)。

# Torchstaar统计API与完整谱后端

## 功能与输入输出

本页说明已经形成Score/协方差后的低级统计API；完整GDS、零模型、15mask调度与原生输出由 [Torchstaar完整指南](torchstaar.md)说明。生产CLI默认TF32/FP32，必要标量概率使用FP64。低级函数保留显式FP64控制，示例必须指定实际模式；低级API计算不包含读取/拟合/写出，不能替代完整benchmark。

输入G为N×M minor dosage，r为N残差，X为N×C满秩固定效应设计；MAF/MAC长M，PHRED注释为M×A。所有行列顺序和样本/变异轴已绑定，dose已填补，函数不再flip。混合模型使用scaled residuals与同模型precision，不将样本方阵dense化。

## Python与参数

```python
import torch
from torchstaar.statistics import score_covariance, staar_test

# 数据来自同一模型和mask；下列变量须由调用者提供，均无缺失/非有限值。
genotype_matrix = genotype_matrix.to(device="cuda:0", dtype=torch.float32)  # [N,M]
phenotype_residual = phenotype_residual.to(device="cuda:0", dtype=torch.float32)  # [N]
covariate_matrix = covariate_matrix.to(device="cuda:0", dtype=torch.float32)  # [N,C]含截距
variant_score, variant_covariance = score_covariance(
    genotype_matrix, phenotype_residual, covariates=covariate_matrix,
    dispersion=residual_variance, matmul_mode="tf32",
)
statistical_result = staar_test(
    variant_score, variant_covariance, minor_allele_frequency, minor_allele_count,
    annotations=annotation_phred, names=annotation_names,
    acat_calibration="gaussian_glm", dof=sample_count-covariate_rank,
    cmac=rare_genotype_sum, matmul_mode="tf32",
    tail_optimization=True, weight_batch_optimization=True,
    output_batch_optimization=True, cct_validation_optimization=True,
)
```

普通模型为 `U=Gᵀr`、`V=dispersion×[GᵀWG−GᵀWX(XᵀWX)⁻¹XᵀWG]`；默认W=I。混合模型以precision与设计投影形成同一V。输出Score长M、协方差M×M，保持设备。直接统计函数的求谱默认调用Torch；CLI的run专属context才按weighted_eigensolver设置选择新后端。

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
| `matmul_mode` | 字符串，低级API默认 `fp64`控制；生产 CUDA 使用原生 `tf32`，FP32 累加与输出。配置与精度边界见 [Torchstaar完整指南](torchstaar.md)。 |

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
| `matmul_mode` | 字符串，默认 `fp64` 对照；生产 pipeline 传入 `tf32`，控制统计内部 burden 矩阵乘法。 |
| `tail_optimization` / `weight_batch_optimization` | 布尔值，低级 API 默认均为 `False`；生产 CLI 默认开启尾部同步优化和每个 mask 内权重批量复用。 |

| 其他公开参数 | 默认与用途 |
|---|---|
| `output_batch_optimization` | True；保留各原CCT计算，最后一次复制完整输出P。False为独立逐项复制控制。 |
| `cct_validation_optimization` | True；合并ACAT-V输入flags回传，保留最后NaN检查。False为独立验证边界控制。 |

## 原公式与概率边界

Burden是 `(wᵀu)²/(wᵀVw)`；SKAT使用 `Σ(w_i u_i)²` 与 `D_wVD_w` 的全部谱。精确比例权重以FP32数的精确FP64交叉乘积证明，不使用容差相似或截断谱。原Saddle谱阈值1e-8、根宽度1e-8、moment切换abs(root)<1e-4保持。CCT保留P<1e-16的小P分支与大statistic>1e15的原尾部；exported CCT的精确0返回0、精确1发警告返回1、同时0/1报错，internal ACAT-V保留原内部规则。invalid/zero weight mass明确拒绝；原P下溢不通过clamp掩盖。

`cct(pvalues,weights=None,internal=False,sync_light=False)` 输入一维有限[0,1] P及同长度有限非负权重；省略权重时等权，权重总质量须有限且>0；返回Python float。`quadratic_form_sf(statistic,eigenvalues,moment_eigenvalues=None)` 输入非负有限statistic及完整有限谱，返回原Saddle/moment概率。`annotation_weights(maf,annotations=None,dtype=torch.float64)` 输入MAF长M/PHRED M×A，返回三组Beta/annotation权重；生产native以FP32设备变换。`names`仅影响原输出列标签，不改变公式。

## 完整谱后端

```python
from torchstaar.cuda_eigen import FP32SmallSpectrumSolver

# 已经生成CUDA FP32 [B,M,M]完整对称weighted矩阵；不含Score/读取/概率。
with FP32SmallSpectrumSolver(memory_limit=40 * 2**30) as spectrum_solver:
    full_eigenvalues = spectrum_solver.eigvalsh(weighted_matrices, UPLO="U")
spectrum_execution = spectrum_solver.report()  # 关闭后记录cleanup/info/library proof。
```

| API/参数 | 输入和输出 |
|---|---|
| `FP32SmallSpectrumSolver(memory_limit=40*2**30)` | 正整数字节预算，默认40GiB，拥有一个串行selector；首个selected CUDA调用才加载官方CUDA库。 |
| `expected_solver_sha256` / `expected_runtime_sha256` | 可选既有cuSOLVER/runtime SHA字符串；提供则精确匹配。实际SHA总是记录，package hash存在时核对。 |
| `eigvalsh(matrix,UPLO='U')` | selected为strided CUDA FP32、requires_grad=False的 `[M,M]`/`[B,M,M]`，33≤M≤512；返回升序全谱 `[M]`/`[B,M]` FP32。范围外由原Torch处理；实际失败及后端由执行报告记录。 |
| `report()` | 实际selected/outside calls、矩阵/维度、API/library/status与cleanup；不隐含科学验收。 |
| `close()` / context退出 | 在创建线程释放ownedhandle，失败抛错；closed后禁止重用，不改global Torch/preference。 |

## CLI、原R与真实验证

本页低级函数没有独立数据读取CLI。完整任务使用 `torchstaar-config expanded_configuration.json --device cuda:0 --weighted-eigensolver auto --report private/report.json`；配置与全部CLI参数见主指南。原R对应 `STAAR::STAAR`/`STAAR_sp`、`CCT`及 `Saddle`，锁定实现见下方。

本版真实测量只覆盖完整 Single 与有界原 R Single。低级集合统计、完整谱或 mock 检查不能代替当前 gene-based 端到端对照；范围见 [主指南](torchstaar.md#真实验证与计时范围)。`eigen_tail` 包含整个 staar_test，内部计时不可相加或称纯 eig 时间。

## 更新、原实现与文献

当前版提供完整FP32小谱API、实际后端报告和ownedcontext；原权重、完整谱与尾概率规则保留。近期完整版本表见主指南，不另重复过期实验过程。

[STAAR固定实现](https://github.com/yuxinyuanqt/STAAR/tree/4bbf77ba8a90894a434f5eb4473d540e172dad05)，[原Score](https://github.com/li-lab-genetics/STAAR/blob/master/src/Indiv_Score_Test_SMMAT_sparse.cpp)、[Saddle](https://github.com/li-lab-genetics/STAAR/blob/master/src/Saddle.cpp)、[CCT](https://github.com/li-lab-genetics/STAAR/blob/master/src/CCT_pval.cpp)。参考文献见 [Torchstaar参考](torchstaar.md#参考)。
