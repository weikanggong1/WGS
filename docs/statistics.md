# STAAR 数值内核

`staar_phewas.statistics` 从已经拟合好的零模型残差和基因型矩阵计算 score、协方差及 STAAR 检验。score、协方差、burden、Cauchy 合并、saddlepoint 根搜索及分布尾概率使用 PyTorch CUDA，计算采用 float64，无需 R 运行时。标量注释权重转换使用 CPU `libm`；少量近均值谱使用锁定的 CPU LAPACK 后返回 CUDA。这个精度边界及必需的独立 Conda 安装见 [precision](precision.md#保持原-saddle-舍入的精度边界)。Python 负责检查收敛条件和将最终标量转换为输出字典。

本模块实现普通单表型分支的两组 Beta 权重、SKAT、burden、ACAT-V、六组注释合并和 STAAR-O。零模型拟合、GDS 读取、样本排列、基因型缺失填补、等位基因方向调整由 pipeline 在调用前完成。二分类 SPA 与多性状 MultiSTAAR 需要各自的实现，不会自动改用普通检验。

### Wrapper 输入的频率与填补

复现原 wrapper 时，MAF 也属于统计输入。Base STAARpipeline 0.9.9 从 REF_AF 的补数获得 MAF；PheWAS 0.9.7.1 在每个表型样本中重新以非缺失 MAC 除以 `2 * (N - missing_count)`。两条 `frequency_mode` 语义分别为 `reference` 和 `count`，各自用 `2 * MAF` 做 mean 填补。完整规则和锁定源码见 [base/PheWAS 频率对照](base_reference_inventory.md#base-与-phewas-的频率和缺失填补)。

在样本和 minor 方向相同、没有半缺失 genotype 时，这些公式在实数算术下等价，浮点减法、除法和填补的先后顺序会改变最后几位。半缺失情况还须区分 SeqArray 的 allele AF/AC/missing rate 和整个 dosage 调用的缺失。并集初始 MAC 是 `min(REF_AC, 2*round(N*(1-missing_rate))-REF_AC)`；它与 base minor 填补后从 MAF 恢复 MAC 的公式不同，PheWAS 每 trait 另按完整 dosage 计数。[具体粒度及合成规则单测](gds.md#半缺失-genotype-的两种统计粒度)。MAF 影响 Beta 权重，填补值影响 U/V；接近谱均值的 Saddle 结果还会放大部分舍入差。输入的 genotype、MAF、缺失填补和 U/V 应分别与实际 R wrapper 对照，再检查最终 P 值。

2026-10-05，执行源码 `c6362c6d392dce8c29668a9563a88182714571ebc8aa68285e5988a7ad423300` 的 chr21、一个连续表型完整串行流程通过验收：795 项任务、15 个 mask、18 个正式文件，18 项 `serial_vs_R` 关联比较、八份 Single 元数据原 R 读回和一项零模型比较均通过。161,839 个数值字段、3,343,119 个数值单元格全部满足 `|GPU-R| <= 1e-10 + 1e-7*|R|`，结构和数值超限差异为零，最大绝对/相对差为 `1.0913936421275139e-10` / `1.7739502638151633e-9`。重新拟合并写出的零模型，12 个数值字段、341,221 个数值单元格与 R 差异为 0。该结果绑定实际核对的 33 个执行文件及 108 个快照文件；完整范围见 [全量对照记录](base_reference_inventory.md)，耗时和步骤边界见 [性能分析](performance.md)。

四份 Single 结果共 318,132 行，包括 238,947 个双等位 SNV 和 79,185 个双等位 Indel，N 均为 42,652。实际解码记录 REF_AF tie 1 个、半缺失与全部 allele 缺失 0 个、最大两层 Bit2。正式串行进程成功执行 23,620 次 Triton CUDA 有序相加、34 次 FP64 FMA 投影；GPU 计算 43,212 个加权谱矩阵，651 个近均值加权行使用固定 `66N` 工作区的参考 CPU 谱后返回 CUDA 尾概率。完整 GPU 实验按用户要求只验收串行模式；未完成的批量运行取消，不提供完整批量性能结论。

历史冻结源码 `56162751…` 的全染色体 base 验收保留 `strict_failed`：五个 Coding 正式文件严格通过，首个 Noncoding 文件的 20,358 个数值单元格中有两个超出原容差，结构差异为零。后续 `8520cf24…` 和 `5b0dfd83…` 的历史失败见 [全量记录](base_reference_inventory.md)。下文七个真实核心集合及批量局部回归保持各自原输入、源码和验证范围。

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

GPU 上的 `V` 仍是变异×变异矩阵，单个 float64 矩阵约占 `8p²` 字节。大矩阵逐列注释计算加权矩阵；CUDA 上变异数不超过 32 时，真实注释权重组成自然批次，工作区与矩阵尺寸见下文。按区域限制变异数可控制显存。

## 有序稀疏求和与预算

对角 `Sigma_i` 的原稀疏实现按已存储行的次序累加乘积。GPU 路径按样本行号整理每列非零 genotype，依次求 U、`Sigma_iXᵀG` 和 `GᵀSigma_iG`，再扣除固定效应投影。乘法各自舍入为 FP64 后才相加，保留原稀疏求和顺序。零模型入口通过 `score_covariance(reduction="reference_sparse", max_workspace_bytes=...)` 显式选择此路径，默认 reduction 为 `blas`；非对角 precision 不接受这一有序方法。算法来源是 [STAAR_O_SMMAT_sparse.cpp](https://github.com/yuxinyuanqt/STAAR/blob/4bbf77ba8a90894a434f5eb4473d540e172dad05/src/STAAR_O_SMMAT_sparse.cpp)。

Torch 生成乘积，Triton 在一次 GPU kernel 中沿行顺序相加，每个输出列独立累加。这样可保留逐项 FP64 加法，同时减少按样本行逐次提交 kernel 的开销。该 kernel 显式关闭浮点融合，未使用 TF32；FP32/TF32 的专项精度验证尚未通过，默认统计精度继续采用 FP64。本轮冻结源码已完成整条 chr21 串行链并通过原 R 严格验收，局部集合记录和完整结果分别保留在 [原版对照记录](base_reference_inventory.md)。

这一有序稀疏相加 kernel 在 Triton 不可用或遇到已识别的兼容错误时，使用输入设备上的 TorchScript 路径并发出一次 warning。诊断字段 `ordered_addition` 记录 `triton_sequential` 或 `torchscript_sequential`；兼容切换另记录 `ordered_addition_fallback_reason`。两种执行路径的耗时应分别记录。该回退规则不适用于下文要求显式 FMA 的 CUDA 投影点积。

`sparse_execution_metadata()` 汇总当前进程真正成功执行的有序相加次数，不把导入模块或允许 Triton 当作已使用该 backend。它返回 `ordered_addition_backend`（`triton`、`torchscript`、`mixed` 或 `not_used`）、总次数、各 backend/设备次数及实际切换原因。计数是累计值；同一进程比较两段执行应记录调用前后的差值。

```python
from staar_phewas.sparse_numerics import sparse_execution_metadata

# 在实际 score/covariance 计算后记录，不以环境配置代替执行证据。
ordered_execution = sparse_execution_metadata()
print(ordered_execution["ordered_addition_backend"])
print(ordered_execution["ordered_addition_device_call_counts"])
```

临时工作空间通过 `max_workspace_bytes` 规划，默认 256 MiB。列分块先按全部样本都非零的最坏情况估算排序和索引空间；协方差的列对及固定效应按剩余预算分块，不建立“非零样本×全部变异×全部变异”的中间 Tensor。预算不足一个最小块时明确报错。输入 genotype、最终变异×变异协方差、固定效应交叉积和 CUDA allocator 的预留缓存另计；该参数不是总显存硬上限，最终须报告进程的 peak allocated/reserved。

Triton 首次使用时编译 GPU kernel。较旧 NVIDIA driver 与环境捆绑的 PTX 编译器可能不兼容；可由用户提供与 driver、GPU 和 Triton 版本相容的 `ptxas`，在启动前选择它。Triton 3.1.0 的 [官方 compiler.py](https://github.com/triton-lang/triton/blob/v3.1.0/third_party/nvidia/backend/compiler.py) 优先读取这一环境变量：

```bash
# 指向用户已安装并确认兼容的 CUDA 工具链。
export TRITON_PTXAS_PATH="/path/to/compatible/cuda/bin/ptxas"
"$TRITON_PTXAS_PATH" --version
python -m staar_phewas.cli analysis.json
```

编译和首次 CUDA 初始化应单独计时；它们不能用重复调用的统计微基准时间代替。

## 可选批量统计与单变异方差

`staar_phewas.batch_statistics.staar_test_batch` 接收由上述 `staar_test` 参数字典组成的列表，返回相同顺序的结果字典列表。每个字典的字段及字段顺序与串行调用一致。它把变异数相同的 mask 的加权协方差一起提交给 `torch.linalg.eigvalsh`，再并行运行独立的 Saddle 二分搜索；burden、ACAT-V 和最终 CCT 保留已有实现。生产默认仍是串行 FP64；pipeline 可显式选择 `statistics_execution="batched"`。

```python
from staar_phewas.batch_statistics import staar_test_batch

# 每个 mask 已按同一零模型生成 score 和 covariance。
mask_inputs = [
    dict(score=mask_score, covariance=mask_covariance,
         maf=mask_maf, mac=mask_mac,
         annotations=mask_phred, names=annotation_names,
         cmac=mask_genotype_sum),
    # 后续 mask 也使用相同的参数结构。
]
mask_results, batch_diagnostics = staar_test_batch(
    mask_inputs,
    max_workspace_bytes=256 * 1024**2,  # 额外加权矩阵的工作区预算。
    return_diagnostics=True,
)

# 单变异检验只需协方差对角线，避免生成整块变异×变异矩阵。
variant_score, variant_variance = gaussian_null_model.individual_score_variance(
    genotype_matrix  # [样本数, 变异数]，已定向、填补并按零模型排列。
)
```

`max_workspace_bytes` 默认 256 MiB，按加权输入、求解器副本和中间缓冲估算额外 mask 批次大小；调用者持有的基因型与协方差另计。单个矩阵超过该预算时，该 mask 使用串行方法。对于 CUDA 小谱，容量不足两个加权矩阵或最后一块仅剩一个矩阵时，将整个受影响 mask 交给串行入口，保留真实注释权重的自然批次。`return_diagnostics=False` 时只返回结果列表；设为 `True` 时还返回 mask 数、加权矩阵数、特征值批次数、敏感行重算数、回到串行的 mask 数和最大加权输入大小。这一预算不约束串行入口内的注释权重栈，也不是分析总显存上限；整个分析仍须计入零模型、基因型、原协方差和求解器缓冲。

批量按精确变异数分组，不以补零统一矩阵大小：原 Saddle 的搜索下界使用谱的长度，补零会改变数值求值。特征值截断、根搜索宽度和 moment 分支阈值保持原值。批量由初始区间宽度确定足够的二分步数，每行仍按原宽度及零导数条件停止更新，从而避免逐步读取 GPU 收敛状态。批量搜索对近均值行重新调用串行 FP64 Saddle，K1 区间筛选决定是否使用参考 CPU 谱校正；保留协方差原上三角的求值顺序。其余 CUDA 的 `m<=32` 且批次含多个矩阵时复用自然批量 Jacobi 谱，其他情况沿用逐矩阵特征值求解。这只选择计算路径，原 moment 切换仍为 `abs(root)<0.0001`；q、K 的求值公式和容差保持原规则。未使用 FP32 或 TF32 替代统计精度。

本轮 Gaussian STAAR 串行入口对 CUDA `m<=32` 的真实注释权重矩阵组成自然批次，再逐列计算尾概率；这是单个 mask 内部的矩阵计算，任务入口仍为 `statistics_execution="serial"`。CPU、大矩阵和仅一列权重保留原求值路径。以 12 项 PHRED 加无注释权重、两组 Beta 为例，32 变异的 26 个实际加权输入约占 `26*32*32*8 = 212,992` 字节，其他分析内存另计。

历史冻结源码 `8520cf24…` 的 15 个真实集合局部对照中，串行、批量各 984 个返回字段满足原容差，字段顺序一致；最大绝对差 `1.936e-8`、相对差 `6.038e-8`，其完整运行后来在 Noncoding 368 中断。历史 `5b0dfd83…` 的 18 个真实集合两模式各 1,248 个字段通过，最大绝对/相对差为 `8.899e-10` / `2.721e-9`，其完整运行后来在 Noncoding 367 中断。`c6362c6d…` 将敏感 DSYEV 工作区固定为原 Armadillo 的 `66×阶数`，19 个真实集合两模式各 1,336 个字段严格通过，最大绝对/相对差为 `8.899e-10` / `2.721e-9`；随后完成了本页开头的完整串行 18/8/1 验收。局部批量结果保持原范围，不用于本轮完整批量结论。

`individual_score_variance` 使用同一 GRM 块旋转和 precision，逐列求 `diag(GᵀSigma_iG)`，扣除固定效应投影的对角项，输出两个长度为变异数的 FP64 Tensor。分配量由变异×变异改为样本×变异，适合分块扫描全染色体。方差没有额外截断。

2026-10-04 的真实缓存验证包含上述七集、42,652 个样本，每集四列 PHRED、共 280 个字段。批量与串行的最大绝对差为 `4.04×10⁻¹⁴`；批量与原 R 的最大 P 值绝对差为 `2.052×10⁻⁹`，全部通过 `abs=10⁻¹⁰ + rel=10⁻⁷`，字段顺序一致。单变异 score 与完整协方差路径一致，方差对角线的最大绝对差为 `5.684×10⁻¹⁴`。七集共 70 个加权矩阵组成 7 个特征值批次，7 行使用敏感路径重算；最大加权输入为 2.10 MiB。

此前七集批量版本在同一共享 A100 上预热后各测三次，统计的中位时间为串行 1.118 s、批量 0.611 s；使用原 R 的相同 score/协方差时为 1.374 s、0.618 s。对应批量源码 SHA 为 `5ceebf1a…`。早期逐步读取批量收敛状态的版本曾测得 3.392 s、1.673 s；两轮串行基准本身也有变化，不能将跨轮时间差全部归因于这次减少同步。这些历史时间只包括统计调用，不包括读取、mask 构建、零模型拟合和输出，也未据此推算全染色体加速比。本轮完整串行的耗时和步骤边界见 [性能分析](performance.md)，批量模式不作完整结论。

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

这些是数值内核的对照入口。本轮 base 串行的区域选择和最终正式文件已按原 wrapper 验收，其他入口按各自的 R wrapper 对照。

## 版本与验证记录

2026-10-04：从作者推荐的 `zilinli/staarpipeline:0.9.9` 镜像的已安装包 DESCRIPTION/NAMESPACE 锁定依赖：作者 [STAAR 0.9.9 源码](https://github.com/yuxinyuanqt/STAAR/tree/4bbf77ba8a90894a434f5eb4473d540e172dad05)、[STAARpipeline 0.9.9 源码](https://github.com/yuxinyuanqt/STAARpipeline/tree/fbce778bf14cc4f9e892989a194c64bae2670311)、[STAARpipelinePheWAS 0.9.7.1 源码](https://github.com/yuxinyuanqt/STAARpipelinePheWAS/tree/6b72cf9d1f5ef001887b37d1d77a0c5370c46be0)。其中 `STAAR_sp` 确实由该作者 STAAR 版本导出。

本次核对 `STAAR_sp` 的区域筛选、注释权重、分支选择和输出列；`STAAR_O`、`STAAR_O_SMMAT`、`STAAR_O_SMMAT_sparse`、Saddle、K/K1/K2、Bisection 和 CCT 内核与最初参照的主仓库源码相同。完整 R oracle 使用上述冻结依赖，保留 `STAAR_sp` 的原始实现。

数值单元检查包含显式矩阵投影与稀疏 precision 的一致性、Gaussian 方差尺度、注释权重公式、CCT 精确边界和极小尾概率、SKAT moment fallback、低 MAC 合并、Gaussian Student t，以及 float64 CPU/CUDA 结果对照。这些小型固定矩阵用于验证数值规则，**不作为真实数据 benchmark**。

真实单表型核心对照使用一个连续表型的 42,652 个完整样本，匿名集合 A、B、C 各自的 pLoF 和 missense 集合，另加集合 B 的 synonymous，共七集，每集四列 PHRED 注释、40 个输出字段。参考运行使用上述镜像中的原始 `fit_nullmodel` 和 `STAAR_sp`，未修改原包源码。使用完全相同的 R score、协方差输入时，七集的全部 280 个输出字段通过 `abs=10⁻¹⁰ + rel=10⁻⁷` 的对照标准，P 值最大绝对差为 2.06×10⁻⁹。

修正零模型的求值顺序后，从同一冻结表型向量开始、由 GPU 原生拟合并生成 score/协方差的七集完整核心链也全部通过上述标准。这轮固定输入对照中，42,652 个样本的五次 AI 更新和最终方差参数、缩放残差、固定效应协方差与原版 R 逐位相同。下表记录这轮核心验证；基因型矩阵由真实数据按相同样本顺序和变异方向提取。

| 匿名 benchmark 集合 | 变异数 | 最大 P 值绝对差 | GPU 核心耗时 | R 核心耗时 |
|---|---:|---:|---:|---:|
| A pLoF | 3 | 5.97×10⁻¹⁶ | 2.056 s，首次 CUDA 初始化 | 1.286 s |
| A missense | 93 | 1.76×10⁻⁹ | 0.220 s | 0.965 s |
| B pLoF | 5 | 5.55×10⁻¹⁶ | 0.203 s | 0.953 s |
| B missense | 74 | 3.89×10⁻¹⁵ | 0.157 s | 0.974 s |
| C pLoF | 9 | 8.22×10⁻¹⁵ | 0.177 s | 0.944 s |
| C missense | 166 | 2.63×10⁻¹³ | 0.245 s | 1.014 s |
| B synonymous | 44 | 2.05×10⁻⁹ | 0.176 s | 1.060 s |

原 Saddle 在接近均值处具有数值敏感性：v0.1 区域验证中，早期拟合实现的约 10⁻¹¹ 二次型相对差，会被 `log(v/w)/w` 放大到约 10⁻⁵ 的 P 值差。当时的修复保持 Saddle 搜索容差，复现原版投影点积与矩阵求解的舍入顺序。`numerics.reference_crossprod` 对单列固定效应使用 16 个 float64 FMA 累加器，保留 8 元素尾块、剩余尾块及水平合并次序；TorchScript 对新布局先预热，AI 矩阵保留 R 的列主序后交给 `torch.linalg.solve`。这段说明对应区域验证版本，不能由预热或区域通过推断全量等价。该区域版本未使用私有 C 诊断或原 BLAS 作为生产后端。后续完整验收发现的近均值谱精度校正采用上文独立安装的参考 LAPACK；其余统计保持 CUDA。

CUDA 单列投影使用 Triton 的 `libdevice.fma_rn` 显式执行 FP64 乘加，保留 16 路独立累加器、8 元素尾块和余数尾块，以及固定水平合并次序。先前 TorchScript `addcmul` 在真实输入的个别累加路上未保持这一乘加舍入；投影点积约 `4.337e-19` 的差异进入最终残差和尺度，并影响敏感尾概率。对同一真实 42,652 样本的 Y、X、GRM 输入重新拟合后，新实现的五次 AI 迭代及最终零模型数值与原 R 逐位一致；未注入原 R 的拟合参数。本轮原始表型经 RINT 和重新拟合写出的零模型已通过 341,221 个数值单元格的原 R 回读，完整串行 18 批次也已通过上述验收。

CUDA 单列参考投影要求 Triton >= 3.0；缺少依赖或 CUDA/PTX 工具链不兼容时明确报错，不改用先前的 TorchScript 乘加路径。[固定 Conda 环境](../environments/staar-gpu.yml)的 PyTorch 2.5.1 / Python 3.12 / CUDA 11.8 对应官方依赖 `torchtriton 3.1.0`，满足最低版本；稳定 channel 的包名是 `torchtriton`。[官方包元数据](https://api.anaconda.org/release/pytorch/pytorch/2.5.1)及 [PyTorch 安装说明](https://pytorch.org/get-started/previous-versions/)给出版本来源。首次真实调用仍须成功编译 kernel，较旧 driver 的 PTX 编译器选择见前文。SDK 整数解码的 CPU 回退与有序稀疏相加的同设备回退各自独立，不适用于这一投影内积。

```python
from staar_phewas.numerics import reference_dot_execution_metadata

# 在实际零模型拟合/投影后读取，记录成功执行的 CUDA FMA 次数。
projection_execution = reference_dot_execution_metadata()
print(projection_execution["reference_dot_backend"])
print(projection_execution["reference_dot_cuda_call_count"])
```

CLI 报告的 `reference_projection_execution` 同样读取实际计数：backend 为 `triton_fp64_fma16` 或 `not_used`，不把导入 Triton 当作已执行。计数是进程累计值；零模型局部检查实际完成 34 次 FMA launch，不能将单次内核耗时作为整个 pipeline 的速度收益。

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
