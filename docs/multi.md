# 联合多表型 Gaussian MultiSTAAR

`staar_phewas.multi` 在同一批完整案例样本上联合建模多个连续表型，估计表型间协方差，再计算联合 SKAT、burden、ACAT-V 和 MultiSTAAR-O。矩阵运算、特征值、卡方尾概率、鞍点近似和 Cauchy 合并均使用 PyTorch float64；生产分析不调用 R。

```mermaid
flowchart LR
  A[多个表型与样本交集] --> B[共同完整案例与可选列内 RINT]
  B --> C[联合 Gaussian 零模型]
  D[原生 GDS 分块与统一 minor 方向] --> E[共同样本插补和 MAF]
  C --> F[联合 score 与完整 trait covariance]
  E --> F
  F --> G[联合 SKAT / burden / ACAT-V]
  G --> H[MultiSTAAR-O 和各注释结果]
```

## 输入与输出

`fit_joint_gaussian_null` 的输入 `phenotypes` 是 `[样本数 n, 表型数 t]` 的有限连续值矩阵，`t>=2`。所有列须来自相同样本、相同顺序，先移除任一列缺失的行。`covariates` 是包含截距的 `[n,p]` 满秩矩阵；省略时使用截距。输入既有残差化表型时，额外协变量须与实际分析设计一致。

输出 `JointGaussianNullModel` 保存 `[p,t]` 固定效应、`[协方差组件数,t,t]` 的 `theta`、每个样本 `[t,t]` 精度块以及拟合状态。`fit_method` 记录普通 REML、严格 AI-REML 或显式因子 REML；`residual_covariance_singular` 记录残差组件边界。`sample_ids` 与基因型行必须严格对齐。模型文件含个体 ID 和残差，属于研究数据。

`model.score_covariance(genotype)` 接收已经统一 minor 方向、插补完成的 `[n,m]` dosage。它返回长度 `t*m` 的 score 和 `[t*m,t*m]` covariance，排列为第一表型的全部变异、第二表型的全部变异，依此类推。使用原始 GDS 时，应先按所有模型的 union 样本确定 minor 方向，再为该联合模型的共同样本重新计算 MAF 和插补。

`multi_staar_test` 返回 `num_variant`、`cMAC`、六组基础检验与各注释检验、六个 STAAR-S/B/A 合并值、`ACAT-O` 和 `STAAR-O`。列名与单表型函数一致；统计量使用表型间协方差。

## 参数

| 参数 | 含义 |
|---|---|
| `phenotypes` | 有限值的 `[n,t]` 共同完整案例矩阵。 |
| `covariates=None` | `[n,p]` 固定效应设计；默认截距。 |
| `sample_ids=None` | 与行对齐的唯一 ID；直接调用省略时生成行号。 |
| `kinship_diagonal=None` | 与行对齐的真实 GRM 对角；传入后拟合残差和遗传两个表型协方差组件。省略表示明确选择普通模型。 |
| `edge_rows/edge_cols/edge_values=None` | GRM 边输入；当前联合实现拒绝非零边，不忽略它们。 |
| `device="cuda"` | 运算设备；`"cpu"` 用于本地数值检查。 |
| `apply_rint=False` | 在共同样本上每列平均 ties 排序，使用 `(rank-3/8)/(n+1/4)` 和 R AS241 正态分位数。 |
| `maxiter=500`、`tol=1e-5` | 对角 GRM 联合 AI-REML 的迭代上限和原版相对参数停止规则。 |
| `robust=False` | 默认使用原版 AI 更新，并在不能表示的奇异残差协方差边界报错。显式 `True` 改用 PyTorch 协方差因子 REML，确保两个完整协方差组件为 PSD，以梯度驻点判断收敛，并使用总协方差的 `P*y`；这是独立的边界修复模式，不声称等于原版失败时的数值输出。 |
| `score/covariance` | 联合 score 与 trait-major covariance。 |
| `maf/mac` | 每个变异的共同样本 MAF 和原版 `round(MAF*2*n)` MAC；MAC 控制 ACAT-V 的 ultra-rare 分组。 |
| `annotations=None`、`names=None` | `[m,q]` PHRED 注释及列名；默认只按 MAF 加权。 |
| `n_pheno=None` | 联合表型数；默认由 score 长度除以变异数推断。 |
| `mac_threshold=10` | `MAC<=10` 的变异先做联合 burden，再并入 ACAT-V。 |
| `rare_maf_cutoff=.01` | 仅保留 `0<MAF<cutoff`。 |
| `rv_num_cutoff=2`、`rv_num_cutoff_max=1000000000` | 纳入变异数须至少达到下限且严格小于上限。 |
| `cmac=None` | 可传入插补后 `sum(G_rare)`，保持原版 cMAC；默认使用 MAC 总和。 |
| `acat_calibration="chi2"` | 联合 MultiSTAAR 使用自由度为表型数的卡方检验。 |

普通模型的残差协方差为 `T=E'E/(n-p)`，联合 score 为 `vec(G'E T^-1)`，covariance 为 `T^-1 ⊗ G'(I-X(X'X)^-1X')G`。对角 GRM 模型按样本计算 `Sigma_i=Te+K_ii*Tk`，AI 更新同时估计两个组件的所有表型方差和协方差。每个关联块只形成 `(t*m)^2` 的统计协方差，避免形成 `(t*n)^2` 的稠密投影。

## Python 调用

```python
import numpy as np
import torch
from staar_phewas.multi import fit_joint_gaussian_null, multi_staar_test

# 私有文件：所有数组均按共同完整案例的样本顺序排列。
aligned_data = np.load("joint_inputs.npz", allow_pickle=False)
null_model = fit_joint_gaussian_null(
    phenotypes=aligned_data["phenotypes"],       # [n,t]，连续表型
    covariates=aligned_data["covariates"],       # [n,p]，包含截距
    sample_ids=aligned_data["sample_ids"],
    kinship_diagonal=aligned_data["grm_diagonal"],
    device="cuda", apply_rint=True,
)
score, score_covariance = null_model.score_covariance(aligned_data["genotype"])
maf = torch.as_tensor(aligned_data["maf"], dtype=torch.float64, device="cuda")
association_results = multi_staar_test(
    score=score, covariance=score_covariance, maf=maf,
    mac=torch.round(maf * 2 * null_model.n),
    annotations=aligned_data["annotation_phred"],
    names=["CADD", "aPC.Conservation"],
    cmac=float(aligned_data["genotype"].sum()),
)
null_model.save("joint_null.pt")  # 含个体数据，只保存到私有运行目录。
```

明确不使用 GRM 的普通模型调用时省略 `kinship_diagonal`。已有 GRM 时，不能为规避拟合失败而自动删除它。二值表型、重复测量、多 GRM 和非零跨样本 GRM 边不属于本函数支持的联合 Gaussian 输入。

## 命令行

关联任务使用通用入口 `staar-phewas-torch analysis.json --device cuda`。联合模型以共同完整案例的二维 `y_raw[n,t]` 准备文件作为一个 phenotype 配置项；它与多个独立单表型配置项有不同统计含义。配置必须明确写 `joint_mode`：`ordinary` 表示省略 GRM，`strict` 保留原版对角 GRM AI 与边界失败，`robust` 显式采用独立因子 REML。加载已拟合 NPZ 缓存时也必须指定与缓存一致的模式，不会自动切换。

```json
{
  "name": "joint_imaging",
  "input": "joint_inputs.npz",
  "joint_mode": "strict",
  "transform": "rint",
  "phenotype_names": ["phenotype_one", "phenotype_two"],
  "sample_id_rule": "exact",
  "save_model": "joint_null.npz",
  "output_null": "joint_null.Rdata"
}
```

准备 NPZ 还包含 `ids[n]`，可含 `covariates[n,p]`、`sample_indices[n]`、`grm_diagonal[n]` 及稀疏边字段，所有数组按共同样本顺序。输入缺失须先去掉；RINT 只在共同样本上逐列排序。`sample_id_rule` 可以是 `exact`、`last_underscore_token` 或 `auto`。首次绑定后缓存实际 GDS ID，后续染色体按 ID 重新定位和验证，保留模型顺序，不复用未经检查的物理行号。

联合 native null 写出真正 `glmmkin.multi`：保留 covariance、scaled residual、稀疏 trait-major `Sigma_i` 与 `Sigma_iX`；`null_layout="multistaar"` 保留原 MultiSTAAR wrapper 字段，默认 `phewas` 追加 pipeline 的 flags。写出只使用 Python XDR 序列化。每个模型的单变异输出包含联合 score 向量和自由度为表型数的卡方 P，没有单一效应估计。

命令行配置与输出结构见 [pipeline 说明](pipeline.md)。直接检查联合模型代数可以运行 `python -m pytest tests/test_multi.py`。

## 原 R 调用与版本问题

参考为 [MultiSTAAR 0.9.7.1 固定源码](https://github.com/xihaoli/MultiSTAAR/tree/c372e135d88d5537c43af2d0f3e935f47cafd11c)，原 R 对照使用作者推荐环境中的 MultiSTAAR 0.9.7.1、GMMAT 1.3.2。真实参考调用为：

```r
null_model <- MultiSTAAR::fit_null_glmmkin_multi(
  cbind(phenotype_one, phenotype_two) ~ 1,
  data=phenotype_data, kins=kinship_matrix, id="sample_id"
)
association_results <- MultiSTAAR::MultiSTAAR(
  genotype, null_model, annotation_phred
)
```

所参考 PheWAS 聚合集合检验的联合分支调用 `MultiSTAAR_sp`，但该固定 MultiSTAAR 包没有导出或实现这个函数。这里复现实际存在的 `MultiSTAAR` 和 `MultiSTAAR_O_SMMAT_sparse` 算法，真实对照直接调用原函数，不构造同名 R 替代函数。因此普通联合集合检验核心的验证不能表述为原 PheWAS 联合集合 wrapper 已直接运行成功。

`Individual_Analysis_PheWAS` 的联合单变异分支独立调用原 `Individual_Score_Test_sp_multi`，可以直接运行。它按 `U' Cov^-1 U` 计算自由度为表型数的卡方检验；原代码在 covariance 行列式恰好为零时返回 P=1。GPU 实现保持该规则，并在 log 空间保留极小尾概率，输出 `pvalue`、`pvalue_log10` 和 `Score1` 至 `Score<t>`。

真实两表型和三表型的对角 GRM 原 R 零模型均在 `glmmkin.multi.ai` 的边界更新处报 `NAs are not allowed in subscripted assignments`，尚无该原版分支的完整关联输出。默认严格模式保留失败；显式边界修复模式单独标注。

## 真实精度与运行记录

2026-10-04 使用原 annotated GDS 的真实变异和三个真实连续脑影像表型，共同完整案例分别为两表型 42,418 人、三表型 42,411 人。按各自样本重新计算 MAF 和均值插补后，对 OR11H1 pLoF（3 RV）、COMT pLoF（5 RV）、COMT missense（74 RV）逐一比较原 R MultiSTAAR 与 float64 A100 GPU。每个集合均核对完整 40 列，含四列 PHRED 注释及各合并检验。

| 联合模型 | 普通零模型 T 最大绝对差 | 三集合最大 P 绝对差 | 三集合 STAAR-O 最大绝对差 | GPU 观测峰值分配 |
|---|---:|---:|---:|---:|
| 两表型 | 7.44e-15 | 1.14e-7 | 4.31e-9 | 0.221 GiB |
| 三表型 | 5.55e-16 | 4.71e-12 | 6.49e-13 | 0.456 GiB |

两表型最大差来自 COMT missense 的 SKAT 鞍点近似；另外两个集合最大差不超过 3.7e-15。两表型 COMT missense 的该列尚未通过本项目 `1e-10 + 1e-7*abs(R)` 严格阈值，不能据此声称联合集合全部严格一致。三表型高相关导致精度矩阵与 scaled residual 最大绝对差分别为 4.43e-11 和 5.67e-11，三个集合的全部 40 列通过严格阈值。

| 步骤 | 两表型原 R / GPU 秒 | 三表型原 R / GPU 秒 |
|---|---:|---:|
| 普通联合零模型 | 1.014 / 3.461 | 1.069 / 0.028 |
| OR11H1 pLoF 关联 | 0.686 / 0.191 | 0.591 / 0.206 |
| COMT pLoF 关联 | 0.640 / 0.230 | 0.588 / 0.243 |
| COMT missense 关联 | 0.961 / 0.383 | 0.674 / 0.396 |

GPU 关联计时含 score/covariance 和全部检验，使用同步后的墙钟；原 R 关联计时是完整 `MultiSTAAR` 调用。GPU 两表型零模型含首次 CUDA/lstsq 初始化，三表型在同一进程随后运行。计时不含 GDS 读取、完整案例准备和结果文件写入，是共享 GPU 的运行记录，不构成全基因组端到端加速结论。

当前版本新增联合 ordinary Gaussian 核心与对角 GRM AI-REML、独立完整矩阵代数检查和上述真实 ordinary R 对照；尚无对角 GRM 原版成功数值基线。八项联合单元检查是代数与尾概率验证，不替代真实 benchmark。

真实 CLI 同进程运行两个联合模型，完成 OR11H1 pLoF、COMT missense 和 COMT 区域单变异三个 job，含 GDS 读取、null 拟合和 native Rdata 保存，观测端到端 130.289 秒，GPU 峰值分配 0.461 GiB。这个入口记录没有对应原 R 端到端计时，不作为加速比。

普通联合 null 的 native Rdata 已在真正原 R 中回读：两表型和三表型各 18 个字段的名称、顺序、每字段 class、dimnames 和 call 全部与原 null 一致；`Sigma_i` 为 `dgTMatrix`、`Sigma_iX` 为 `dgCMatrix`，`validObject` 均通过。原 MultiSTAAR 直接使用 Python 保存的 null 完成六个真实集合统计。数值仍受上表中零模型和关联误差约束。

联合单变异另有真正 PheWAS wrapper 对照：在 COMT 的同一 1,000 bp 区域，两个 ordinary 联合模型各纳入 11 个变异，样本数分别为 42,418 和 42,411。原 `Individual_Analysis_PheWAS` 与 GPU 的 209 个数值单元全部通过严格阈值；Score、P 和 log10(P) 最大绝对差分别为 `1.63e-10`、`1.51e-11`、`1.35e-11`。正式 Rdata 的对象名、类型、属性、因子水平、行列名均与原输出一致，原 R 回读比较无结构或数值失败。原 wrapper 用时 8.596 秒，GPU 关联 job 用时 13.776 秒，含模型加载与 native 文件保存共 19.389 秒，峰值分配 36.29 MiB；这些是同区域运行记录。

显式 `robust=True` 的因子 REML 在相同真实两表型、三表型样本上分别用 194、443 次迭代达到梯度收敛，零模型运行 8.742、8.130 秒，GPU 峰值分配为 0.229、0.466 GiB。优化先作可逆 float64 表型白化，最终协方差变回原表型空间。将这些已拟合的总协方差固定后，真正原 R MultiSTAAR 与 GPU 六个关联集合的 40 列最大 P 差分别为 5.62e-12、1.08e-13。这个对照验证关联统计，不验证原 GMMAT 零模型优化；原 GMMAT 在这两个输入上仍然报错。

## 原实现与参考文献

- [MultiSTAAR R 与 C++ 源码](https://github.com/xihaoli/MultiSTAAR/tree/c372e135d88d5537c43af2d0f3e935f47cafd11c)，GPL-3。
- [GMMAT 原实现](https://github.com/hanchenphd/GMMAT)，联合 AI-REML 参考 `glmmkin.multi.ai`；真实 oracle 的包版本为 1.3.2。
- Li X, Chen H, et al. [MultiSTAAR 多表型稀有变异方法](https://doi.org/10.1101/2023.10.30.564764)。
- Li X, Li Z, et al. [STAAR](https://doi.org/10.1038/s41588-020-0676-4), Nature Genetics, 2020。
- Liu Y, et al. [ACAT](https://doi.org/10.1016/j.ajhg.2019.01.002), American Journal of Human Genetics, 2019。
- Chen H, et al. [SMMAT](https://doi.org/10.1016/j.ajhg.2018.12.012), American Journal of Human Genetics, 2019。
