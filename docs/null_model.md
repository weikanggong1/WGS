# Gaussian 零模型

`fit_gaussian_null` 用PyTorch拟合固定效应与稀疏亲缘矩阵的 Gaussian 零模型。输入样本已经对齐，输出可直接交给关联 pipeline，并保存为原生 `obj_nullmodel.Rdata`。计算采用亲缘连通块分解；无亲缘边的样本保留各自的对角项，不将矩阵替换为单位阵。

## Python 调用、输入和输出

```python
import numpy as np
from torchstaar.null_model import fit_gaussian_null
from torchstaar.rint import rank_inverse_normal_tensor
from torchstaar.compat import write_gaussian_null

aligned_data = np.load("private/aligned_trait.npz", allow_pickle=False)
# 在最终有效样本上取平均并列秩，采用 Blom offset，再计算正态分位数。
transformed_trait = rank_inverse_normal_tensor(aligned_data["y_raw"], device="cuda")
null_model = fit_gaussian_null(
    transformed_trait,
    sample_ids=aligned_data["ids"],
    kinship_diagonal=aligned_data["grm_diagonal"],
    edge_rows=aligned_data["grm_edge_row"],
    edge_cols=aligned_data["grm_edge_col"],
    edge_values=aligned_data["grm_edge_value"],
    device="cuda", tol=1e-5, maxiter=500, matmul_mode="tf32",
)
# 原始 GDS 字符串 ID 与 transformed_trait 严格同序。
write_gaussian_null("runs/obj_nullmodel.Rdata", null_model,
                    original_sample_ids=aligned_data["gds_sample_ids"])
```

- `phenotype`：长度 N 的有限连续值。函数本身不变换表型、不删除缺失行。
- `sample_ids`：N 个唯一字符串，对应表型顺序。省略时使用零起始行号字符串。
- `covariates`：N×P 的有限满秩设计矩阵，必须显式包括所需截距。省略时仅拟合截距。
- `kinship_diagonal`：长度 N 的非负有限 GRM 对角。省略时拟合普通 Gaussian 模型。
- `edge_rows`、`edge_cols`、`edge_values`：相同长度的稀疏非对角项；索引从零开始。每条边可出现一次或对称两次，相同边的数值必须一致。输入稀疏矩阵不再次阈值化。
- `device`：`cpu`、`cuda` 或具体 CUDA 设备，默认 `cpu`。
- `tol`、`maxiter`：收敛阈值与迭代上限，默认 `1e-5`、`500`。
- `max_block_size`：最大亲缘连通块规模，默认 2048；超出时明确报错。
- `trace_callback`：可选开发诊断函数，接收每轮状态。默认不输出；状态包含个体数据，应存放在私有目录。

返回 `GaussianNullModel`：`coefficients` 为固定效应系数，`theta` 为最终方差分量，`scaled_residuals` 为关联计算使用的缩放残差；`precision_theta`、`inverse_variance`、`precision_x` 与 `fixed_effect_covariance` 保留原 GMMAT 停止时使用的精度状态。`phenotype`、`working_phenotype`、`fitted_values` 用于构造原生 R 对象；`iterations` 和 `converged` 记录拟合状态。关联时调用 `score_covariance(genotype)`，得到 score 向量和变异协方差矩阵。

`rank_inverse_normal_tensor(values, device=...)` 接受 N 向量或 N×T 矩阵，分别按列处理平均并列秩，使用 `(rank−3/8)/(N+1/4)` 与 R AS241 正态分位数。输出为同形状 float64 Tensor。必须先筛选实际分析样本。Tensor 分母保留原除法顺序，避免标量倒数优化产生末位差异。

## 命令行与原 R

CLI 在 `phenotypes[].input` 指定对齐 NPZ，在 `transform` 指定 `none` 或 `rint`，在 `output_null` 指定正式文件：

```bash
torchstaar private/analysis.json --device cuda --report runs/summary.json
```

原 R 对应：

```r
library(STAARpipeline)
aligned_phenotypes$transformed_trait <- qnorm(
    (rank(aligned_phenotypes$raw_trait, ties.method="average") - 3/8) /
    (nrow(aligned_phenotypes) + 1/4)
)
obj_nullmodel <- fit_nullmodel(
    transformed_trait ~ 1, data=aligned_phenotypes,
    kins=sparse_relationship_matrix, id="sample_id",
    family=gaussian(), method.optim="AI", tol=1e-5, maxiter=500
)
save(obj_nullmodel, file="runs/obj_nullmodel.Rdata")
```

原生单性状对象保存 `glmmkin` 类和原 20 个字段，包含 `dsCMatrix`/`dgeMatrix`、模型矩阵的 `assign`、原调用表达式与原始 GDS `id_include`。NPZ 是内部拟合缓存；正式 R 文件及验证方式见 [原生输出](r_native_output.md)。

## 算法、版本与验证

混合模型保留 GMMAT 的 EM 初始化、AI REML 更新、非负边界步长、边界重拟合及停止条件。原返回值保留停止前一轮用于关联的 precision，并按最终 dispersion 缩放残差；不能用最终方差重新计算 precision 后当作相同对象。原 Matrix 的对角 Cholesky 求逆采用 `(1/sqrt(variance))²`，本实现保留此顺序。显式 FP64 控制中，涉及 R 基础 `sum` 的项保留补偿求和；生产模式显式使用 TF32/FP32。

本版 Single 对照加载固定已有模型，未重新进行拟合 benchmark；实际范围见 [主指南](torchstaar.md#真实验证与计时范围)。低级fit_gaussian_null默认FP64控制；生产应显式指定TF32，输入/停止时模型状态和原序列化规则保持。

参照 [GMMAT 1.3.2](https://github.com/cran/GMMAT/tree/ef49eec8d0d95951a48f7055a77321077dbc8c13)、[STAARpipeline 0.9.9](https://github.com/yuxinyuanqt/STAARpipeline/tree/fbce778bf14cc4f9e892989a194c64bae2670311) 和 [R 3.6 AS241](https://github.com/wch/r-source/blob/R-3-6-branch/src/nmath/qnorm.c)。Chen H et al., *American Journal of Human Genetics* (2019), [DOI](https://doi.org/10.1016/j.ajhg.2018.12.012)；Wichura MJ, *Applied Statistics* (1988), [DOI](https://doi.org/10.2307/2347330)。
