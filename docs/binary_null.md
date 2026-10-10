# 二项表型零模型与 SPA 接入

`fit_logistic_null` 用 float64 PyTorch 在指定设备执行普通二项 logistic IRLS；初始化、工作权重和收敛判据遵循 R 的 `glm.fit(..., family=binomial())`。`binary_prefitted_state` 将已拟合的原生状态接入 GPU 关联统计，保存完整投影和样本顺序。它不进行零模型拟合。当前没有原生二项混合模型拟合器；给定 GRM 时必须显式选择普通模型或传入已拟合的混合状态。

| 输入/参数 | 格式与作用 |
|---|---|
| `phenotype` | `[n]`、仅含 0/1，须同时包含两个类别。 |
| `covariates=None` | `[n,p]`、包含截距且满秩；省略时生成一列截距。 |
| `sample_ids=None` | `[n]` 唯一个体标识，保持输入顺序；正式分析必须提供。 |
| `device="cuda"` | 拟合与统计设备；全部计算使用 float64。 |
| `use_spa=True` | 在 pipeline 中选择二项 SPA burden 和单变异检验。 |
| `maxiter=100`, `tol=1e-8` | IRLS 迭代上限和相对 deviance 收敛阈值；不收敛、饱和或分离时抛错。 |

输出 `BinaryNullModel` 包含 `[p]` 系数、`[n]` 预测概率和残差、`[p,p]` 固定效应协方差，以及 SPA 所需的 `xw`（`[p,n]`）和 `projection_left`（`[n,p]`）。普通模型分别使用 `X.T*mu*(1-mu)` 和 `X*inv(X.T*W*X)`。混合状态使用 `X.T*Sigma_i` 和 `X*cov`，还必须提供 `precision=Sigma_i` 与 `precision_covariates=Sigma_iX`。精度矩阵可为 torch sparse COO/CSR；不将样本方阵稠密展开。

```python
import numpy as np
from torchstaar.binary_null import fit_logistic_null
from torchstaar.io import save_null_model, load_null_model

# 私有准备文件：已去掉缺失，所有数组按同一完整案例顺序排列。
inputs = np.load("binary_inputs.npz", allow_pickle=False)
model = fit_logistic_null(
    phenotype=inputs["y_raw"],
    covariates=inputs["covariates"],
    sample_ids=inputs["ids"],
    device="cuda", use_spa=True,
)
save_null_model(model, "binary_null.npz")  # 含个体数据，保存到私有目录。
reloaded_model = load_null_model("binary_null.npz", device="cuda")
```

二分类 SPA 使用显式 FP64 对照 API，配置 `matmul_mode="fp64"`、`precision_control=true` 后使用 `torchstaar analysis.json --device cuda`；自动展开的完整染色体入口保持 Gaussian 限制。0.5.0 的已展开单表型配置及 [Torchstaar PheWAS](torchstaar_phewas.md) 还接受完整固定状态的 `use_spa=false` 二分类，在 `matmul_mode="tf32"` 下执行原 Score/协方差公式；不重新拟合状态。对角精度、带截距的非 SPA Single 用当前表型样本中心化计算同一方差公式，并保留有限固定状态的常数方向校正。独立与 PheWAS 共用此核心和有效列合批。SPA 不使用这条 TF32 路径。`phenotypes` 中每项采用 `family="binomial"`、`binary_mode="ordinary"`、`transform="none"`，或者用 `model` 加载完整 NPZ 状态。普通模式明确表示不使用准备文件内的 GRM。关联 Rdata 按实际检验保存原字段：非 SPA Single 为普通 Score/SE/beta/P 列，SPA 分支保存其原 SPA 字段；二项 null 的原生 Rdata 写出尚未实现，应保存完整 NPZ 缓存。原 R 验算缓存标记为 reference，默认不能用于正式 CLI；它需要显式 `validation_reference=true`，执行报告也保留该标记。

原软件的普通拟合对照命令是 `glm.fit(X, y, family=binomial())`；关联统计参考 [STAAR 二项 SPA 源码](https://github.com/yuxinyuanqt/STAAR/tree/4bbf77ba8a90894a434f5eb4473d540e172dad05) 和 [二项 SPA 文档](binary.md)。R 仅用于开发时对照，生产代码不调用 R。

0.4.0 的历史 benchmark 为连续单表型 Single，不覆盖二项零模型或 SPA。0.5.0 的完整非 SPA 固定状态对照记录见 [PheWAS 验证](torchstaar_phewas.md#验证与近期记录)；零模型重新拟合与 SPA 的精度和耗时仍需按各自范围验证。本文保留已有独立 API 的参数与输出约定；其完整二项流程需另行真实对照。

参考：[R stats::glm 官方源码](https://github.com/wch/r-source/blob/trunk/src/library/stats/R/glm.R)、[R binomial family](https://github.com/wch/r-source/blob/trunk/src/library/stats/R/family.R)、[Dey et al. SPA](https://doi.org/10.1016/j.ajhg.2017.05.014)。
