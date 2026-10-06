# 保留原公式的 GPU 矩阵复用

本优化适用于同一连续表型、同一零模型和同一局部区域的多个 mask。输入是按零模型样本顺序排列的 `G`（N×M 基因型）、`r`（N scaled residuals）、`A=Sigma^{-1}X`（N×C）、`C=(X^T Sigma^{-1}X)^{-1}`（C×C）及零模型的稀疏亲缘结构。各 mask 的过滤、变异方向、均值插补和样本集合一致才共享结果。

## Score 与协方差

```text
u = G^T r
V = G^T (Sigma^{-1} G) - (A^T G)^T C (A^T G)
```

保留原 STAAR 稀疏混合模型的计算公式，不构造 N×N 的 P。亲缘块的旋转沿用零模型表示；对角精度可直接逐元素相乘。Single 只求 V 的对角线，避免构建整块 M×M 协方差。

同一区域先取全部 mask 的变异并集，计算一次 u/V。mask 的整数索引 `indices` 按原变异顺序从并集中取 `u[indices]` 和 `V[indices,indices]`。整条染色体不建立完整协方差。显存不足或并集产生过多不需要的跨 mask 协方差时分开计算；结果语义保持一致。

## 小矩阵的并集选择

优先沿用实际 rare 列数的 `M_union² <= sum(M_mask²)` 规则。对没有亲缘旋转块的单 native Gaussian 模型，当实际并集不超过 64 列时，再按当前后端的 route 和 tile 填充量检查计算成本；原显存保护继续独立执行。成本不使用 P 值，也不删除 mask。

协变量数大于 1 时，每个相应矩阵或向量产品的工作量均不得增加。只有截距时，协方差 TF32 MMA 填充工作量必须严格减少，各 FP32 向量产品不得增加；K=1 outer 的总元素及增量均限制为 4096。不同 route 的工作量独立判断，不按统一系数折算吞吐。报告中的 `geometry_small_tile_outer_extra_output_bytes` 是 outer 输出形状差值，不能当作完整工作区或实际显存分配。

一组真实 47 列并集、37/10 列两个 mask，Score/协方差调用由 2 次减为 1 次。该组 178 个 P 与冻结旧路线最大 logP 差为 0.000000702241，与实际原 R 最大差为 0.000158636；原生结构通过。峰值 GPU 分配约 166.51 MiB。两张 GPU 均有其他任务，尚无可信端到端加速比例，见[匿名记录](../benchmarks/staar_small_union_2026-10-05.json)。

## 权重一起计算

将原 Burden 权重按列排列为 `W_B`（M×K），一次得到：

```text
s = W_B^T u
d = colsum(W_B * (V W_B))
Q_B = s^2 / d
Q_SKAT = colsum((W_SKAT * u[:,None])^2)
```

SKAT 仍对每个权重的 `D_w V D_w` 求完整特征值，执行原 Saddle 和回退分支。完全相同权重的谱可复用；不以近似相似、截断特征值或其他检验替换。ACAT-V 复用原单变异 P；极罕见变异仍按原 Burden 合并规则。CCT 保留原小 P 稳定分支。

```python
import json
from pathlib import Path
from staar_phewas.chromosome import run_chromosome

analysis_configuration = json.loads(Path("private/full_chromosome.json").read_text())
analysis_configuration["matmul_mode"] = "tf32"
analysis_configuration["local_mask_reuse"] = True
analysis_configuration["weight_batch_optimization"] = True
analysis_report = run_chromosome(analysis_configuration, device="cuda")
```

上述配置的每项输入格式见 [TF32 pipeline](tf32_pipeline.md)。返回运行报告并写出原 STAAR 的所有 native 文件；没有新增位点或 gene 汇总列。命令行和原 R 对应调用见 [chromosome](chromosome.md)。

## 验证范围

新候选只以 `-log10(P)` 的最大绝对误差 <=0.001 做数值验收，并单独检查原生文件结构、行序、标识、NULL 与 mask 覆盖。完整每染色体墙钟目标 <=300 秒，包含所有输入准备与写出。新版本尚无完整真实 benchmark；已取消的分量重建实验不能作为原生 TF32 性能证据，见 [TF32 benchmark](tf32_benchmark.md)。

参考：[原 Score 实现](https://github.com/li-lab-genetics/STAAR/blob/master/src/Indiv_Score_Test_SMMAT_sparse.cpp)、[原关联检验](https://github.com/li-lab-genetics/STAAR/blob/master/src/STAAR_O_SMMAT_sparse.cpp)、[权重构造](https://github.com/li-lab-genetics/STAAR/blob/master/R/STAAR.R)。版本与文献见 [reference inventory](base_reference_inventory.md)。
