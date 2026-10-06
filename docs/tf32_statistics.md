# 原生 TF32 统计计算

输入是单个 mask 的 Score `u`（M）、对称协方差 `V`（M×M）、`maf`（M）、`mac`（M）、注释 phred 矩阵（M×A）及按原 mask 算得的 `cmac`。变异与注释顺序必须对应；频率、singleton、极罕见变异及 cMAC 定义沿用 STAAR。输出是 `num_variant`、`cMAC` 及原命名的 SKAT、Burden、ACAT-V、STAAR-S/B/A/O P 值，随后由 pipeline 写成原 R 文件结构。

Score、协方差和加权矩阵采用原生 TF32 / FP32；元素变换、归约和完整特征值计算在 GPU 批量执行。原 Saddle 求根及回退、CCT 小 P 分支和 ACAT-V 的极罕见 Burden 规则保持。不使用分量乘法重建、CPU 精度匹配谱或 CPU libm 权重匹配。极小概率的表达需要保留原双精度输出范围；必要的标量尾部稳定运算服务于 P 值，不重做 FP64 关联矩阵。

```python
from staar_phewas.statistics import staar_test

# score/covariance 的计算和变异顺序已与该 mask 绑定。
staar_result = staar_test(
    score, covariance, maf, mac, annotations=annotation_phred,
    names=annotation_names, cmac=mask_cmac,
    matmul_mode="tf32", weight_batch_optimization=True,
    output_batch_optimization=True, cct_validation_optimization=True,
)
```

详细参数、原 R `STAAR()` 调用和引用见 [statistics](statistics.md)；完整 pipeline 的 Python/CLI 调用见 [TF32 pipeline](tf32_pipeline.md)。低级函数接受显式 FP64 历史对照，其结果不能替代正式原 R 文件。

尾部批处理将布尔检查合并为小张量传回 CPU，正常非零路径减少五次显式同步；原求根中点、终止宽度、完整谱、阈值和分支保持。208 个真实 SKAT 尾部 P 与冻结旧实现逐位一致，这个差分结果不改变此前原 R 超限结论；共享 GPU 条件下未报告尾部加速比例。

`output_batch_optimization` 和 `cct_validation_optimization` 是低级函数的布尔参数，默认 True；设为 False 可分别保留逐项输出和逐项输入验证对照。前者在所有原 CCT 调用完成后一次复制最终 P，W=26 时从 86 次输出复制降为 1 次。后者把六项合法性 flags 一次复制，按原优先级检查，保留最后的 NaN 检查；这里只统计这两个边界，不包含谱、求根和 cMAC 的同步。

2026-10-05 真实 coding 的两次有效检验：最终 P 输出复制 172→2 次，全部字段相对同版逐项对照及旧 native 逐位一致；178 个 P 对独立原 R 的最大 logP 差为 `0.000158636111464494`，原生结构通过。

2026-10-06 开发版把 SKAT 权重的精确比例判定按行批量执行。输入是转置后的 `W×M` FP32 权重；设备上保留每行首个非零位置，使用两个 FP32 数的精确 FP64 乘积判定比例关系，不设容差，不重做 FP64 Score/协方差。判定表 `W×W` 一次传回 CPU，保持原代表列及结果列顺序。原 FP32 加权矩阵、完整特征值、比例平方与尾概率规则保持。

临时比较矩阵默认至多 64 MiB，并按配置的进程显存上限、当前空闲显存和原安全余量缩小行批量；容纳不下一行时明确报错。没有新增用户参数。`statistics_execution_metadata()` 的 `native_relation_row_batches` 记录实际行批量数，`native_relation_bulk_d2h_calls` 仅记录关系表回传，`native_relation_max_scratch_bytes` 是比较矩阵的保守临时空间估计；这些指标不包含合法性检查、求谱或概率回传。

四组真实固定输入的完整谱、重新计算的 U/V 及 248 个 P 相对前版逐位一致。两组较大输入的关系判定主机墙钟中位数约为 3.15/0.87 ms，属于共享 GPU 上的阶段观察。Single/coding/noncoding/ncRNA 五任务的四份文件、全部 15 个原 mask 槽位和 377 个 P 另通过原 R 对照，最大 logP 差为 0.000158636；完整 CPU 回归 724 通过、2 跳过、36 子测试通过。小流程总墙钟约 61 秒，没有显示明显端到端改善，见[匿名记录](../benchmarks/staar_tf32_followup_2026-10-06.json)。

本版完整 chr21 已完成 795 项任务：19 份原生文件结构和严格零模型通过，478,082 个 P 全部有效、可比较。按原 R 或本版任一 P<0.05 的联合范围检查 24,713 个 P，最大 logP 差为 `0.0003579714`，无超限；全部 P 的误差继续作为诊断保存。进程墙钟 `429.097 s`，300 秒目标未达。上文局部记录属于早期开发轮次，独立固定矩阵探针的非显著近均值误差不替换本版完整门槛。旧权重批处理的固定 u/V 实验属于退役分量版，不作为本版速度证据，见 [TF32 benchmark](tf32_benchmark.md)。

参考：[STAAR 原检验](https://github.com/li-lab-genetics/STAAR/blob/master/src/STAAR_O_SMMAT_sparse.cpp)、[Saddle](https://github.com/li-lab-genetics/STAAR/blob/master/src/Saddle.cpp)、[CCT](https://github.com/li-lab-genetics/STAAR/blob/master/src/CCT_pval.cpp)、[权重](https://github.com/li-lab-genetics/STAAR/blob/master/R/STAAR.R)。文献见 [reference inventory](base_reference_inventory.md)。
