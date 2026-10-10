# 多个独立表型的接口

生产入口为 [`torchstaar_phewas.run_configuration`](torchstaar_phewas.md)。它接收多个已展开的单表型 Torchstaar 配置，复用同一已验证六状态缓存、CPU–GPU 传输和注释准备；每个表型保持自己的完整案例、频率、等位基因方向、零模型及原生输出，目标与分别运行单表型流程一致。不同样本数和 NaN 位置按各表型处理，不取共同完整案例，不插补缺失表型。

```python
import json
from pathlib import Path
from torchstaar_phewas import run_configuration

# cache_specs 来自当前真实源绑定、缓存物理轴和实时证明函数。
# 完整示例见 torchstaar_phewas.md，不能用常量证明跳过检查。
analyses = [json.loads(Path(filename).read_text()) for filename in (
    "private/trait_01_analysis.json", "private/trait_02_analysis.json"
)]
report = run_configuration(analyses, cache_specs=cache_specs, device="cuda:0")
```

命令行为 `torchstaar-phewas private/phewas.json --device cuda:0 --report private/report.json`。当前支持独立单 Gaussian 与完整 prefitted 非 SPA 二分类状态，使用原生 TF32/FP32。全部输入、参数、输出、mask 上限、实时证明、原 R 调用及本轮验证状态见[完整说明](torchstaar_phewas.md)。

## 与原低级接口的关系

| 接口 | 统计语义与用途 |
|---|---|
| `torchstaar_phewas.run_configuration` | 独立单表型 `base` 语义，各表型有自己的样本方向和输出文件，共享原始缓存读写与设备传输 |
| `torchstaar.pipeline.PheWASPipeline` 直接传多个模型 | 保留原低级 PheWAS 并集提取语义，不同模型共用该并集的 minor 方向；它不是本轮独立单表型等价性的验收入口 |
| [联合 MultiSTAAR](multi.md) | 一个模型联合估计相关表型及其协方差，统计假设与输出不同，不由独立 PheWAS 自动代替 |

0.4.0 单表型 benchmark 保留在[配置指南](torchstaar_configuration.md#真实验证与计时范围)，不能推导新 PheWAS 的时间或精度。已完成共享来源的 13 固定表型 chr21 四类计划共有 234 native 与 8,803,388 行，共享 6,091.267 s、独立合计 16,772.722 s，显著联合最大 logP 误差为 0。该配置使用 `variant_type=variant`，纳入 SNV 和 Indel、两组 Beta 权重，实际 PHRED 权重数为 0。0.7.0 基于 `dbda0dc` 合并后关闭 M 上限覆盖全部 mask；全部 mask 共享、保存的独立参考与长 mask 补充、完整回归和安装已通过。本版 13 个固定独立模型完成全部 mask，maximum_mask_variants=null，rv_num_cutoff_max 恢复原默认 10^9，跳过 0 个 mask；共 10,335 个表型作业，原生变异数（#SNV）列计数得到 39 个 M>5000 结果行，输出 234 native 和 8,803,427 行。共享入口 8,443.879 s、外层 driver 8,495.115 s。本轮复用保存的独立参考，共同 8,803,388 行、长 mask 对照补充 39 行、未覆盖 0 行；历史参考有 0 行未匹配。历史组合参考比较 9,143,830 个 P，不可比较 0 个（验收门控），显著联合 536,242 个，最大 logP 误差 1.76570982e-08，超标 0，阈值跨越 0；原生结构与非 P 诊断通过。allocated/reserved 峰值 14.430/19.811 GiB。CPU intra-op 请求/实际线程数为 8/8。历史有限 M 独立入口曾合计 16,772.722 s；其 CPU 线程数、源码及 mask 范围与当前运行不同，不据此计算本轮速度比。该时间来自既有缓存，OS page cache 和共享 GPU 负载未受控；首次转存、原 R 验算、回归、打包和 CSV 均另计。此前有限 M 候选的回归和有界原 R 对照保留在来源记录中，不替代全部 mask 验证。完整方法与范围见[当前指南](torchstaar_phewas.md#验证与近期记录)。

参考：[STAARpipelinePheWAS](https://github.com/li-lab-genetics/STAARpipelinePheWAS)、[STAARpipeline](https://github.com/li-lab-genetics/STAARpipeline)。
