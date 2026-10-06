# TF32 与 P 值精度

核对后的原 R/STAAR 使用 double / FP64。当前候选采用相同公式的原生 TF32 矩阵乘法、FP32 累加和输出；关联 Score 和协方差不要求逼近 FP64；零模型文件仍保留原严格数值验收。TF32 的低位舍入是这一计算方式的性质，多分量重建已经退役。

关联数值验收使用原 R 或候选任一 `P<0.05` 的联合范围，要求每个值的 `-log10(P)` 绝对差不超过 `0.001`。全部 P 仍须有效、完整、可比较，并通过原生 log 一致性检查；非显著 P 的误差完整保存作诊断。文件名、类型、列名、维度、属性、顺序、标识、factor 与 NULL 保持原格式。零模型仍需通过原严格数值比较。正式输出序列化为 R double，极小概率采用原有稳定尾部公式和 native log 字段。缺失、无穷、越界、无有效 log 的下溢不能算通过，见 [验收规则](../validation/README_logp.md)。

生产 dense 矩阵路径不进行 FP64 分量重建或 CPU 谱匹配。GPU 的原生 TF32 核记录实际 MMA PTX 与执行次数；小向量 FP32 运算单独计数。FP16/BF16 未使用。旧 FP64 低级控制保留用于历史对照，不作为生产回退。运行配置、每项输入格式和报告见 [TF32 pipeline](tf32_pipeline.md)。

数值缩放只用于改善表示范围，不增加 TF32 有效位数。可将 Score 和协方差按 `u_scaled = s*u`、`V_scaled = s²*V` 同步变换，其中 `s=2^k`；单变异和 Burden 的统计量保持不变，SKAT 的 Q 与特征值共同缩放。在该 Score 变换中，计算得到的 `β_scaled=β/s`、`SE_scaled=SE/s` 都要乘 s 恢复原单位；原特征值绝对阈值也必须按原单位判断。不能先把 P 放大后直接输出，已经下溢到零的概率也不能靠事后乘法恢复。当前 Single 使用原正态尾的 log 表达，避免先计算极小 P 再取对数；gene 尾概率保留局部概率向量的宽范围表示，矩阵仍为 TF32/FP32。没有自动缩放用户数据或新缩放参数。

原 Saddle 已将 Q 和特征值除以最大特征值。因此共同用 2 的幂放大这两项会在归一化时抵消，不改变近均值求根的条件。真实定位中最大超限项 P 约 0.43，归一化 Q 接近谱均值；它不能解释为小 P 下溢。缩放与稳定计算的效果必须用相同输入的原公式分别验证，不能为了匹配参考 P 选择特定分支。

本版真实 chr21 的 795 项任务与 19 份原生文件已验收：478,082 个 P 有效、可比较，显著联合范围的 24,713 个 P 无超限，最大 logP 差为 `0.0003579714`，原生结构与严格零模型通过。进程墙钟 `429.097 s`，300 秒目标尚未达到。先前 B/C2/E 的严格零模型比较曾有 332 个 residuals 单元格超限；本版按原输入 cache dtype 序列化 `phenotype-fitted_values` 修复，关联状态仍为 TF32/FP32，旧报告不改写。旧三个集合探针、0.2.0 FP64 基线与退役分量版保留为历史，见 [benchmark](benchmark.md)和[TF32 benchmark](tf32_benchmark.md)。

参考：[PyTorch TF32 文档](https://docs.pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices)、[STAAR 原代码](https://github.com/li-lab-genetics/STAAR)、[STAARpipelinePheWAS](https://github.com/li-lab-genetics/STAARpipelinePheWAS)。
