# Torchstaar 原生输出的独立验证

验证器读取同一任务的 reference 与 candidate 原生 R 文件，比较结构、非 P 数值和显著联合的 `−log10(P)` 误差，不重新计算关联、不运行 GPU。reference 可以是固定输入下的官方 R 输出，或明确标注的上一接受版本；两者的验证范围分别记录。生产流程与参数见[主指南](../docs/torchstaar.md)。

## 输入、参数与输出

输入是同一模型、样本、变异、QC、注释及有序目录下的 `.Rdata/.rds`；保持相同对象名和布局。输出 JSON 只报告匿名计数、失败数与最大误差。任务身份、文件 SHA 与输入绑定保存在私有验证记录。

```bash
# 原生类型、属性、顺序、标识与 NULL。
Rscript validation/compare_structure.R \
  private/reference/result.Rdata private/candidate/result.Rdata private/structure.json
# 已有零模型的结构及非 P 数值。
Rscript validation/compare_nonp.R \
  private/reference/obj_nullmodel.Rdata private/candidate/obj_nullmodel.Rdata private/null.json
# 严格数值诊断；关联非 P 诊断与显著联合精度分别记录。
Rscript validation/compare_native_outputs.R \
  private/reference/result.Rdata private/candidate/result.Rdata
```

| 位置参数 | 格式、默认值与意义 |
|---|---|
| `reference/candidate` | 前两项，必填同任务原生文件路径，顺序固定。Rdata 按保存顺序读具名对象，RDS 读单个对象。 |
| `report` | structure/nonp 第三项，必填私有 JSON 输出路径。 |
| native comparator `atol/rtol` | 可选第三/第四项，默认 `1e-10/1e-7`；double 诊断门槛为 `absdiff<=atol+rtol*abs(reference)`。 |

结构比较保持类型、class、维度、属性/名称顺序、factor levels、row.names、S4 slots、NA/NaN/Inf 与 NULL；整数与字符串精确比较。完整范围要核对计划中的每个作业、类别槽和输出文件，不能用单个文件代替覆盖证明。合法空输出先确认双方原结构与非 P 状态一致，再记录 P 验证不适用；未知缺失不能当作合法空结果。

零模型没有关联 P，仍须独立检查模型身份、样本/设计和非 P 状态。当前性能对照复用已有固定模型，不重新拟合或导出零模型。显著联合、下溢和 native log 的细节见[logP 验证](README_logp.md)。R、Matrix 和 jsonlite 是独立验证依赖，生产运行不调用它们。

## 当前记录

0.4.0 的真实全量验证覆盖 4 份 chr21 Single 文件、1,065,735 行。与上一接受的 TF32 输出相比，结构、顺序、factor 属性、row names 和 AF/MAF/N 一致；55,377 个显著联合项的最大 log 误差为 `7.0916163e-6`。官方 R 另有同模型有界对照：28 个 P、2 个显著项，最大显著误差 `2.3897789e-6`。本版未重算全染色体 R，也未在本次 Single 测量中重测 gene-based。

计时范围与原实现见[真实验证](../docs/torchstaar.md#真实验证与计时范围)、[匿名汇总](../benchmarks/torchstaar_single_chr21_2026-10-09.json)和[参考](../docs/torchstaar.md#参考)。验证器未随本次文档整理更改算法。
