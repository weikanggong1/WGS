# Torchstaar的P有效性与显著联合精度验证

## 功能与门槛

验证工具递归读取原R与Torchstaar同任务的Rdata/RDS，比较原生结构、全部P有效性/覆盖、native log一致性，以及显著联合范围的 `-log10(P)` 误差；不重新分析数据、不运行GPU。

主范围 `S={i:P_reference,i<0.05 或 P_candidate,i<0.05}`，要求每项 `|-log10(P_reference,i)+log10(P_candidate,i)|<=0.001`。任一侧跨0.05的值仍参加误差检查，boundary crossing另计；不因跨界本身更改门槛。全部P必须有效且可比较，原生结构与固定模型身份单独核对；reference 是官方 R 还是上一接受版本须明确记录。

## 输入、识别和输出

| 输入/字段 | 格式与含义 |
|---|---|
| reference / candidate | 同一任务、相同原对象名/布局的Rdata或RDS；Rdata按保存顺序读具名对象列表，RDS读单个对象。 |
| P字段 | 按明确列名识别pvalue/Pvalue/P.value/p_value、ACAT-O/STAAR-O、SKAT/Burden/ACAT-V与STAAR-S/B/A两种Beta名称和注释后缀；不按值是否在[0,1]猜P。 |
| 原native log | Single `pvalue_log10`为正 `-log10(P)`，直接比较；优先pvalue_log10，否则pvalue_log为正 `-ln(P)`后除ln10。不是再次取log。 |
| target | 有限非负logP绝对差，默认0.001；完整验收保持该值。 |
| significance threshold | 当前显著联合脚本只接受0.05。 |
| 输出JSON | P数量、有效/缺失/无穷/越界/零值、raw P和logP误差、分位数、显著联合数量/超限/最大差与boundary crossing。空分层误差为null，不能作0误差。 |

原data.frame、具名vector、matrix和mixed list matrix均按同一递归规则识别，继承父P语义的向量继续视为P向量；未知统计字段须先核对规则和coverage。MAF/BETA/CHISQ/Score/Est不是P。结构规则独立检查类型、维度、属性/名称/顺序、factor/S4/row.names与NULL。

正P用double计算log，接近1用log1p(P-1)，不clamp。raw P=0单记underflow；双方都0也不自动通过。仅在原Single有有限非负native log、符合double最小次正规数阈值且原恢复/一致性规则通过时恢复比较，恢复数和原不可比较数分开报告。无可用native log的0、NA/NaN/Inf/越界或遗漏不能通过。

## R与命令行调用

```r
source("validation/compare_significant_logp.R")
compare_significant <- significant_logp_function("validation/compare_logp.R")
native_reader <- attr(compare_significant, "oracle")$read_logp_native
reference_objects <- native_reader("private/reference/result.Rdata")
candidate_objects <- native_reader("private/candidate/result.Rdata")
significant_report <- compare_significant(reference_objects, candidate_objects, target=0.001)
# 只读取匿名判定字段，不输出实际P/任务表。
print(significant_report$significant_logp_passed)
```

```bash
Rscript validation/compare_significant_logp.R \
  validation/compare_logp.R private/reference/result.Rdata \
  private/candidate/result.Rdata private/significant.json 0.001 0.05
# 全部P误差诊断；该脚本的logp_passed不等于主显著联合gate。
Rscript validation/compare_logp.R \
  private/reference/result.Rdata private/candidate/result.Rdata private/all-P.json 0.001
Rscript validation/compare_structure.R \
  private/reference/result.Rdata private/candidate/result.Rdata private/structure.json
Rscript validation/compare_nonp.R \
  private/reference/obj_nullmodel.Rdata private/candidate/obj_nullmodel.Rdata private/null.json
```

| 显著脚本位置参数 | 意义 |
|---|---|
| 1 | 原compare_logp.R路径，复用同次识别/有效性/native log恢复规则。 |
| 2、3 | 同任务reference/candidate原生文件，顺序固定。 |
| 4 | 匿名JSON报告路径。 |
| 5 | 可选target，默认0.001。 |
| 6 | 可选显著阈值，当前固定只接受0.05。 |

输出保留原全部P诊断，并增加 `all_P_contract_passed`、`significant_logp`、`significant_stored_pvalue_log10`、`significant_logp_passed`。CLI退出码按显著联合gate决定；原 `logp_passed` 仍是全部P诊断，可为false。两侧均非显著的差异也完整保留，不更改原报告来声称所有FP64值相等。

## 完整范围与固定零模型

完整验收绑定同一模型、样本/变异轴、QC、注释、原目录/顺序、实际作业计划、reference/candidate SHA 及输入前后证明。当前完整 Single 计划包含 4 个区间、4 份文件；gene-based 的完整验证需另核对全部 mask 槽及其 NULL。双方均无 P 且结构与非 P 状态通过时，空文件 P 门槛记不适用，保留原空 P validator 的 raw false。未知缺失不能当合法空文件。

零模型没有关联 P，仍检查结构、样本/设计与严格非 P 数值，默认阈值 `absdiff<=1e-10+1e-7*abs(reference)`；关联非 P 的严格误差只作诊断。复用已有模型时记录身份，不将新拟合加入对照。下列 CPU schema/特殊值契约不替代真实 benchmark：

```bash
Rscript tests/test_structure_validation.R
Rscript tests/test_logp_validation.R
```

## 当前真实结果、计时与来源

0.4.0 的完整 Single 输出包含 1,065,735 个有效、可比较 P。相对上一接受 TF32 输出，显著联合 55,377 项，最大 log 误差 `7.0916163e-6`，0 项跨越 0.05；4 份原生文件的键/顺序、类型、factor/row.names 与 AF/MAF/N 均一致。官方 R 的同模型有界对照另包含 28 个 P、2 个显著项，最大显著误差 `2.3897789e-6`。这两个参考范围分开报告。

本版 Single 作业墙钟 1125.923 秒，启动至文件输出 1183.551 秒；首次转存、新拟合与独立 R 验证另计。两次运行的启动/I/O 状态不同，阶段 host/stream 指标不能相加。细节见[主指南](../docs/torchstaar.md#真实验证与计时范围)和[匿名汇总](../benchmarks/torchstaar_single_chr21_2026-10-09.json)。

原 R 调用与文献见[命令行与原 R](../docs/torchstaar.md#命令行与原-r)、[参考](../docs/torchstaar.md#参考)。R/Matrix/jsonlite 仅用于独立验证，Python生产运行不调用它们。
