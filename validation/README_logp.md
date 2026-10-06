# Torchstaar的P有效性与显著联合精度验证

## 功能与门槛

验证工具递归读取原R与Torchstaar同任务的Rdata/RDS，比较原生结构、全部P有效性/覆盖、native log一致性，以及显著联合范围的 `-log10(P)` 误差；不重新分析数据、不运行GPU。

主范围 `S={i:P_R,i<0.05 或 P_Torch,i<0.05}`，要求每项 `|-log10(P_R,i)+log10(P_Torch,i)|<=0.001`。任一侧跨0.05的值仍参加误差检查，boundary crossing另计；不因跨界本身更改门槛。全部P必须有效且可比较，不能只验证显著项；19份原结构与strict null同为必过条件。

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

## 完整文件与strict null

完整验收必须绑定同一科学源码、原目录/顺序、795任务、全部15mask、18关联+1null文件、原reference与candidate SHA及输入前后证明。15文件包含478,082 P，3文件原本无P；只有独立原预检绑定双方均0P，并且结构与非P严格比较通过时，空文件P门槛才记not applicable。未知缺失不能当合法空文件；保留原空P validator raw false。

null没有关联P，必须通过结构和原strict nonP数值阈值 `absdiff<=1e-10+1e-7*abs(reference)`；关联非P的严格误差只作诊断。下列CPU schema/特殊值合同不替代真实benchmark：

```bash
Rscript tests/test_structure_validation.R
Rscript tests/test_logp_validation.R
```

## 当前真实结果、计时与来源

当前Torchstaar公开科学运行完成795项/19文件：全478,082 P有效/可比较；显著联合24,713、0超限、最大logP差0.000357971421594216、boundary crossing0。strict null341,221数值单元格/0超限，最大绝对/相对差1.47138834449834e-7 / 5.95656454671069e-8。全部P仍有49个诊断超限，最大0.0708922025240736，关联strict nonP诊断false；它们不替代主科学验收。

关联进程墙钟296.196098秒，独立R比较127.953秒在外。共享节点没有独占证明，原组合速度/精度字段false保留；本轮观测wall<300与主科学门槛通过。已有cache/固定null、暖缓存与首次转存范围、近期F/G/H2/H3对照统一见 [主指南](../docs/torchstaar.md#5-最新真实精度与耗时)和 [匿名汇总](../benchmarks/torchstaar_chr21_2026-10-06.json)。

原R调用、固定版本与文献见 [主指南](../docs/torchstaar.md#4-原r对应调用)、[参考](../docs/torchstaar.md#7-原实现许可与参考文献)。R/Matrix/jsonlite仅用于独立验证，Python生产运行不调用它们。
