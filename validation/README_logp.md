# 小 P 值的独立原生输出对照

`compare_logp.R` 读取原软件 R 与 PyTorch 保存的 Rdata/RDS，对相同结构中的 P 值报告原始误差和 `|-log10(P_R)+log10(P_Torch)|`。它只做文件读回验证，不运行关联分析，也不调用 GPU。本版关联数值使用原 R 或候选任一 P<0.05 的联合范围逐值检查 logP 差 <=0.001；全部 P 的有效性、覆盖和原生 log 一致性仍必检。`compare_structure.R` 检查全部原生结构与标识；零模型文件及原合法空结果仍需通过原严格比较。非 P 关联浮点数只保留诊断，不要求逼近 FP64。`compare_logp.R` 本身保留全部 P 的原诊断语义，不能将其全部 P 的 pass 字段当作显著范围 gate。

## 输入和识别规则

两个输入文件应来自同一批任务，文件名、保存对象名、列表顺序、矩阵形状、属性和列名逐一对应。Rdata 读为按原保存顺序排列的具名对象列表，RDS 读为单个对象。支持 data.frame、具名列表、具名 double 数值向量、数值矩阵，以及 gene 输出采用的混合列表矩阵。具名 double 向量按元素名称识别统计字段；已经从父列表继承 P 语义的向量仍按完整 P 向量处理，不把观察名称误作字段名称。

P 列使用明确规则：`pvalue`、`Pvalue`、`P.value`、`p_value`、`ACAT-O`、`STAAR-O`，以及 `SKAT(1,1)` / `SKAT(1,25)`、`Burden`、`ACAT-V` 的相同权重名称和注释后缀，或 `STAAR-S/B/A` 的相同权重名称。不会按数值是否介于 0 和 1 判断 P，也不会把 `MAF`、`BETA`、`CHISQ`、`Score`、`Est` 识别为 P。未知检验列应先扩展识别规则并验证，不得假定已覆盖。

Single 输出已有 `pvalue_log10` 定义为 `-log10(P)`，在 P 小于 1 时为正。工具直接比较这个字段，并检查它与原始 P 的一致性。它不会对该字段再次取对数。具名 double 统计向量中的唯一 `pvalue` 与唯一优先 native log 也按相同规则配对；优先 `pvalue_log10`，否则使用 `pvalue_log`。重复标签不猜测配对，原属性和名称顺序检查继续执行。

## 调用

```r
source("validation/compare_logp.R")
reference_objects <- read_logp_native("reference.Rdata") # 原 R 保存的文件
candidate_objects <- read_logp_native("candidate.Rdata") # PyTorch 保存的文件
logp_report <- compare_logp_objects(reference_objects, candidate_objects,
                                  target=0.001) # log10 尺度的最大绝对差目标
```

```bash
# 原生类型、属性、顺序、标识和 NULL；不比较浮点误差。
Rscript validation/compare_structure.R reference.Rdata candidate.Rdata structure-report.json
# 全部 P 的诊断比较；最后一个可选参数为 log10 尺度绝对差目标。
Rscript validation/compare_logp.R reference.Rdata candidate.Rdata report.json 0.001
# schema、下溢与 native log 字段控制测试；不是科学 benchmark。
Rscript tests/test_logp_validation.R
```

独立验证需要 R 的 Matrix 和 jsonlite；生产 PyTorch 运行环境无需增加 R 依赖。没有对应原软件的关联命令：这里对照的是既有原生输出，原关联命令仍以各功能文档为准。

## 输出及目标

JSON 只包含匿名汇总：字段数、P 数量、结构错误数、raw-P 最大绝对/相对误差、logP 最大差及 p50/p90/p95/p99、超过 `1e-4` / `1e-3` / `1e-2` 的数量、缺失/无穷/越界/零值数量。按参考 P 的互斥区间分层，并另给 `P<1e-4`、`P<1e-8`、`P<1e-12` 累计分层。空区间的误差为 JSON null，不能解释为误差为零。

本版主数值目标是在 `P_R<0.05 OR P_GPU<0.05` 的联合范围内，每个值都满足 `|Δlog10(P)|<=0.001`，对应 P 比率最多约 `10^0.001=1.0023`。任一侧跨越 0.05 的值仍参与比较，边界跨越数量单独记录；若 logP 差在目标内，不因跨界本身拒绝。两侧均非显著的 P 差异完整保存，作为全部 P 的诊断。任何 P 无效、不可比较、覆盖不足或 native log 不一致都不能因显著数为零而通过；全部原生结构和严格零模型仍是主验收条件。旧 FP64 和既有失败报告不改写。

正数 P 按 double 计算对数；接近 1 使用 `log1p(P-1)`。不会 clamp。零值单独记为 underflow，即使双方都为零也不能自动通过。只有 Single 已有有限、非负的原生 `pvalue_log10` 与 P 一致，且零值的 native log 不小于 double 最小次正规数的负 log10，才用原有 log 值恢复比较；恢复数量和原始不可比较数量分别报告。参考 raw P=0 只有在双方 native log 已通过原恢复规则时，才按已有 `-log10(P)` 计入极小 P 分层及三个累计分层；原恢复阈值使这些值属于 `(0,1e-100)`。正 raw P 的分层比较式和边界保持不变；未恢复的零值不会计作通过的极小尾部。没有可用原生 log 的零值、NA/NaN、Inf 或越界值会使 logP gate 失败。缺失值结构由 `compare_structure.R` 检查。

## 本版与历史验证记录

2026-10-06 本版 F 完整 chr21：795 项任务、19 份原生文件结构及严格零模型通过；15 份非空 P 文件覆盖 478,082 个有效、可比较 P，三个原合法空关联文件保持 NULL 并严格检查。显著联合范围有 24,713 个 P，超限为零，最大 logP 差 `0.0003579714`。 全部 P 的旧诊断仍有 49 个超限、最大差 `0.0674462632`；这些双方非显著值完整保留在诊断中，旧全部 P 的 pass 字段保持 false。进程墙钟 `429.097 s`，300 秒目标未达；固定零模型、已有转存缓存、首次转存与暖文件系统边界见 [完整 benchmark](../docs/tf32_benchmark.md)。先前 B/C2/E 的严格零模型各有 332 个 residuals 超限，旧报告保留失败状态。

2026-10-05 新增独立工具和 schema/下溢控制。已有完整 FP64 serial 对原 R 的 18 文件只读基线覆盖 478,082 个 P：全部可比较、无下溢；最大 logP 差 `3.41330297359832e-11`，p99 `4.15778522722121e-14`。该记录来自旧运行产物，未重跑关联，不是 TF32 的精度或耗时证据。完整 SHA 绑定与详细分层保存在 private，仅匿名汇总可用于发布。

该真实基线 `P<1e-4` 有 42 个值，最大 logP 差 `5.86197757002083e-14`；`P<1e-8` 和 `P<1e-12` 均没有值，相关误差返回 null。Single 的 318,132 个原生 `pvalue_log10` 最大直接差 `7.90478793533111e-14`，与原始 P 的定义一致性错误为 0。这套基线未覆盖更小 P 的真实尾部。

## 显著联合范围的调用

`compare_significant_logp.R` 复用 `compare_logp.R` 的同一次递归 P 识别、原始值检查及 native log 恢复规则，增加显著联合范围汇总；不会另选少数统计列或重写原全部 P 诊断。

```r
source("validation/compare_significant_logp.R")
compare_significant <- significant_logp_function("validation/compare_logp.R")
native_reader <- attr(compare_significant, "oracle")$read_logp_native
reference_objects <- native_reader("reference.Rdata") # 原 R 的同一任务输出
candidate_objects <- native_reader("candidate.Rdata") # PyTorch 的同一任务输出
significant_report <- compare_significant(reference_objects, candidate_objects,
                                          target=0.001)
print(significant_report$significant_logp_passed)
```

```bash
Rscript validation/compare_significant_logp.R \
  validation/compare_logp.R reference.Rdata candidate.Rdata \
  significant-report.json 0.001 0.05
```

| 输入 | 含义与格式 |
|---|---|
| 第一个参数 | 原 `compare_logp.R` 脚本路径，保留原识别及有效性语义 |
| 第二、第三个参数 | 同一任务的原 R / PyTorch 原生 Rdata 或 RDS 路径，顺序固定 |
| 第四个参数 | 写出的匿名 JSON 报告路径 |
| 第五个参数 | 可选有限非负 logP 绝对差目标，默认 0.001 |
| 第六个参数 | 可选显著性阈值；本版固定只接受 0.05 |

输出保留全部 P 的原诊断，并新增 `all_P_contract_passed`、`significant_logp`、`significant_stored_pvalue_log10`、`significant_logp_passed`。显著汇总含参考/候选/联合数量、仅一侧显著数量、0.05 边界跨越数量、逐值超限数及最大误差。退出码按显著联合范围 gate 决定，原 `logp_passed` 仍表示全部 P 旧诊断，可为 false。合法原空文件没有 P，该显著比较器单独调用不将其判为 P 精度通过；完整验收按原空结果、结构及严格比较处理。单文件 CLI 不能代替完整调度的文件数、原 basename、目录和全部 mask 覆盖检查。

## 结构、全部 P 诊断与严格零模型

```bash
Rscript validation/compare_structure.R reference.Rdata candidate.Rdata structure-report.json
Rscript validation/compare_logp.R reference.Rdata candidate.Rdata logp-report.json 0.001
# 零模型和原合法空结果的严格数值比较；关联非 P 误差另作诊断。
Rscript validation/compare_nonp.R reference-null.Rdata candidate-null.Rdata null-report.json
Rscript tests/test_structure_validation.R
Rscript tests/test_logp_validation.R
```

结构报告检查 typeof、长度、属性和顺序、S4 class/slots、factor、整数/字符串标识、NA/NaN/Inf 和 NULL；不比较 double/complex 的误差。null 文件没有关联 P，必须通过结构检查及原严格非 P 数值比较；没有 P 不能免除零模型验收。Single 的原生 log 字段参与 logP 门槛，`pvalue_log` 的正 `-ln(P)` 先除以 `ln(10)`；不新增原文件没有的列。

原本合法空 mask 文件可以没有 P 字段：logP 报告的比较数是零，明确标为不适用，必须通过结构报告及原严格空结果比较。所有非空 P 字段都必须有效、完整、可比较；主误差门槛逐值作用于显著联合范围，不只看均值或分位数。不能用少数字段推断显著范围或替代全部 P 的识别和覆盖检查。原生文件的集合、basename、保存对象名和 mask 覆盖按完整调度核对；候选前后 SHA、输入绑定和冻结源码哈希保存在 private。

`compare_nonp.R` 与 `compare_native_outputs.R` 保留原语义：严格零模型与合法空结果继续作为 gate，非 P 关联误差为诊断；原 FP64 基线记录不改写。完整速度目标为每染色体端到端 <=300 秒，本版缓存流程为 `429.097 s`，目标未达。首次转存和新拟合另计，不能用缓存统计矩阵或短选段计时替代完整关联流程。
