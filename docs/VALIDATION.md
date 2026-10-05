# 真实 discovery 验证：2026-10-04–05

本轮验证对象为一个连续表型的完整 chr21 discovery：独立对照的全芯片 Step1、整条染色体 single-variant、全部 coding/noncoding gene-based 组及汇总。chr21 是现场清点后位点数最少的染色体；分析入口也支持其余常染色体和更换表型列。按照本次交付范围，当前 v5 使用 GPU 串行重新验证全部关联与汇总；历史 v4 的 2 workers 结果另行标明。

本轮使用同一份已冻结的计算源码，21 个实现模块通过 SHA-256 核对一致。当前全串行验证标识为 `discovery_all26_gpu_serial_final_v5`，采用 `ExecutionConfig(parallel_level="serial", workers=1, max_gpu_gb=20.)`；最终完成状态、耗时、显存与严格数值验收待本轮输出闭合后登记。完整 509k Step1 LOCO 在本轮导入，拟合证据见下表。历史快照 `discovery_all26_adaptive_final_v4` 的 single、全部 26 组 gene 和 Summary 已完成，但属于 2 workers 运行。当前全量 GPU 串行验证尚未完成。

| 阶段 | 真实数据范围 | 当前证据 |
|---|---|---|
| Step1 | 509,468 个 QC 芯片位点；41,538 个有效表型 | GPU 与原 REGENIE 完整拟合及 LOCO 对照已完成 |
| Single | chr21 全部 13,733,596 个 WGS 位点；41,538 个有效样本 | v5 串行已完成，314,209 行格式及五项数值门槛全部通过；阶段耗时 424.529767 秒 |
| Gene 原程序 | 全部 14 Main + 12 Sub；各组全部 gene/masks | 26 组完成，共 97,222 行、12 种 TEST；文件审计通过 |
| Gene GPU | 同一 chr21、26 组及全部源定义 | v5 串行正在验证；历史 v4 为 97,222 行、26/26 结构及附属文件通过，1/26 组全部数值通过 |
| GPU 串行测量 | PTV、Missense、Intron、Pseudo、RNA、Intron_Gnocchi4 六个完整组 | 已完成 32,842 行；6/6 结构及附属文件通过，0/6 组严格数值通过；计时见下文 |
| 最终完整 pipeline | v5 GPU 串行全量关联与显著性/locus 汇总 | 正在验证，最终闭合记录待登记；历史 v4 已核验 27 个关联阶段及三个 Summary 文件 |

上述 chr21 验证没有筛取少量 gene 或截短 mask。早期的 320 位点、3-gene 对照保留在本文后部，作为历史验证记录。

## 数据、参数与原软件

依据用户提供的 Methods PDF，原软件基线固定为 REGENIE 3.4.1 官方 tag 的 Centos7 MKL 二进制，现场核对版本。服务器旧版本的结果不混入本轮对照。原软件仅由独立验证脚本执行，生产分析不调用它。

正式 discovery 名单为 44,441 人；与最终芯片/WGS 和排除名单对齐后输入 44,365 人，单个连续表型（公开代称 `trait_01`）有效 41,538 人、缺失 2,827 人。Step1 保留缺失表型对应的基因型行与 mask，维持原 CV 边界。最终 QC 版本的 22 条常染色体合计 1,024,174,307 个位点，与论文 QC 数量一致；旧版本提交脚本用于核对命令和分析规则。chr21 有 59,360 个 FAM 行、13,733,596 个位点，BED 为 203,806,564,643 字节。输入覆盖、annotation/setlist/mask 数量及 SHA-256 见 [全量范围清单](../benchmarks/discovery_scope_2026-10-04.json)。

真实验收使用覆盖完整 discovery 的既有残差表。另一个名称含 discovery 的旧残差表仅覆盖 32,908 名名单成员，未用于本轮。输入参与者记录、基因型、逐变异/逐基因结果、私有脚本和统计矩阵保留在 验证服务器；仓库发布代码与聚合验证记录。

冻结原软件的 ridge、连续表型投影和 VC 协方差使用 Eigen `MatrixXd`/`SparseMatrix<double>`，L0 中间文件也写 double，没有 float32 运行开关。因此基线为 float64。PyTorch Step1/single 默认使用 float32/TF32；可设 `dtype="float32", tf32=False` 或 `dtype="float64", tf32=False`。gene 投影、协方差与推断保持 float64，共线 mask 的 SBAT 选列对舍入敏感。不使用 float16。原实现见 [Step1](https://github.com/rgcgithub/regenie/blob/v3.4.1/src/Step1_Models.cpp)、[QT 投影](https://github.com/rgcgithub/regenie/blob/v3.4.1/src/Step2_Models.cpp)和 [VC 计算](https://github.com/rgcgithub/regenie/blob/v3.4.1/src/SKAT.cpp)。

硬件为共享 A100 80GB；PyTorch 2.0.0+cu118、Triton 2.0、NumPy 1.26.4、SciPy 1.13.1、pandas 1.4.4。当前容器 CPU 总配额为 8 核。原 gene 驱动为两进程，每进程请求 8 线程，共享该配额。报告显存为本进程 PyTorch 峰值分配；时间为共享负载下的观测，不作为独占硬件性能。

## 完整 Step1

真实 509,468 个 QC 芯片位点，521 个 block、2,605 个 L0 预测列、5-fold。默认 GPU float32/TF32 与同队列原程序（8 CPU 线程）对照：

| 指标 | 结果 |
|---|---:|
| 两者选中 L1 h | 0.25 |
| CV MSE 最大差（原日志 6 位值） | 9.98×10⁻⁶ |
| chr1–22 LOCO 最大绝对差 | 5.92×10⁻⁴ |
| LOCO RMSE | 1.28×10⁻⁴ |
| LOCO 相关系数 | 0.999999869 |
| chr23 PRS 最大绝对差 | 5.58×10⁻⁴ |
| GPU 总耗时 / L0 / L1 | 644.708 / 639.34 / 0.78 秒 |
| 原程序总耗时 | 1131.51 秒 |
| GPU 峰值分配 | 986,269,184 字节 |

总耗时包含初始化、整理和保存。LOCO 按 FID/IID 对齐；原预测文件只写 6 位有效数字。23 行染色体、样本映射、chr23 全基因组预测和 `pred.list` 均已检查。原程序/GPU 的观测耗时比约 1.76（1131.51/644.708）。

整条 chr21 的 Step2 使用这次完整 509k Step1 的 LOCO。用于样本顺序检查的小芯片输入不重新拟合 null，不能把它解释为 320 位点训练的 Step1。

Step2 对照两边共用上述 GPU 导出的完整 LOCO，以单独检查关联统计；Step1 的预测误差独立记录在前表。

## 全染色体 single-variant

GPU packed BED 解码扫描 chr21 全部 13,733,596 个位点。原始关联使用 `minMAC20`，不提前做 MAF 筛选；输出 314,209 行、原 13 列及对应 `.regenie.ids`。在结果层应用论文的 MAF>0.001 后保留 185,552 行；本染色体使用旧 验证服务器 的 A1FREQ>0.001 恰好得到同样数量，两种定义仍分别开放。

当前 `discovery_all26_gpu_serial_final_v5` 已按串行配置扫描上述完整范围，N=41,538，输出 314,209 行，single 阶段耗时为 424.529767 秒。独立的新输出比较确认格式和五项数值门槛全部通过；该时间属于 single 阶段，整体 pipeline 尚未完成。

历史 `discovery_all26_adaptive_final_v4` 已扫描全部位点，N=41,538，输出 314,209 行，single 阶段耗时为 220.776900 秒。独立输出比较确认全部原始结果行的表头、空白布局、行序、结构字段、NA 和 `.ids` 与原程序一致，五项数值门槛全部通过；最大打印差为 A1FREQ 0、BETA/SE 约 10⁻⁶、CHISQ/LOG10P 约 10⁻⁵。以下 v3 结果也保留为历史验证。

上一冻结版 `discovery_all26_adaptive_final_v3` 的 single 已完成上述完整扫描，N=41,538，阶段报告耗时为 225.320409 秒。全部 314,209 行的表头、空白布局、行序、结构字段、NA 和 `.ids` 与原程序一致；A1FREQ、BETA、SE、CHISQ、LOG10P 五项数值门槛的超容差数均为 0。最大打印数值差为 A1FREQ 0、BETA/SE 约 10⁻⁶、CHISQ/LOG10P 约 10⁻⁵。两份 `.regenie` 文件大小相差 169 字节，结果数值并非逐字节相同。

v3 和 v4 都只记录 single 阶段耗时，没有独立测量该阶段的进程墙钟、读取/解码与计算 profile 或峰值分配。v4 的 8,044,102,656 字节峰值属于整个 pipeline，不能标为 single 峰值。两轮阶段耗时不沿用更早运行的拆分时间与显存，也不与旧 CPU 的 9646.819 秒换算加速倍数。不同运行的共享负载、缓存和计时边界不同。

以下是此前完整扫描的历史计时：

| 首轮 GPU 全量运行 | 时间 / 显存 |
|---|---:|
| 端到端总耗时 | 1419.548 秒（23.66 分钟） |
| 关联阶段总耗时 | 1409.768 秒 |
| packed BED 读取/解码 | 994.597 秒 |
| 关联计算与写出 | 415.171 秒 |
| 峰值分配 | 937,278,464 字节（0.873 GiB） |

原 REGENIE 的全位点扫描耗时 9646.819 秒，与上述 GPU 端到端时间的观测比为 6.796 倍。两边完整比较 314,209 行：表头、空白布局、行序、所有结构字段、NA 及 `.ids` 完全一致，五个数值字段的超容差数均为 0。最大打印数值差为 A1FREQ 0、BETA/SE 约 10⁻⁶、CHISQ/LOG10P 约 10⁻⁵。两份 `.regenie` 的文件大小相差 169 字节，结果内容因打印数值差异而非逐字节相同。旧 A1FREQ 与论文 MAF 汇总都保留 185,552 行。[完整匿名对照](../benchmarks/full_chr21_single_2026-10-04.json)记录源码 hash、逐列误差及实际计时；共享负载期间同时运行过其他验证任务。

前一冻结版 v2 再次扫描全部 13,733,596 个位点，输出 314,209 行、N=41,538；single 阶段报告耗时为 226.106 秒，逐行格式与数值检查全部通过。该版为补齐 Step1 日志而停止，保留为历史记录。它未单独记录解码/计算 profile 或阶段峰值，不能沿用首轮的拆分耗时和显存。

## 全部 coding/noncoding gene-based

`study_gene_analyses()`按现场文件展开 26 组；每组读取全部 setlist gene，并生成 source category、C/R/UR domain、overall、singleton 与 AAF=.01 masks。

| 分组 | 完整范围 |
|---|---|
| Main：14 组 | PTV、Missense、Splice、Inframe、Synonymous、Intron、UTR_5、UTR_3、Upstream、Downstream、Intergenic、Pseudo、RNA、nctev |
| Sub：12 组 | Intron/Intergenic/UTR_5/UTR_3/Upstream/Downstream × GERP2；Intron/Intergenic × Gnocchi4；Splice_splice05；Missense_REVEL50；Upstream/Downstream × JARVIS99 |
| 源定义 | 26 组共 42 个基础 mask 定义全部读取；本次 chr21 共注册 37 个活跃表头定义 |
| 统计检验 | ADD、SKAT、SKATO、ACATV、ACATO、SKATO-ACAT、BURDEN-ACAT、ACATV-ACAT、BURDEN-SBAT/POS/NEG、GENE_P |

原提交脚本列出 13 个 Main；本次使用已有完整 annotation/setlist/mask 增加 Intergenic Main，以覆盖全部现成 mask 文件。评分条件读取原白名单，不从文件名推断未核实的阈值。

原 REGENIE 26 组全部完成，共 97,222 行、12 种 TEST，LOG10P 缺失行数为 0。两进程驱动的总墙钟时间为 6657.882 秒；各进程耗时之和为 12943.135 秒，不能用该和替代端到端时间。

v5 采用 GPU 串行配置，从已验证的完整 509k Step1 LOCO 导入，重新验证全部 single、26 组 gene 和三个 Summary 文件。本轮全量结果、耗时与显存待闭合后登记。

历史 v4 使用 `parallel_level="mask", workers=2`，从同一完整 509k Step1 LOCO 导入，重新计算全部 single 与 26 组 gene，并完成三个 Summary 文件。统一 pipeline 记录耗时为 10945.423 秒，包含 single、gene 和 Summary，属于函数内流程计时；未记录独立子进程墙钟，也未重新拟合 Step1。峰值分配 8,044,102,656 字节（约 7.49 GiB），低于 20 GiB 分配器预算。该记录属于历史 2 workers 运行，时间边界与原 gene-only 驱动不同，两者不换算受控倍速。

### 文件格式与 mask 表头

结果保留 REGENIE 的文件名、13 列、6 位有效数字、空格分隔、行序、ID/allele、N、TEST、EXTRA、NA 位置与 `##MASKS` 标头；`.ids` 首行按 `--print-pheno` 写表型标签和 `NA`，以通用代称表示为 `trait_01\tNA`。开启 `write_masks=True` 时检查 `_masks.bed/.bim/.fam/.snplist` 的内容和顺序。

表头类别按完整 annotation 与 BIM/全局白名单/评分白名单的交集注册。未知类别从表头删除；类别注册发生在 setlist、gene 选择、MAC/AAF 筛选之前。chr21 的 Pseudo 8 个源定义实际注册 5 个，RNA 10 个实际注册 8 个，26 组共注册 37 个活跃定义；42 个源定义仍完整读取。其他染色体根据各自输入注册。这一规则已加入回归测试，避免为了保留空类别而改变原表头。

Step1/Step2 的 `.log` 与主 `discovery.log` 为可读文本，记录实际 PyTorch 引擎、参数、样本/位点/test 数与耗时。新增 `Step1/discovery.log` 区分实际拟合、缓存加载和外部 LOCO 导入，Elapsed time 记录本次操作；缓存另标原拟合时间，导入不推测源拟合耗时。三种分支均由既有编排测试实际检查，缓存同时核对并更新日志身份。Step1 进度仍记录在主日志，拟合参数与阶段统计另存 `null_model.json`。软件标识、真实耗时和事件内容不与原 REGENIE 程序的日志逐字节一致。结构化事件保存为独立 `discovery.events.jsonl`，审计信息保存为 manifest/progress JSON，不改变原结果列。

早期冻结版完成的14组中，12组结构检查通过，Upstream/Pseudo存在kernel行数差异，RNA/nctev因退化残差谱中断。v3完成single后因Pseudo多一行ADD-SKATO停止，保留为历史诊断。修复后的v4已完成全部26组：97,222行与原程序一致，表头、行序、N、NA、EXTRA、`.ids`及四类mask文件全部通过。

Upstream 退化 kernel 的定向修复已完成：多位点秩一核的burden消除残差谱为空时，按原版保留ACAT-V并跳过kernel tests；联合检验只计入其有效mask。真实问题基因的CPU复查为29/29行、全部结构字段一致，GENE-P差降至2×10⁻⁶；[匿名记录](../benchmarks/real_upstream_kernel_failure_fix_2026-10-04.json)保留该案例的SKATO数值误差。全组结构验收已纳入v4。

v3 的 Pseudo 全组比原程序多一行 ADD-SKATO。真实 mask 的 CPU 重放发现，排除原程序允许的主动零值后，194 个有效积分节点的 log-SF 有限，但普通 float64 SF 下溢为 0。原 `SKATO_integral_fn` 在 `S<=0` 时使积分失败；该 mask 的 Bonferroni P>1，因此原版省略 SKATO。修复恢复此规则，同时保留其他 kernel 检验。新源码对同一真实 mask 的独立 CPU 重算得到 `SKATO=None`、kernel 仍有效，SKAT、BURDEN、SKATO-ACAT 和全部 rho 结果逐位相同；总耗时 16.074 秒。条件 SF 的下溢判定使用 IEEE float64 日志域阈值，避免 CUDA `exp` 在最后正 subnormal 附近提前返回 0；14 个匿名邻界数值与独立 CPU `math.exp` 判定一致。积分目标和最终 SKATO P 也采用原 `10*DBL_MIN` 地板，单站点、固定 rho 与原始数学 API 保持既有行为。v4完整Pseudo GPU输出与原程序同为4,287行，ADD-SKATO均478行，结构和附属文件通过；该组整体数值仍未通过。[匿名记录](../benchmarks/real_pseudo_source_gate_2026-10-05.json)同时保留CPU修复与完整GPU结构证据。

### 预先设定的数值门槛与未通过项

逐行比较的容差为：A1FREQ 绝对差 2×10⁻⁶；BETA/SE 绝对差 1×10⁻⁶ 加相对差 1×10⁻⁵；CHISQ 绝对差 1×10⁻⁴ 加相对差 1×10⁻⁵；LOG10P 绝对差 1×10⁻⁴。非有限值与 NA 检查原始 token。文件结构通过与数值门槛通过分别报告。

以下为历史 v4 的 2 workers 运行结果，v5 串行的全 26 组数值验收待本轮完成后登记。v4 全部 26 组已按上述门槛逐行比较，只有 `Splice_splice05` 全部数值通过（1/26）；其余组仍登记为严格未通过。ADD、ACAT-V、ACAT-O、SKATO-ACAT、BURDEN-ACAT及ACATV-ACAT的全部行通过。未通过项为：

| TEST | CHISQ超门槛行数 | LOG10P超门槛行数 | LOG10P最大绝对差 |
|---|---:|---:|---:|
| ADD-SKAT | 1 | 0 | 5.7×10⁻⁵ |
| ADD-SKATO | 38 | 0 | 1.0×10⁻⁴ |
| ADD-BURDEN-SBAT | 1254 | 818 | 0.207425 |
| ADD-BURDEN-SBAT_POS | 951 | 595 | 0.186240 |
| ADD-BURDEN-SBAT_NEG | 959 | 566 | 0.105340 |
| GENE_P | 513 | 345 | 0.094880 |

ADD-SKAT的1行属于`Downstream_JARVIS99`，CHISQ最大差为1.72×10⁻⁴；ADD-SKATO的38行也仅CHISQ超门槛，最大差为3.30×10⁻⁴，二者LOG10P均通过。下述Splice积分归因只覆盖其中两个实际mask，不能扩展到其他未定位行或ADD-SKAT。

另已确认两个真实 Splice mask（VC维数2和4）的 SKAT-O CHISQ 微超既定门槛，LOG10P 均合格。相同原坐标、epsabs=1e−25、epsrel=2⁻¹³和1000区间预算下，验证用QAGS默认路径可复现原程序；收紧QAGS预算后的结果更接近当前PyTorch默认值。

| 两例定向方法 | 对原程序CHISQ最大绝对差 | LOG10P最大绝对差 | 严格门槛 |
|---|---:|---:|---|
| 当前PyTorch默认GK21自适应路径 | 1.45664×10⁻⁴ | 4.78777×10⁻⁵ | CHISQ未通过，LOG10P通过 |
| 验证用QAGS，原默认预算 | 4.22755×10⁻⁶ | 2.82617×10⁻⁷ | 两者通过 |
| 验证用QAGS，epsrel=1e−7 | 1.46454×10⁻⁴ | 4.80058×10⁻⁵ | CHISQ未通过，LOG10P通过 |

收紧预算后的QAGS与当前默认结果的LOG10P差≤2.19×10⁻⁷；所有方法的SKAT、BURDEN、rho及SKATO-ACAT保持逐位相同。两例的诊断支持原QAGS外推/停止路径与当前自适应路径的积分近似值差异，双方均报告收敛。CHISQ门槛保持原值，这两例仍登记为严格未通过；归因范围仅为捕获的两例。CPU验证专用QAGS不进入生产计算，[匿名诊断](../benchmarks/real_splice_quadrature_2026-10-05.json)保存误差和收敛信息，不含原始统计值、输入矩阵或身份记录。

PTV 的诊断中，QR 保留列次序与原程序一致，NNLS 系数最大绝对差为 1.62×10⁻⁹，但共线 Gram 矩阵条件数约 7.94×10⁸。原程序重复运行也有部分 SBAT 波动；这不足以解释当前最大的差异。验收容差保持原值。

Inframe 最差 SBAT 案例已经通过捕获原二进制的实际 OLS 协方差定位。原算法先形成逆 Gram，再以 Schur 相减计算正交概率协方差；该病态模型中两个 Schur 矩阵出现非正方差，原概率子程序仍返回 0.5。对同一实际输入按原子矩阵次序进行 80 位精度重算，两处方差均为正。当前稳定 Gram 权重与高精度独立结果最大差为 8.63×10⁻¹³，原权重误差为 0.0350；将捕获的原权重代入 PyTorch NNLS/尾概率，SBAT/POS/NEG 的 LOG10P 差均小于 1.92×10⁻⁶。这说明该案例的主要差异来自原混合权重的消减失稳。保留稳定公式，同时将原阈值未通过项明确列出；[匿名证据](../benchmarks/native_sbat_precision_2026-10-04.json)不包含输入矩阵或身份记录。

RNA 与 nctev 的首个运行失败案例也已完成真实 CPU 回放：原版对退化的双位点核只保留 ADD 与 ACAT-V，最新实现分别输出 55/55 与 29/29 行，行序、结构字段及 NA 位置均一致。[退化核修复记录](../benchmarks/real_gene_kernel_failure_fix_2026-10-04.json)的范围为定向验证；完整GPU组结构已在v4通过，数值限制见上表。

## 大 mask 的 GPU 优化与串行测量

packed 解码将 BED 原始字节送入 GPU；gene 稀疏路径保存折叠后的非零基因型，通过分块交叉乘积计算 score 和协方差，减少 N×M 中间矩阵。真实最大 RNA mask 的 VC 维数为 7,343；最大 Intron overall mask 为 11,392。预估稀疏工作区分别为 2.73 GB 和 6.42 GB；旧稠密路径估计为 9.91 GB 和 17.59 GB。估计不是实测峰值。真实构造与 256 列 score/协方差对照见 [大 mask 预检](../benchmarks/vc_highdim_preflight_2026-10-04.json)。

本次交付采用 `ExecutionConfig(parallel_level="serial", workers=1, max_gpu_gb=20.)`，各组依次在 GPU 上计算全部 gene/masks。同一冻结源码的六个完整组串行测量已经完成：PTV、Missense、Intron、Pseudo、RNA、Intron_Gnocchi4，共 32,842 行。进程墙钟为 6473.804485 秒，pipeline 函数计时为 6470.960217 秒，峰值分配为 6,605,298,688 字节（约 6.15 GiB），总分配器预算为 20 GiB。六组结构及附属文件全部通过，严格数值为 0/6 组通过。该测量覆盖六组，不能替代全 26 组串行验收，也不与原全 26 组 gene-only 驱动换算倍速。当前 v5 正在重新验证全部 single、26 组 gene 和汇总。

API 保留 `parallel_level="mask"` 的组并行和 `parallel_level="chromosome"` 的染色体并行能力，工作线程各自使用独立 CUDA stream。每个任务保留全部 masks，总预算按并发数分配；预算不足或 CUDA OOM 的任务在其他任务结束后按总预算串行重试，失败尝试与重试时间都计入总流程。本次性能结论以 GPU 串行测量为准。

大矩阵数值后端已经单独完成真实对照：

| 优化 | 同一真实输入的分步骤结果 | 精度 |
|---|---|---|
| 秩一 secular 特征值更新 | RNA 7,343 维，8 次 rho/残差分解合计约 9.94→2.82 秒 | 最终 rho LOG10P 最大差 1.54×10⁻⁸；SKATO 差 3.64×10⁻⁹ |
| Davies NumPy 误差界控制器 | 585 个真实规划，CPU 标量 131.021→3.679 秒 | 所有规划、故障和预算字段逐位相同 |
| 完整 SKAT-O 阶段 | 同一 RNA 协方差、固定 dense 特征值后端，145.241→18.2767 秒 | SKAT/BURDEN/所有 rho/SKATO/SKATO-ACAT 逐位相同 |

同一真实 RNA 7,343 维 score/协方差还完成了积分后端的配对检查。固定 dense 特征值后端、auto Davies 控制器和原 kernel 有效性规则，依次运行三种积分路径；这里只比较 SKAT-O 阶段，不包含 BED、mask 构建或整条 pipeline。

| 积分后端 | 阶段耗时 | 积分节点 / 自适应区间 | 收敛诊断 |
|---|---:|---:|---|
| segmented：早期分段 Gauss-Legendre | 13.823 秒 | 历史分段规则 | 数学 API 保留此默认路径 |
| adaptive_sqrt：可选平方根坐标 | 12.985 秒 | 819 / 20 | 相对误差估计 9.82×10⁻⁵，通过 |
| adaptive_x：gene 默认，原 χ² 坐标 | 18.474 秒 | 1,869 / 45 | 相对误差估计 9.59×10⁻⁵，通过 |

两种自适应路径使用独立 PyTorch Gauss-Kronrod 21 规则，绝对误差预算为 10⁻²⁵、相对误差预算为 2⁻¹³、最多 1,000 个区间。三个后端的 SKAT、BURDEN、8 个有效 rho、所有 rho LOG10P 及 SKATO-ACAT 逐位相同；adaptive_sqrt 相对 adaptive_x 的 SKATO LOG10P 差为 3.31×10⁻⁷，相对早期 segmented 为 2.49×10⁻⁸。峰值均约 3.032 GB，CPU 尾积分 fallback 为 0。共享 A100 上的这次阶段计时不能替代全 pipeline 的端到端测量；[匿名配对记录](../benchmarks/real_rna_quadrature_2026-10-04.json)保存源码哈希和完整诊断。

后续全组对照发现，真实 Missense 的一个 9 维 mask 中，平方根坐标的初始 21 个取样节点漏过窄特征：它报告相对误差 2.83×10⁻¹²、已经收敛，但相对原 REGENIE 的 SKATO LOG10P 差为 1.09×10⁻⁴；将预算收紧至 2×10⁻⁶仍给出相同结果。保留原 χ² 坐标及原误差预算后，独立 PyTorch GK21 使用 1,323 个节点、32 个区间，LOG10P 差降为 4.06×10⁻⁷，CHISQ 差为 6.16×10⁻⁸。验证专用的原坐标 QUADPACK 对照差分别为 3.30×10⁻⁷和 3.59×10⁻⁸；生产计算仍使用 PyTorch。各方法的 SKAT、BURDEN、rho 值和 rho LOG10P、SKATO-ACAT 逐位相同。该 [匿名 CPU 定向记录](../benchmarks/real_missense_quadrature_2026-10-04.json)只覆盖这个实际误差最大的 mask，并非完整 pipeline 验收。GeneConfig 因此默认使用 adaptive_x；adaptive_sqrt 保留为可选项。回归同时使用独立的解析窄端点积分说明这种采样盲点。原坐标改善了这些实测案例的复现，但自适应误差估计不能普遍保证捕获所有窄特征。

随后按相同误差预算，在 2 个 CPU 线程上定向重算旧平方根路径的全部 34 个 Missense SKATO 失败行，34/34 通过既定 CHISQ 和 LOG10P 门槛。最大 LOG10P 差为 4.88×10⁻⁶，最大 CHISQ 差为 1.00×10⁻⁵；节点数为 1,113–1,869，包含读取和准备共耗时 55.435 秒。该 [失败子集回放记录](../benchmarks/real_missense_failed_rows_quadrature_2026-10-04.json)包含匿名聚合诊断与冻结源码哈希；完整染色体和全部 mask 仍以全流程对照为准。

积分修复也通过真实 Upstream 问题基因的 GPU 定向对照：全部 29/29 行的结构字段与 DF 一致，三个有效 VC mask 的 ADD-SKATO 打印值与原程序完全相同；该案例最大 LOG10P 差为 2.7×10⁻⁵，CHISQ 差为 6×10⁻⁵，剩余差异来自 SBAT。该范围是单个实际问题基因，v4全26组验收见前表。[匿名定向记录](../benchmarks/real_upstream_adaptive_integral_2026-10-04.json)不含 gene/site/样本标识或输入矩阵。

上述 Davies 控制器对照中约 7.95 倍是完整 SKAT-O **阶段**的观测加速，包含 585 次尾概率规划、一个原预算失败和一次 Kuonen 路径；不含 BED、mask 构建或整组 gene pipeline。参与者矩阵、特征值与 Fourier 积分留在 GPU；CPU 控制器只规划误差界与预算，默认 CPU 尾积分 fallback 为 0。详细计时、哈希、显存和共享 GPU 状态见 [秩一对照](../benchmarks/real_rna_rank_one_2026-10-04.json)、[585 个规划](../benchmarks/real_rna_davies_plans_2026-10-04.json)及 [完整 SKAT-O 对照](../benchmarks/real_rna_davies_controller_2026-10-04.json)。

## 汇总规则与运行验证

论文单变异阈值为 5×10⁻⁹/831.50；gene 阈值为 .05/(831.50×17,863)。Single 汇总按 MAF>0.001 过滤，按 ±500 kb 递归挑选 lead，再按相邻 lead≤1 Mb 合并 locus。它是物理距离规则，没有计算 LD 或 r²。

历史 v4 已比较 single 原始 314,209 行、论文 MAF>0.001 的 185,552 行和全部 26 组 gene 的 97,222 行，身份全部匹配，论文阈值下显著性决策全部一致，两边均 0 命中；三个 Summary 的显著 single 行、gene/TEST 行和 locus 数均为 0。[匿名决策记录](../benchmarks/real_discovery_significance_2026-10-05.json)仅覆盖本连续表型 chr21 的无命中场景，尚未验证阳性命中一致性或数值等价；该历史运行的 gene 严格数值验收为 1/26 组通过。v5 的三个 Summary 将随完整串行运行重新核验。

研究源脚本 `clump.py`另在 lead 候选中排除 chr6 的闭区间 [25,000,000,34,000,000]；PDF Methods 的 locus 描述没有这一额外区间。`excluded_locus_regions=((6,25000000,34000000),)`保留该源代码规则，设为 `()`可关闭。它只影响 `single_loci.tsv`，不删除 `.regenie` 或 `single_significant.tsv` 中的位点。

RINT 是可修改选项，统一关闭开关传到 Step1、single、gene 和 raw 输入分位数正态化。缓存核对输入、实现、参数和输出身份；改变汇总阈值可复用关联结果，缺失或修改结果文件会使对应阶段重算。独立任务写独立输出，完成后按输入顺序合并；失败不会把 partial 文件登记为完成。

上一冻结版 v3 的 105 项测试在本地与验证服务器 GPU 均通过，分别耗时 100.880 秒与 50.652 秒。覆盖 ridge/QT 独立解、Davies 冻结参考与故障路径、SKAT-O 极小 P、自适应积分与预算回退、窄端点积分回归、退化 kernel、SBAT/NNLS、packed BED 与样本重排、全部 mask 表头、评分与全局白名单交集、原格式导出、缓存及失败恢复。默认 mask 文件导出、Step1拟合/缓存/导入日志、Step2文本日志和完整 EOF 扫描计数也已检查。数学/模拟 fixture 用于回归检查，本文性能对照全部采用真实数据。

v4 冻结源码的完整 CUDA 回归在本地与验证服务器 均为 111/111 通过，测试框架报告耗时分别为 119.770 和 54.939 秒；验证服务器 测试前后的 21 个实现模块 SHA-256 一致，v5 沿用这套计算源码。新增六项覆盖积分概率范围、近 P=1 边界、条件 SF 下溢、CPU/CUDA 下溢边界、主动零值与概率地板。历史 v4 全量运行的文件通过与严格数值未全部通过分别记录；当前 v5 全串行验收状态见前表。

## 早期有界验证记录

以下是扩展到整条 chr21 之前的结果，保留用于追踪实现变化。

| 范围 | 当时的结果 |
|---|---|
| 真实 320 芯片位点，RINT 开/关 | float64 导出相对原程序差≤1×10⁻¹⁰；float32/TF32 LOCO 最大差≤3.80×10⁻⁴ |
| 固定 null 的 320 个真实 single 位点 | float64 `.regenie`与`.ids`逐字节一致；float32 最大打印 LOG10P/CHISQ 差 1×10⁻⁶ |
| 3 个 gene，Missense 910 输入与 REVEL50 288 白名单位点 | header/行序/ID/N/TEST/EXTRA/NA 和四种 mask 文件一致；最大 SKATO LOG10P 差 6.9×10⁻⁵，SBAT 6.1×10⁻⁵，GENE_P 1.5×10⁻⁵ |
| 小范围公开 CLI 完整 pipeline | 320 芯片位点 + 228 WGS 位点；RINT 开/关各输出 64 single 行与 75 gene 行；single 最大 LOG10P 差 1×10⁻⁶，gene 4.4×10⁻⁵ |
| df=1 逆 χ² GPU 派发优化 | 同一真实小范围 Main/Sub 的中位耗时 4.462→2.177 / 3.576→1.983 秒；全部打印统计值与优化前一致 |
| 白名单前移/BIM 缓存 | Sub 每次解码 910→288 列，Main 仍 910 列；四组 RINT 开/关结果和 mask 文件保持一致 |

小范围计时含不同的初始化及共享负载，不用于推断全量加速。逐次计时与误差见 [早期 discovery](../benchmarks/discovery_2026-10-04.json)、[逆 χ² 优化](../benchmarks/inverse_chi2_optimization_2026-10-04.json)和 [IO 优化](../benchmarks/gene_io_optimization_2026-10-04.json)。

## 输入来源和版本范围

本包读取现成 QC 的 BED/BIM/FAM、annotation、setlist、mask 定义和评分白名单。Calling、VEP 及 Table S24 注释资源生成属于上游输入；C/R/UR 标签按源文件保留。Raw 模式要求显式提供论文协变量，当前全量验收使用既有残差表，尚未完成 raw 协变量处理相对原 R 流程的全队列对照。

本版范围为连续单表型、常染色体 discovery。全部统计方法和参数入口已经实现；整条 chr21 的运行与严格验收状态以上述阶段表为准。22 条染色体全研究结果和二分类/生存/联合多表型不在本次验收范围。调用和原命令见 [API](API.md)、[完整流程](REGENIE.md)与 [冻结原软件源码](https://github.com/rgcgithub/regenie/tree/v3.4.1)。
