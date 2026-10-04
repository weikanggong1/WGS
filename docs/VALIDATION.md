# 真实 discovery 验证：2026-10-04

本版完成连续单表型discovery的GPU统计实现和真实数据对照。已执行完整芯片Step1，以及有界的真实WGS Step2和完整pipeline；尚未生成22条染色体、全部基因和全部表型的研究结果。

## 数据与原软件

依据用户提供的Methods PDF，默认采用REGENIE 3.4.1。参考程序是官方tag的Centos7 MKL二进制，现场确认版本；服务器已有4.1只作历史参考，不混入本表的3.4.1对照。原软件仅在独立验证脚本执行，`torchwgs`运行时不调用它。

冻结原软件的ridge、连续表型投影和VC协方差使用Eigen `MatrixXd`/`SparseMatrix<double>`，L0中间文件也写入double；没有可切换为float32的运行参数。因此下面的原程序基线为float64。PyTorch默认在Step1和single的大矩阵使用float32并开启TF32，可通过`dtype="float32", tf32=False`选择普通float32，或`dtype="float64", tf32=False`进行严格对照。Gene的投影与小矩阵保持float64：真实共线mask在float32投影下会改变SBAT选列。原实现见[Step1](https://github.com/rgcgithub/regenie/blob/v3.4.1/src/Step1_Models.cpp)、[QT投影](https://github.com/rgcgithub/regenie/blob/v3.4.1/src/Step2_Models.cpp)和[VC计算](https://github.com/rgcgithub/regenie/blob/v3.4.1/src/SKAT.cpp)。

discovery名单44,441人。与芯片/WGS及排除名单对齐后保留44,365人；表型`24485-2.0`有效41,538人、缺失2,827人。Step1保留原基因型行和缺失表型的mask，以复现原CV边界。实际采用覆盖完整discovery的残差表，现场发现另一个名称含discovery的旧表只覆盖32,908名名单成员，因此没有使用。

硬件为共享A100 80GB；PyTorch 2.0.0+cu118、Triton 2.0、NumPy 1.26.4、SciPy 1.13.1、pandas 1.4.4。报告显存为本进程PyTorch峰值分配，未包含其他用户进程或CUDA外部分配。时间是本次观测，未控制共享机器负载。

## 完整Step1

真实509,468个QC芯片位点，521个block、2,605个L0预测列，5-fold。默认GPU float32/TF32与同队列原程序（8 CPU线程）比较：

| 指标 | 结果 |
|---|---:|
| 两者选中L1 h | 0.25 |
| CV MSE最大差（原日志6位值） | 9.98×10⁻⁶ |
| chr1–22 LOCO最大绝对差 | 5.92×10⁻⁴ |
| LOCO RMSE | 1.28×10⁻⁴ |
| LOCO相关系数 | 0.999999869 |
| chr23 PRS最大绝对差 | 5.58×10⁻⁴ |
| GPU总耗时 / L0 / L1 | 644.71 / 639.34 / 0.78秒 |
| 原程序总耗时 | 1131.51秒 |
| GPU峰值分配 | 986,269,184字节 |

总耗时还包括初始化、整理和保存，不能由L0+L1简单代替。LOCO比较按FID/IID对齐，原程序预测文件只有6位有效数字。23行染色体、样本映射、chr23全基因组预测和pred.list均已检查。时间观测比约1.76，不能泛化为所有数据/机器的加速比。

另以真实320个芯片位点、相同有效样本数对照RINT开启/关闭两种设置。float64导出相对原程序差≤1×10⁻¹⁰；float32/TF32原始LOCO最大差≤3.80×10⁻⁴。该float64小范围结果没有替代全509k默认精度对照。

## Single-variant

独立检验用真实320个位点和41,538有效样本，先固定null以隔离关联核心。float64 `.regenie`与`.regenie.ids`逐字节一致；原13列、行序、allele、N、NA均一致。float32/TF32最大打印数值差：A1FREQ、CHISQ、LOG10P为1×10⁻⁶，BETA为1×10⁻⁷，SE为0。

这项有界检验的不同运行处于不同CUDA冷/热状态，因此不报告速度优势。MAF>0.001与minMAC20独立实施；原软件初始minMAC扫描结果在对照时按正文MAF再筛选。

## Gene-based

真实3个gene，Missense读取910个输入变异，REVEL50白名单提取288个；AAF筛选和低计数折叠后的最大VC维数为9和6。Main/Sub分别开启和关闭RINT，先隔离Step2，再输入原320位点Step1产生的同一LOCO，共8组真实对照。每组重复运行的文件逐字节相同。

所有对照的表头、MASKS顺序、结果行序、ID/allele/N/TEST/EXTRA、NA位置、`.ids`和mask BED基因型/BIM/FAM/snplist均一致。最终SBAT保留列顺序也与冻结二进制一致。带LOCO四组的最大打印−log10(P)差：

| TEST | 最大差 |
|---|---:|
| ADD、SKAT、ACATV、ACATO、SKATO-ACAT、BURDEN-ACAT、ACATV-ACAT | 0 |
| SKATO | 6.9×10⁻⁵ |
| BURDEN-SBAT / POS / NEG | 6.1×10⁻⁵ / 4.3×10⁻⁵ / 3.6×10⁻⁵ |
| GENE_P | 1.5×10⁻⁵ |

剩余差异来自数值积分和chi-bar权重近似；文件格式一致不表示这些统计值逐位相同。GeneConfig公开SBAT子集、Sobol样本数和seed，可调整数值精度。

带LOCO四组GPU观测耗时5.18/4.71/3.56/3.63秒，峰值约0.64GB；计时包含gene读取、mask构建、统计和输出，开始于context构建后。原程序日志为0.99/1.00/1.20/1.23秒，计时边界不同，这个小范围没有GPU提速结论。Davies CPU标量规划约占上述GPU端时间1.89–2.22%；默认CPU/SciPy尾积分fallback为0。参与者矩阵投影、eig、NNLS、Fourier积分和SBAT正交积分由GPU执行。

随后针对GPU调用开销优化df=1输出逆χ²。同一真实Missense/REVEL50、固定原LOCO、开启RINT和warm状态成对测量三次，计时仍从context构建后开始：

| 分析 | 旧版三次 / 秒 | 新版三次 / 秒 | 中位数旧→新 / 秒 |
|---|---|---|---|
| Main（81行） | 4.462 / 5.959 / 4.311 | 1.910 / 2.191 / 2.177 | 4.462→2.177 |
| Sub（75行） | 5.727 / 3.576 / 3.404 | 1.894 / 1.983 / 3.990 | 3.576→1.983 |

保留全部观测，共享负载造成明显波动。Main/Sub全部12个TEST的六位LOG10P、CHISQ及其他字段与优化前输出完全一致，原程序差异范围保持。逐次GPU同步的归因测量中，66/61次df=1逆变换累计从2.411/1.764秒降至0.056/0.070秒；Davies、Kuonen路径次数和积分项数相同。该归因计时有额外同步，不与未插桩的总耗时相加。

## 数值修复与单元验证

最终使用公开CLI，经JSON配置运行完整`run_discovery`：320个真实芯片位点拟合，加164个真实PTV WGS位点和64个真实常见WGS位点，共228个WGS输入。41,538有效样本；RINT开启/关闭都输出64条single和75条gene结果。与使用同一导出LOCO的原程序比较，single最大LOG10P差1×10⁻⁶，gene最大差4.4×10⁻⁵；gene全部header、ID、NA和四种mask文件逐字节一致。

`run_discovery`最终重跑端到端分别10.89/12.69秒，GPU峰值322,288,128字节；single阶段0.19/0.18秒，gene阶段2.35/5.65秒，其余包括Step1和输入/汇总。重复CLI运行复用三个阶段，仅0.85/0.82秒。以上包含首次CUDA/JIT初始化且共享负载不同，不能与前面的warm成对观测直接比较。修改显著性阈值保留统计缓存；删除gene ID文件后只重算gene并修复ID文件。RINT关闭确实传到Step1、single和gene，不只改变配置文本。

本轮实现和修复记录：

- 按原3.4.1保留缺失表型行、原5-fold边界和held-out模型，修复只筛有效样本后拟合的差异。
- single使用原QT score和χ²尾概率，保留表型及LOCO残差两次尺度；支持FID/IID重排。
- mask先按ALT取max，再处理缺失；VC使用minor方向和原低ALT计数折叠规则，singleton与carrier可独立配置。
- SKAT-O实际最高rho为`.999`；独立GPU Davies保留原误差预算失败、Kuonen、strict Davies、Liu路径，而非仅按P大小切换。
- 共线mask的SBAT pivot依赖原Eigen/SSE2求和顺序。独立Triton kernel保持逐次float64舍入；12个真实范数fixture逐位匹配，最终真实QR选列匹配。没有按基因名或精度模式硬编码保留列。
- RINT可独立设定，并提供全流程关闭开关；JSON配置往返后实际mask构建通过。
- 缓存核对输入、实现、参数与输出身份；输出使用partial提交，Step1双文件失败时恢复旧文件。

最终58个CUDA单元测试通过，覆盖独立ridge直接解、QT闭式解、加权χ²低秩解析/独立积分、冻结Davies数值/故障参考、SKAT-O极小P和缩放、SBAT约束/混合权重、PLINK码与缺失、原格式导出及异常恢复。另以30项冻结C++参考核对Davies值与故障路径。模拟/数学fixture用于回归检验，上述benchmark全部使用真实数据。

完整逆变换优化的逐TEST误差、三次计时和CUDA派发计数见[聚合JSON](../benchmarks/inverse_chi2_optimization_2026-10-04.json)。新增白名单前移与BIM缓存回归检查实际读取列数、BIM顺序、缺失ID、新ID、重复ID、容量上限及文件变化失效。

白名单前移另以同一统计后端做Main/Sub×RINT开/关四组真实配对验证，每组warm后重复三次。Sub每次解码从910列降至288列，Main仍为910列；全部优化前后结果与历史PyTorch输出逐字节相同，四种mask文件也与原程序逐字节相同。每个reader首次查询扫描BIM一次，之后三次查询相同子集为0次；新ID仍需要重扫。共享负载下Main和Sub时间有不同方向的波动，不据此声称额外整体加速。全部观测及读取计数见[IO优化聚合JSON](../benchmarks/gene_io_optimization_2026-10-04.json)。

## 运行边界与输入来源

本包读取现成QC的BED/BIM/FAM和原annotation/setlist/mask/评分白名单。calling、VEP和Table S24注释资源的重新生成没有纳入GPU复现；C/R/UR标签原样保留，不凭名字猜频率域，未核实的JARVIS条件不从文件名推断。raw模式需要显式提供论文协变量，真实验收使用既有残差表，尚未重新核验raw协变量处理对原R流程的全队列差异。

本轮支持连续单表型、常染色体discovery。完整芯片拟合已完成，Step2验证为有界真实子集；没有把验证结果称为全WGS研究重现。软件方法与原命令见[API](API.md)及[官方冻结源码](https://github.com/rgcgithub/regenie/tree/v3.4.1)。
