# WGS GPU association analysis

本仓库提供两个独立的关联分析入口：

| Python模块 | 分析与输出 | 文档 |
|---|---|---|
| `torchwgs` | REGENIE连续单表型discovery：ridge/LOCO、single-variant、gene-based；原格式`.regenie` | [完整流程](docs/REGENIE.md)、[Python参数](docs/API.md)、[真实验证](docs/VALIDATION.md) |
| `staar_phewas` | STAAR / MultiSTAAR PheWAS；原格式`.Rdata` | [完整流程](docs/pipeline.md)、[真实验证](docs/benchmark.md) |

## REGENIE discovery

`torchwgs` 用 PyTorch 计算连续单表型的完整 discovery：芯片 ridge/LOCO、全染色体 single-variant、coding/noncoding gene-based，以及显著性和 locus 汇总。运行时不调用 REGENIE、PLINK 或 R。`WGSConfig.paper()`采用核对后的论文参数，Python/JSON 可以修改；RINT 可设为 `WGSConfig.paper(apply_rint=False)`或 CLI `--no-rint`。

`study_gene_analyses()`展开全部 14 Main + 12 Sub，共 26 组、42 个源 mask 定义，并保留每组的 domain、overall、singleton 和 AAF masks。本次交付使用 `ExecutionConfig(parallel_level="serial", workers=1, max_gpu_gb=20.)`，在 GPU 上串行完成各组；API 另支持在 mask 组或染色体层级使用独立 CUDA streams 并行。完整参数、输入格式、原软件命令和流程图见 [REGENIE指南](docs/REGENIE.md)及 [Python参数](docs/API.md)。

GeneConfig 的 SKAT-O 积分默认采用原 REGENIE 的 χ² 坐标（adaptive_x），绝对/相对误差预算为 10⁻²⁵ / 2⁻¹³、最多 1,000 个区间，四个参数均可修改。平方根坐标 adaptive_sqrt 保留为可选项；真实 9 维 mask 曾出现节点漏过窄特征却报告收敛，原坐标对照及限制见 [真实验证](docs/VALIDATION.md)。

真实验证使用最小的 chr21：13,733,596 个 WGS 位点，41,538 名有效样本。完整 509,468 位点 Step1 已独立对照。当前全串行运行 `discovery_all26_gpu_serial_final_v5` 沿用同一份已冻结的 21 个实现模块，源码通过 SHA-256 核对一致；single 已完成全位点扫描，314,209 行格式及五项数值门槛全部通过，阶段耗时为 424.529767 秒。全部 26 组 gene 和汇总仍在验证；本轮导入已验证的完整 Step1 LOCO，整体耗时、显存与最终验收结果待运行完成后登记。

同源六个完整组的 GPU 串行测量已完成，共 32,842 行，进程墙钟为 6473.804485 秒，pipeline 函数计时为 6470.960217 秒，峰值分配为 6,605,298,688 字节（约 6.15 GiB）。六组结构及附属文件全部通过，严格数值为 0/6 组通过；该测量覆盖六组，不代表全部 26 组的最终串行结果。[匿名计时和显存记录](benchmarks/real_gpu_serial_resources_2026-10-05.json)、[测量图](benchmarks/gpu_serial_2026-10-05.png)仅保留串行结果。

历史 v4 使用 2 workers，已完成 single、全部 26 组 gene 和汇总：single 314,209 行格式及五项数值门槛全部通过；gene 97,222 行、26/26 组结构及附属文件全部通过，严格数值仅 `Splice_splice05` 全组通过（1/26）。ADD-SKATO 38 行、ADD-SKAT 1 行仅 CHISQ 超门槛，LOG10P 均通过；SBAT/POS/NEG 和 GENE_P 仍有 LOG10P 差异。v4 统一 pipeline 函数计时为 10945.423 秒，峰值约 7.49 GiB；这些数值属于历史 2 workers 运行，详情见 [真实验证](docs/VALIDATION.md)。

同一冻结源码的本地与验证服务器 111 项回归均通过，测试报告耗时分别为 119.770 和 54.939 秒。全量 GPU 串行验证尚未完成；共享资源下的不同计时范围不换算受控倍速。

历史 v4 在论文阈值下两边的显著性决策一致，均无显著命中；该[匿名记录](benchmarks/real_discovery_significance_2026-10-05.json)仅覆盖本连续表型 chr21 的无命中场景，尚未验证阳性命中一致性或数值等价。v5 的汇总将随全串行运行重新核验。

上一冻结版 v3 的 single 阶段耗时 225.320409 秒，314,209 行格式与数值均通过；该轮因 Pseudo 多一行 ADD-SKATO 而停止。原生条件 double 概率下溢规则修复后，v4 的 Pseudo 已与原程序同为 4,287 行。single 阶段没有独立进程墙钟或峰值记录，不与旧 CPU 计时换算加速倍数。

```bash
conda env create -f environment.yml
conda activate torchwgs
python -m unittest discover -s tests/regenie_gpu -v
torchwgs --inputs discovery_inputs.json --out /results/discovery --no-rint
```

## STAAR / MultiSTAAR PheWAS

`staar_phewas` 使用 PyTorch float64 复现 STAARpipelinePheWAS 的关联计算，并从原生 GDS 流式读取基因型。R 用于独立验证，生产分析不调用 R。

安装、Python/命令行调用、全部参数、原 R 示例与流程图见 [pipeline 文档](docs/pipeline.md)。真实数据对照范围和误差见 [benchmark](docs/benchmark.md)。

读取、主机准备、小矩阵同步与精度校正的速度成本，以及串行/mask 批量和完整计时口径，见 [性能分析](docs/performance.md)。

单个连续表型的完整染色体入口见 [chromosome](docs/chromosome.md)：全部单变异、7 类 coding、7 类 noncoding 及独立 ncRNA mask，按原完整基因目录与 array ID 生成批次文件；支持 mask 批量 GPU 统计。精度选择及实际 FP32/TF32 探针见 [precision](docs/precision.md)。

0.2.0 已在 42,652 个真实样本、一个连续表型的 chr21 上完成 GPU 串行全流程：221 个 coding、221 个 noncoding、349 个 ncRNA 和四个单变异区段，共 795 项任务。18 份关联文件的 3,343,119 个数值单元格全部通过原 R 的严格容差检查，文件结构、8 次单变异元数据读回及零模型对照均通过；单变异输出共 318,132 行。范围、实测耗时和聚合记录见 [最新 benchmark](docs/benchmark.md)。

正式输出采用 STAAR 原生 `.Rdata` 文件，保留保存对象名、列表层次、混合矩阵、factor 与 `row.names`；[文件格式](docs/r_native_output.md)给出批次命名和 R 读回方式。[联合多表型](docs/multi.md)与[二分类 SPA](docs/binary.md)分别说明相关性模型、原包兼容问题和实际验证范围。

```bash
conda env create -f environments/staar-gpu.yml
conda activate staar-phewas-torch
LZMA_PREFIX="$CONDA_PREFIX" python -m pip install --no-deps --no-build-isolation \
  'git+https://github.com/CoreArray/pygds.git@b7a2dbbebf3b06ac4e97c806e36ec4e1a6af5bdd'
python -m pip install -e .
# 安装并配置独立精度库，命令见下方链接。
# STAAR_REFERENCE_LAPACK_LIBRARY 指向已校验的 libmkl_rt.so。
staar-phewas-torch examples/staar-analysis.json --device cuda --report runs/summary.json
```

保持原版近均值 Saddle 输出的 CUDA 分析还需安装独立的参考 LAPACK prefix，设置 `STAAR_REFERENCE_LAPACK_LIBRARY`；完整命令见 [精度与安装说明](docs/precision.md#安装锁定的参考-lapack)。少量标量权重转换和敏感谱求解使用 CPU，score、协方差和关联检验由 PyTorch CUDA 执行；报告记录这些校正的实际次数和耗时。

示例配置使用占位路径。输入 GDS、表型、亲缘信息和零模型均保存在分析者的私有目录。

从表型表和原生 R 格式的稀疏 GRM 准备输入：见 [样本对齐](docs/prepare.md)。零模型输入、输出和原 R 调用见 [零模型](docs/null_model.md)。
