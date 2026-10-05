# WGS GPU association analysis

本仓库提供两个独立的关联分析入口：

| Python模块 | 分析与输出 | 文档 |
|---|---|---|
| `torchwgs` | REGENIE连续单表型discovery：ridge/LOCO、single-variant、gene-based；原格式`.regenie` | [完整流程](docs/REGENIE.md)、[Python参数](docs/API.md)、[真实验证](docs/VALIDATION.md) |
| `staar_phewas` | STAAR / MultiSTAAR PheWAS；原格式`.Rdata` | [完整流程](docs/pipeline.md)、[真实验证](docs/benchmark.md) |

## REGENIE discovery

`torchwgs` 用 PyTorch CUDA 执行连续单表型的完整 discovery：芯片两级 ridge/LOCO、全染色体 single-variant、coding/noncoding gene-based，以及显著性和 locus 汇总。运行时不调用 REGENIE、PLINK 或 R。

`study_gene_analyses()`提供全部 14 Main + 12 Sub，共 26 组、42 个基础 mask 定义；每组保留 category、domain、overall、singleton 和 AAF masks。完整流程采用 GPU 串行配置 `ExecutionConfig(parallel_level="serial", workers=1, max_gpu_gb=20.)`，默认导出原格式关联结果和全部 mask 文件。

`WGSConfig.paper()`采用核对后的论文参数，Python 和 JSON 可修改全部配置。RINT 是选项：`WGSConfig.paper(apply_rint=False)` 或 CLI `--no-rint` 可统一关闭。Step1/single 默认 float32/TF32，gene 的协方差与推断采用 float64。详见[完整流程和全部 mask](docs/REGENIE.md)、[输入与参数](docs/API.md)、[真实验证](docs/VALIDATION.md)。

### 安装与完整运行

需要 Linux、CUDA GPU 和 Triton；Conda 环境包含全部运行依赖。

```bash
conda env create -f environment.yml
conda activate torchwgs
python -m unittest discover -s tests/regenie_gpu -v

python examples/configure_discovery.py \
  --array-prefix /data/array/genotype_array --array-variant-include /data/array/qc.snplist \
  --phenotype-file /data/phenotype_residuals.txt --phenotype-column trait_01 \
  --discovery-samples /data/discovery.keep --sample-remove /data/sample_exclude.txt \
  --wgs-prefix-template '/data/wgs/chr{chromosome}' \
  --annotation-root /data/annotation --json-directory /results/config \
  --max-gpu-gb 20
torchwgs --inputs /results/config/discovery_inputs.json \
  --config /results/config/analysis_config.json --out /results/discovery_trait_01
```

以上配置读取 22 条常染色体的完整输入，每条染色体包含全部 26 组。先分析 chr21 时，在配置生成命令加 `--chromosomes 21`。私有表型、样本、基因型、注释和运行结果保存在分析者自己的目录；公开示例仅使用通用路径。

### 当前真实验证

验证范围是最小的 chr21：13,733,596 个 WGS 位点、41,538 名有效样本。完整 509,468 位点 Step1 已独立对照；当前完整 GPU 串行运行导入该 LOCO，single 已输出 314,209 行，格式和五项数值门槛全部通过，阶段耗时 424.529767 秒。全部 26 组 gene 与汇总尚未完成最终验收。

同源六个完整组的 GPU 串行测量已完成，共 32,842 行，进程墙钟 6473.804485 秒，pipeline 函数计时 6470.960217 秒，峰值分配 6,605,298,688 字节（约 6.15 GiB）。六组结构和附属文件全部通过，严格数值为 **0/6 组通过**；完整 gene 数值等价尚未达到。六组测量不能代替全 26 组验收。计时属于共享 A100 环境，不换算受控 CPU/GPU 加速倍数。

[匿名串行资源记录](benchmarks/real_gpu_serial_resources_2026-10-05.json)、[资源图](benchmarks/gpu_serial_2026-10-05.png)、[资源表](benchmarks/gpu_serial_2026-10-05.csv)保留实际计时、显存和验收数量。此次串行入口清理包含 21 个实现模块，本地 111 项回归全部通过，框架报告耗时 64.704 秒。上述真实测量来自已冻结的统计实现，入口清理未新增完整端到端测量；回归 fixture 用于检查算法，不替代真实 benchmark。

原算法链接与参考文献见[指南](docs/REGENIE.md#数值后端与参考)。

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
