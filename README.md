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

当前 v13 包含 23 个实现模块。本地 CUDA 回归 239/239 项通过，零跳过，框架计时 87.197 秒；共享 A100 上核对冻结源码、实际导入路径与测试摘要后，按 GPU 串行分三组通过全部 239 项。分组框架计时为 43.468、50.353 和 77.908 秒，这些是回归测试时间。

真实前100,000个位点、41,538个有效样本的相同切片中，动态列宽解码把新Triton缓存下的single阶段从26.08047降至5.59638秒，kernel缓存文件从35降到2；新版本热缓存为3.51617秒。1,302行结果与旧PyTorch逐字节相同，并通过原REGENIE五项数值门槛。探针逐阶段同步、输入存储缓存未隔离，这些数字只对应该切片。

真实范围为 chr21 的 13,733,596 个 WGS 位点、41,538 名有效样本。历史版本的完整 single 已输出 314,209 行，格式和五项数值门槛通过，阶段耗时 424.529767 秒。此前 v12 的全染色体 single 探针触及 180 秒上限，进程观测 182.243 秒后以超时代码 124 结束，私有部分输出有 41,844 行；它尚未完成完整文件及原 REGENIE 数值验收。旧完整 gene 运行已停止于 13/26 组，后续完整候选队列也已停止。完整 single、26 组 gene、全部 mask 附件与 Summary 的五分钟目标尚未达成。

[历史匿名串行资源记录](benchmarks/real_gpu_serial_resources_2026-10-05.json)、[资源图](benchmarks/gpu_serial_2026-10-05.png)和[资源表](benchmarks/gpu_serial_2026-10-05.csv)覆盖同源六个完整 gene 组，共 32,842 行：进程墙钟 6473.804485 秒、pipeline 函数计时 6470.960217 秒、峰值分配约 6.15 GiB。六组结构和附属文件通过，严格数值仍为 **0/6 组通过**；该记录不能代替当前版本或全 26 组验收。

四个真实、完整样本的有界 mask 统计探针显示：v7 → v9 更换为 QAGS 后，四例统计调用的中位耗时之和从 11.15249 降至 4.47655 秒，积分节点总数从 6,426 降至 1,008；SKAT-O 的 CHISQ 与 LOG10P 同时通过原门槛的案例数从 0/4 变为 4/4。这是不同进程、共享 A100 上复用既有 score/covariance 的局部观测，未计入基因型读取、mask 准备或完整 gene/pipeline。v11 fused 探针降低了统计计算峰值分配，尚未显示稳定计时收益；v12同进程每种后端各五轮交替对照已通过四例原数值门槛；四例中位耗时合计torch为3.14907秒、fused为3.44327秒，融合未显示普遍更快，v13默认保留torch。匿名计数、误差与软件摘要见[本轮优化记录](benchmarks/regenie_bounded_optimization_2026-10-05.json)。

完整 509,468 位点 Step1 的历史独立对照、当前优化范围、原数值门槛与各项限制见[真实验证](docs/VALIDATION.md)。原算法链接与参考文献见[指南](docs/REGENIE.md#数值后端与参考)。

## STAAR / MultiSTAAR PheWAS

`staar_phewas` 从原生 GDS 读取基因型，以 PyTorch GPU 实现 STAAR 原零模型和关联公式，生产分析不调用 R。已发布 0.2.0 的 FP64 版本在真实单连续表型 chr21 上完成全部 795 项串行分析和原 R 输出对照，见 [benchmark](docs/benchmark.md)。

本版使用原生 TF32 / FP32，移除多分量精度重建。覆盖 Single、全部 coding/noncoding mask 与 ncRNA，保留原 `.Rdata/.rds` 文件名、类型、属性及顺序；固定窗口和滑动窗口已移除。各任务串行，任务内部使用 GPU 矩阵批处理、完整谱和局部中间结果复用；生产关联不调用 R。

真实 chr21 的 795 项任务已完成：19 份原生文件结构与严格零模型比较通过，478,082 个 P 全部有效、可比较。以原 R 或本版任一 `P<0.05` 的联合范围验收，24,713 个 P 的 `-log10(P)` 误差全部 <=0.001，最大为 `0.0003579714`；其余 P 的误差继续保存作诊断。固定零模型和已有转存缓存上的公开代码复验墙钟为 **419.926 秒（约 7 分钟）**；带细分 profiler 的 F 初轮为 429.097 秒，每染色体 300 秒的速度目标尚未达到。首次转存另需 10,854.84 秒、新增存储约为原 GDS 的 8.53%；该次测量使用共享 A100、暖文件系统及 Triton 缓存。详见 [TF32 benchmark](docs/tf32_benchmark.md)和[匿名完整记录](benchmarks/staar_chr21_native_tf32_2026-10-06.json)。

```bash
conda env create -f environments/staar-gpu.yml
conda activate staar-phewas-torch
LZMA_PREFIX="$CONDA_PREFIX" python -m pip install --no-deps --no-build-isolation \
  'git+https://github.com/CoreArray/pygds.git@b7a2dbbebf3b06ac4e97c806e36ec4e1a6af5bdd'
python -m pip install -e .
# 配置保存在私有目录，matmul_mode 为 tf32。
staar-phewas-chromosome private/full_chromosome.json --device cuda --report private/summary.json
```

[完整流程](docs/pipeline.md)、[全部输入格式和参数](docs/tf32_pipeline.md)、[染色体任务](docs/chromosome.md)、[零模型](docs/null_model.md)、[样本对齐](docs/prepare.md)、[无损剂量缓存](docs/sixstate_cache.md)、[矩阵复用](docs/matrix_reuse.md)、[统计公式](docs/tf32_statistics.md)、[原生输出](docs/r_native_output.md)与[logP 验收](validation/README_logp.md)给出 Python/CLI、原 R 调用和参考文献。[多表型](docs/multi.md)及[二分类](docs/binary.md)保留各自实际验证范围。

公开配置使用占位路径；真实基因型、表型、亲缘信息和分析结果保存在私有目录。原生 TF32 无需旧参考 CPU LAPACK 精度库。
