# WGS GPU association analysis

本仓库提供两个独立的关联分析入口：

| Python模块 | 分析与输出 | 文档 |
|---|---|---|
| `torchwgs` | REGENIE连续单表型discovery：ridge/LOCO、single-variant、gene-based；原格式`.regenie` | [完整流程](docs/REGENIE.md)、[Python参数](docs/API.md)、[真实验证](docs/VALIDATION.md) |
| `torchstaar` | Torchstaar：完整染色体Single/15类基因mask；原格式`.Rdata/.rds` | [完整指南与真实验证](docs/torchstaar.md) |

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

## Torchstaar

Torchstaar 使用PyTorch CUDA完成原GDS读取、连续Gaussian零模型、Single、全部coding/noncoding/ncRNA共15类mask与原生R文件输出。默认TF32/FP32，一个GPU串行执行任务，保留完整谱和原STAAR检验规则，显存预算20 GiB，生产运行不调用R；没有固定窗口或滑动窗口分析。用户Python入口为 `torchstaar`，与REGENIE的 `torchwgs` 独立。

当前公开科学代码已完成真实chr21全部795项、进程墙钟 **296.196098秒**；19文件结构、strict null、全部478,082 P有效/可比较及24,713显著联合P的独立原R验收全部通过，0超限，最大logP差0.0003579714。此前H2候选299.009秒与公开F419.926秒保留为近期对照，不能替代当前公开版本验收。已有转换缓存/固定null、暖文件系统和Triton缓存的测量边界、共享节点与转换3.02小时成本均单独记录。默认decoded LRU64 MiB，512MiB候选未显示完整收益；不启用私有RAM预载、counts seed或XDR候选。

```bash
conda env create -f environment.yml
conda activate torchwgs
LZMA_PREFIX="$CONDA_PREFIX" python -m pip install --no-deps --no-build-isolation \
  'git+https://github.com/CoreArray/pygds.git@b7a2dbbebf3b06ac4e97c806e36ec4e1a6af5bdd'
python -m pip install -e .
# JSON使用本人的私有输入和完整manifest；后端默认auto。
torchstaar-chromosome private/full_chromosome.json \
  --device cuda:0 --report private/analysis_report.json
```

主页Conda recipe采用PyTorch2.5.1/CUDA11.8/torchtriton3.1/NumPy1.26和显式zstandard；Python3.10/rdata1.1.0/editable安装保持原配置。[官方历史安装页](https://docs.pytorch.org/get-started/previous-versions/#v251)提供该PyTorch/CUDA组合。小谱后端只用标准库调用已有官方CUDA库，无额外后端依赖；该recipe尚未独立创建并完成数值验收，真实benchmark使用既有服务器环境。

[完整Torchstaar指南](docs/torchstaar.md)按功能、流程图、完整Python示例、每项输入/输出/参数、CLI、原R、真实端到端/阶段对照、近期版本和文献组织。[匿名完整记录](benchmarks/torchstaar_chr21_2026-10-06.json)与[独立验证规则](validation/README_logp.md)分别保存实际结果和验收定义。必要专门API：[输入对齐](docs/prepare.md)、[零模型](docs/null_model.md)、[统计/完整谱](docs/statistics.md)、[GDS](docs/gds.md)、[六状态缓存](docs/sixstate_cache.md)、[原生输出](docs/r_native_output.md)。所有个体输入、实际gene/位点表与表型标签保存在私有目录，公开只发布匿名汇总。
