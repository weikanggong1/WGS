# Torchstaar

Torchstaar 用 PyTorch GPU 完成 Single、coding、noncoding 和 ncRNA 关联分析。新版入口只接收两份 CSV 和一个已转存的缓存目录，按 `eid` 对齐样本，为每个表型列拟合独立 Gaussian 零模型，再运行全部缓存染色体。关联分析不需要原始 GDS、PyGDS 或 R。

```python
from torchstaar import run

if __name__ == "__main__":
    report = run(
        phenotype_csv="private/phenotypes.csv",   # eid 后每列是一个命名表型
        covariate_csv="private/covariates.csv",    # eid 后每列是需要回归的协变量
        cache_directory="private/population_cache", # 基因型、样本索引、注释和基因目录
        output_directory="private/results",      # 可选；默认在表型文件旁新建目录
    )
```

```bash
conda env create -f environment.yml
conda activate torchstaar
python -m pip install -e .
torchstaar private/phenotypes.csv private/covariates.csv private/population_cache \
  --output-directory private/results
```

第一列必须名为 `eid`，采用标准正整数文本；其他列必须有唯一名称并包含数值。每个表型分别删除缺失观测，任何协变量缺失则该样本不参与相应分析。程序加入一个截距，CSV 已有全一截距时不会重复加入。缓存目录包含全部运行元信息，无需另传注释、基因目录、模型或 GDS 路径。三输入的具体格式、全部超参数和输出结构见[运行指南](docs/cache_only_run.md)。

默认最多使用 8 张 GPU，每个 worker 的显存上限为 **40 GiB**。关联矩阵乘法采用 TF32、FP32 存储；零模型拟合显式采用 FP64。`M <= 5000` 保留成熟完整谱和原 Saddle/矩匹配尾概率，`M > 5000` 使用 FastSKAT 风格前 512 个特征值与残差谱的两矩匹配，seed 为 1729、probes 为 0，并标记 `approximate=true`。长 mask 协方差默认采用 4096 缓存分块，按预算自动选择完整驻留或双 panel；score/投影预处理仍为 512。Single 保留 1024 有效变异批宽、MAC 20 和原 5000 变异分组。精度不会因内存不足自动降为 float16/BF16。

GPU 主进程默认各用 1 个 CPU 计算线程；准备进程数默认为 `auto`，按全部 worker 共享的有效 CPU 额度分配，并扣除加载线程、4 核后台预算和 4 核预留。40 核配额与 8 张卡时分配为 8 个主线程、8 个加载线程和 16 个准备子进程。主机预算默认 200 GiB，在实际 cgroup 上限内保留 20 GiB 余量；64 GiB compact 磁盘缓存和每 worker 256 MiB 已准备队列是独立预算，后者不代表总 RSS。

原始 GDS 只在一次性缓存准备阶段使用。已有六状态缓存需要补齐独立 metadata 和 `cache_dataset.json` 后复用；[缓存格式](docs/sixstate_cache.md)和[新入口](docs/cache_only_run.md)说明所需内容。旧配置接口保留为 `torchstaar-config`，用于兼容已有脚本。

同一批样本可通过按来源、完整样本轴与请求变异绑定的 compact 缓存复用准备结果。默认共享磁盘预算为 64 GiB，CPU 预取队列每 worker 最多 2 条，字节预算为 256 MiB，单条超预算等边界见运行指南；设 `cohort_cache_gib=0` 或 `prefetch_depth=0` 可分别关闭。缓存保留原稀疏六状态与精确统计，不存 dense 矩阵或模型结果；新 subset 自动使用不同条目。gene masks 共用并集，Single 保留染色体顺序。首次构建和命中复用的时间分别记录，说明见[运行指南](docs/cache_only_run.md)。

0.6.0 修复了 portable gene reader 回落 CPU 的能力识别，并加入 compact 复用及多进程 CPU 预取。最终冻结实现 `51810fd` 使用真实 339,013 人、同一个由三输入新拟合的模型，在同一 GPU 顺序重算旧 TF32 参考、首次构建和缓存命中三组。两组候选共比较 44,600 个 P/logP 值，最大 `|Δ(-log10 P)|=0`，原生结构和非概率字段一致；相同源码的 GPU 组件为 275 项测试和 33 项子测试通过。

| 选定作业 wall | 本轮旧参考 | 首次构建 | 缓存命中 |
|---|---:|---:|---:|
| Single 区间，11,063 行 | 50.140 s | 46.927 s | 34.537 s |
| coding，5 行 | 18.636 s | 9.461 s | 7.378 s |
| noncoding，6 行 | 115.758 s | 105.179 s | 91.665 s |
| ncRNA，M=9854 | 144.430 s | 101.562 s | 87.052 s |
| 整组启动至完成，不含模型拟合 | 342.114 s | 275.148 s | 228.534 s |

本轮候选为单卡、1 个主线程、1 个读取线程和 4 个准备子进程；生产 8 卡默认使用上文的全局 CPU 分配。271 个 compact 条目在暖组全部命中，人口 frame 读取从冷组 300 次降为 0。预飞保持 27 GiB 预算，生产上限为 40 GiB。冷暖仅指派生缓存，共享 GPU 负载和 OS page cache 未受控；gene 的改善也包含 GPU 准备入口修复。完整来源、分步骤和历史线程版记录见[最终匿名基准](benchmarks/torchstaar_cohort_compact_process_preflight_2026-10-09.json)及[版本与基准](docs/hybrid_validation.md)。官方 R 与全基因组验收仍需分别完成。

本版合并了已接受的 Single 优化、缓存协方差和长 mask 数值修复。之前真实 339,013 人 chr21 Single 的作业墙钟为 `2331.276 -> 1125.923 s`，全部概率相对上一 TF32 结果的最大 `|Δ(-log10 P)|` 为 `7.0916163e-6`；官方 R 仅有 28 个概率字段的有界对照。真实 `M=9854` 的协方差阶段预热计时为 `64.938 -> 4.088 s`，U/V 逐位一致；这些记录不包含本版三输入全基因组的端到端精度结论。测量范围和各版验收记录见[版本与基准](docs/hybrid_validation.md)。

源码遵循 GPL-3.0-only。算法、原软件入口和参考文献见[统计说明](docs/statistics.md)及[完整流程](docs/torchstaar.md)。输入、个体模型、实际表型标签、位点/基因结果表和服务器配置应保存在私有目录；公开报告只包含匿名计数、耗时、精度差异和配置。
