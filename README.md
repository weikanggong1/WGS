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

原始 GDS 只在一次性缓存准备阶段使用。已有六状态缓存需要补齐独立 metadata 和 `cache_dataset.json` 后复用；[缓存格式](docs/sixstate_cache.md)和[新入口](docs/cache_only_run.md)说明所需内容。旧配置接口保留为 `torchstaar-config`，用于兼容已有脚本。

本版合并了已接受的 Single 优化、缓存协方差和长 mask 数值修复。之前真实 339,013 人 chr21 Single 的作业墙钟为 `2331.276 -> 1125.923 s`，全部概率相对上一 TF32 结果的最大 `|Δ(-log10 P)|` 为 `7.0916163e-6`；官方 R 仅有 28 个概率字段的有界对照。真实 `M=9854` 的协方差阶段预热计时为 `64.938 -> 4.088 s`，U/V 逐位一致；这些记录不包含本版三输入全基因组的端到端精度结论。测量范围和本版验收记录见[版本与基准](docs/hybrid_validation.md)。

源码遵循 GPL-3.0-only。算法、原软件入口和参考文献见[统计说明](docs/statistics.md)及[完整流程](docs/torchstaar.md)。输入、个体模型、实际表型标签、位点/基因结果表和服务器配置应保存在私有目录；公开报告只包含匿名计数、耗时、精度差异和配置。
