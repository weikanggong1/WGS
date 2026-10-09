# Torchstaar

Torchstaar 用 PyTorch CUDA 实现 STAAR 的 Single、gene-based coding、noncoding 和 ncRNA 分析，输出原生 `.Rdata/.rds`。生产流程使用 TF32 矩阵乘法和 FP32 累加，默认一个 GPU 串行执行，显存预算 20 GiB；关联计算和文件写出不调用 R。

完整染色体入口支持一个连续 Gaussian 表型，包含全部 7 个 coding mask、7 个 noncoding mask 和 ncRNA。固定窗口、滑动窗口分析不在当前范围。已有六状态基因型缓存可以直接复用；各模型使用其可用观测，不填补缺失表型或协变量。

## 安装与运行

需要 Linux、支持 TF32 的 NVIDIA CUDA GPU、编译器和 CoreArray PyGDS。

```bash
conda env create -f environment.yml
conda activate torchstaar
LZMA_PREFIX="$CONDA_PREFIX" python -m pip install --no-deps --no-build-isolation \
  'git+https://github.com/CoreArray/pygds.git@b7a2dbbebf3b06ac4e97c806e36ec4e1a6af5bdd'
python -m pip install -e .

# 填写私有输入、完整有序目录和输出配置；先展开计划。
torchstaar-chromosome private/chromosome.json --plan-only --report private/plan.json
# 原 GDS 路线；已有六状态缓存使用下方 Python 缓存入口。
torchstaar-chromosome private/chromosome.json --device cuda:0 --report private/report.json
```

Conda recipe 使用 Python 3.10、PyTorch 2.5.1/CUDA 11.8、Triton 3.1 和 NumPy 1.26。本次真实 benchmark 使用已有环境，不包含全新 Conda 安装验收；安装后核对 PyGDS/SDK 和 CUDA。输入、每个参数、Python 缓存调用和原 R 命令见[完整指南](docs/torchstaar.md)。

## 本版真实验证

版本 0.4.0 优化了样本轴验证、解码器行映射、安全 MAC 预筛、有效变异合批和显存预检。统计公式与已接受版本的零模型实现保持一致；不使用 TF32 分量重建或 FP64 矩阵回退。

匿名真实 chr21 Single 对照使用固定零模型、339,013 个有效样本，完整扫描 13,733,596 个输入，输出 1,065,735 行、4 个原生文件。读取和计算的 Single 作业合计由 **2331.276 秒降至 1125.923 秒**，本版启动至文件输出为 **1183.551 秒**。两次启动及文件系统缓存状态不同；这是实测观测，不能解释为隔离条件下的普遍倍速。

全部输出的结构、顺序、factor 属性、行名及 AF/MAF/N 与上一接受的 TF32 版本一致。55,377 个显著联合结果的最大 `−log10(P)` 差异为 **7.0916163e-6**，显著阈值跨越 0 项。官方 R 的同模型有界对照另包含 28 个 P、2 个显著项，最大显著误差 **2.3897789e-6**。全量数值参考为上一 TF32 结果；本版未重新执行全染色体 R。峰值 allocated/reserved 为 **5.137/8.287 GiB**。

这次完整测量只覆盖 Single；coding、noncoding 和 ncRNA 的当前耗时不能从该结果推导。分步骤、范围和近期记录见[真实验证](docs/torchstaar.md#真实验证与计时范围)，机器可读记录见[匿名汇总](benchmarks/torchstaar_single_chr21_2026-10-09.json)。

## 文档

| 内容 | 说明 |
|---|---|
| [完整流程](docs/torchstaar.md) | 流程图、全部 mask、配置、Python/CLI、原 R 与真实验证 |
| [输入对齐](docs/prepare.md)、[Gaussian 零模型](docs/null_model.md) | 可用样本、协变量、稀疏亲缘矩阵与模型缓存 |
| [GDS](docs/gds.md)、[packed 读取](docs/gds_packed.md)、[六状态缓存](docs/sixstate_cache.md) | 输入格式、读取与既有缓存复用 |
| [统计 API](docs/statistics.md)、[原生 R 输出](docs/r_native_output.md) | Score/协方差、完整谱、尾概率和文件类型 |
| [独立多表型](docs/phewas_multiple.md)、[联合多表型](docs/multi.md)、[二分类](docs/binary_null.md) | 独立函数/FP64 对照接口；不属于本版 Single TF32 benchmark |

源码遵循 GPL-3.0-only；原算法和文献见[参考](docs/torchstaar.md#参考)。个体输入、模型、表型标签、实际位点/基因表与服务器配置均保存在分析者自己的私有目录，公开仅提供通用示例和匿名汇总。
