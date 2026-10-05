# FP32、TF32 和原 R 精度

原参考 R/STAAR 使用 double：零模型数值向量、密集矩阵和稀疏 Matrix 的 `x` slots 均为双精度，C++ 内核也使用 double。因此原版 R 结果的参照是 FP64；当前没有将 R 原版称为 float32 实现。

为判断哪些步骤可降精度，真实 42,652 样本的三个集合固定同一 FP64 零模型，只将 score/协方差计算分别改为 FP32 和允许 TF32 的 FP32 矩阵乘法，再将输入转回 FP64 计算尾概率。每集有 38 个 P 值字段，使用已验证的同输入 FP64 路径作为直接参照。容差保持 `abs(error)<=1e-10+1e-7*abs(reference)`。

| 真实集合 | 变异数 | FP32 最大 P 值绝对差 | 允许 TF32 最大 P 值绝对差 | 严格验收 |
|---|---:|---:|---:|---|
| GENE_A plof | 3 | 6.42×10⁻⁸ | 3.26×10⁻⁶ | 两种模式均未通过 |
| GENE_A missense | 93 | 1.85×10⁻⁴ | 5.24×10⁻⁴ | 两种模式均未通过 |
| GENE_B synonymous | 44 | 1.061×10⁻² | 5.473×10⁻³ | 两种模式均未通过 |

GENE_B 的协方差满秩，最小特征值约 0.997；较大的 P 值差来自接近谱均值时原 Saddle 公式对舍入的敏感性。把尾概率恢复成 FP64，仍不能消除输入 score/协方差已丢失的精度。

生产计算因此保留 FP64，速度优化优先针对注释重复扫描、基因型读写、CUDA 同步、批量核与无需计算的协方差元素。已有 A100 score/协方差专项 probe 出现 `cutlass_80_tensorop_d884gemm`，该探针使用了双精度 Tensor Core；这不表示保持原稀疏相加顺序的完整生产链也使用同一矩阵乘法。允许 TF32 的 probe 出现 float Tensor Core 内核；模式由 PyTorch 的 `matmul.allow_tf32` 控制，不能仅凭内核名称中有无 `tf32` 字样判断。

完整数值及探针范围见 [precision-fp32-tf32.json](../validation/precision-fp32-tf32.json)。其中事件时间仅是预热后的 score/协方差计算，不包含传输、GDS、拟合和统计检验，且设备共享；不据此报告完整 pipeline 加速比。

PyTorch 的 [TF32 说明](https://docs.pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices) 解释了 FP32 矩阵乘法的 TF32 控制。保持原统计精度的批量方法见 [statistics](statistics.md)。

## 保持原 Saddle 舍入的精度边界

全染色体检查发现，原 Saddle 在根接近零、但尚未进入原 moment 分支时，会把少数几个 ULP 的权重或特征值差放大到超过上述容差。改用更稳定的公式仍会改变原版输出。因此当前实现保留原公式，并在两个小范围使用 CPU 精度校正：

1. 注释权重及原始 Local Diversity 的互补 PHRED 使用标量 float64 `libm`，保留原 `exp`、`pow`、`log` 和除法次序。调用者已经提供的完整 PHRED 列保持原值。转换后的权重回到输入设备；score、协方差和关联检验继续在 CUDA 计算。
2. CUDA 先求加权协方差的谱，用单调 K1 的 `[-0.01-1e-8, 0.01+1e-8]` 区间筛选近均值行。仅这些行的特征值使用与参考一致的 MKL 2019.2 `DSYEV(N,U)`，工作区固定为原 RcppArmadillo 0.10.8.1.0 的 `66×矩阵阶数`，不查询更大的工作区。输入保留原协方差上三角，不以平均上下三角改变原舍入。特征值返回 CUDA，二分搜索、moment 分支及尾概率仍由 Torch 计算。原 moment 阈值 `1e-4`、谱截断与二分精度 `1e-8` 均未改动。

该路径不调用 R。精度校正的 CPU 边界在运行报告中可见：`annotation_weight_execution` 记录转换次数、逻辑传输字节及转换时间，`precision_eigen_execution` 记录实际 CPU 谱次数、求解与传输总时间、库版本/哈希、CUDA 谱次数及 `workspace_policy="fixed_66n_original_armadillo"`。传输总时间包含同步和初始化，不能作为纯内核时间。

## 安装锁定的参考 LAPACK

Linux x86-64 上将精度库安装到独立 Conda prefix。主 Python 环境仍使用 PyTorch 自己的依赖。下面的显式锁文件指向原作者发布的三个包；仓库不包含 Intel 二进制。

```bash
# 在仓库根目录执行；精度库 prefix 放在仓库之外。
conda create --prefix ../staar-reference-lapack \
    --file environments/reference-lapack-linux-64.lock --yes
export STAAR_REFERENCE_LAPACK_LIBRARY="$(realpath ../staar-reference-lapack/lib/libmkl_rt.so)"
export LD_LIBRARY_PATH="$(dirname "$STAAR_REFERENCE_LAPACK_LIBRARY")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MKL_THREADING_LAYER=SEQUENTIAL
export MKL_NUM_THREADS=1
sha256sum "$STAAR_REFERENCE_LAPACK_LIBRARY"

# 主环境中的 Python 执行分析，不调用参考库 prefix 的 Python。
python -m staar_phewas.chromosome analysis.json --device cuda:0 \
    --report runs/full_chr21/report.json
```

验证的 `libmkl_rt.so` SHA-256 为 `f66c03e52d8a9e0102c515c148cf36941016c6dcdd8e38f20a9f56ab9b7f2fd8`。程序在首次实际精度校正时校验库内容；没有配置、库不匹配或加载失败均明确报错。该锁定只覆盖已验证的参考版本，不据此声称与其他 R/BLAS 版本逐位一致。安装包遵循其各自的 Intel 许可。
