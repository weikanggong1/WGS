# 原生 GDS 读取和多表型样本对齐

## 1. 功能

`SeqArrayGDS` 直接以只读方式打开 CoreArray/SeqArray GDS，读取选定样本和位点。计算过程中不启动 R。读取器按 `genotype/@data` 重建每个位点的 Bit2 编码，保留多等位位点，识别当前编码层数对应的缺失值；读取 QC 时恢复 factor 标签，读取变长 INFO 时按索引恢复各位点的数组。

多表型分析先在所有表型的样本并集确定 minor allele 方向，再为每个表型提取自己的样本、计算 MAF 和填补缺失值。单个表型提取后不会再次翻转 allele。这对应 STAARpipelinePheWAS 的读取顺序。

```mermaid
flowchart LR
  A[原生 GDS] --> B[筛选位点和样本并集]
  B --> C[按 Bit2 索引解码 REF dosage]
  C --> D[并集 AF 决定 minor 方向]
  D --> E[稀疏 COO 保留显式缺失]
  E --> F[按当前表型顺序取样本]
  F --> G[重新计算 MAF 和处理缺失]
  G --> H[float64 分块基因型]
```

GDS 是 CoreArray 格式。它需要原生 GDS 解码器；不能用 HDF5 读取器打开。

## 2. Python 调用、输入和输出

输入为一个 SeqArray GDS 文件，以及零起始的位点索引和样本索引。数组中索引必须唯一；返回顺序与请求顺序一致。基因型计算限定二倍体。文件中的一维编码索引可以整体读入，样本 × 全部位点矩阵不会整体读入。

```python
import numpy as np
from staar_phewas.gds import SeqArrayGDS

genotype_path = "cohort.gds"  # 用户已处理的原生 GDS
sample_ids = np.loadtxt("phenotype_sample_ids.txt", dtype=str, ndmin=1)
variant_indices = np.arange(100, 356, dtype=np.int64)  # 零起始位点索引

with SeqArrayGDS(genotype_path) as genotype_reader:
    union_sample_indices = genotype_reader.sample_indices(sample_ids)
    quality_labels = genotype_reader.read_field(
        "annotation/info/QC_label", variant_indices
    )
    variant_indices = variant_indices[quality_labels == "PASS"]
    minor_block = genotype_reader.minor_block(
        variant_indices, union_sample_indices
    )
    # 此例单表型使用全部并集样本；多表型可传任意子集及顺序。
    phenotype_rows = np.arange(len(union_sample_indices), dtype=np.int64)
    genotype, maf, mac, missing_counts, minor_is_alt = minor_block.trait_dense(
        phenotype_rows, imputation="mean"
    )
    # genotype: [当前表型样本数, 当前块位点数]，float64，缺失已填补。
    # maf: 每个位点当前表型的 minor allele frequency。
    # mac: 填补前、非缺失样本的 minor allele count。
    # missing_counts: 每个位点缺失的当前表型样本数。
    # minor_is_alt: 并集方向是否为 ALT（与 REF 不同的全部 allele 合并）。
```

各接口和参数：

| 接口 | 参数和返回值 |
|---|---|
| `SeqArrayGDS(path)` | `path` 为原生 GDS 路径；只读打开；支持 `with` 自动关闭。 |
| `n_samples`, `n_variants`, `ploidy` | 文件的样本数、位点数和固定基因型倍性。 |
| `sample_ids()` | 返回文件顺序的样本 ID 数组；调用者负责保留在本地数据环境。 |
| `sample_indices(sample_ids)` | 按请求的 ID 顺序返回文件中的零起始样本索引；不存在或重复的 ID 报错。 |
| `describe(path)` | 返回节点的维度、存储类型、压缩和属性，不返回载荷。 |
| `read_field(path, variant_indices=None)` | `path` 为数组节点；省略索引则读取整个该字段。固定字段返回位点位于轴 0 的数组；变长 INFO 返回每个位点一个数组的列表，长度 0 返回空数组。 |
| `read_field("$ref"/"$alt"/"$num_allele", indices)` | 返回 REF、完整逗号分隔 ALT 字符串或 allele 数量，不将多等位位点拆成新位点。 |
| `read_genotype(variant_indices, sample_indices)` | 返回 `[位点, 样本, 倍性]` 整数 allele code，缺失为 `-1`。 |
| `read_ref_dosage(variant_indices, sample_indices)` | 返回 `[样本, 位点]` float64 REF 拷贝数，缺失为 NaN；对应 SeqArray 的 `$dosage`。 |
| `minor_block(variant_indices, union_sample_indices)` | 在并集样本确定方向，返回 `SparseMinorBlock`。COO 包含非零 dosage 和显式 NaN；未存储元素为零。 |
| `iter_minor_blocks(indices, union_samples, block_size=256)` | 按请求顺序逐块提取；`block_size` 必须为正整数。可使用 128 或 256 控制中间矩阵大小。 |
| `SparseMinorBlock.trait_dense(trait_rows, imputation="mean")` | `trait_rows` 是并集样本列表中的零起始行，按当前 null model 顺序提供。`mean` 用当前表型非缺失样本的 `2*MAF` 填补，保留原 MAF；`minor` 填零并以全部当前表型样本计算 MAF。返回上例中的五项。 |
| `SparseMinorBlock.to_torch_sparse(device="cpu")` | 返回指定设备的 float64 COO tensor，保留显式 NaN；调用统计计算前须按表型处理缺失。 |

QC 节点由数据配置指定。应先核对实际字段，不能仅因节点存在便认为其值可用于筛选。`read_field` 对 folder 报错；本模块尚未提供 FORMAT 数据提取接口。

## 3. 命令行和安装

只检查维度和字段结构：

```bash
python -m staar_phewas.gds --gds cohort.gds --node annotation/info/QC_label
```

原生依赖固定为 CoreArray/pygds commit `b7a2dbbebf3b06ac4e97c806e36ec4e1a6af5bdd`。Conda 环境可提供 Python、NumPy、C++ 编译器和 liblzma 后构建该依赖：

```bash
conda create -n staar-gds -c conda-forge python=3.12 numpy=1.26 pip setuptools wheel xz cxx-compiler
conda activate staar-gds
LZMA_PREFIX="$CONDA_PREFIX" python -m pip install --no-deps --no-build-isolation \
  'git+https://github.com/CoreArray/pygds.git@b7a2dbbebf3b06ac4e97c806e36ec4e1a6af5bdd'
```

直接从固定官方来源安装：PyPI 同名 `pygds` 项目是另一用途的软件。读取器会检查 `pygds.gdsfile`，发现错误依赖时明确报错。原生解码依赖采用 GPL-3；安装后保留它的许可证和来源信息。

## 4. R 原软件对应调用

```r
library(SeqArray)
genotype_file <- seqOpen("cohort.gds")
seqSetFilter(genotype_file, variant.id = selected_variant_ids,
             sample.id = phenotype_sample_ids)
reference_dosage <- seqGetData(genotype_file, "$dosage")
quality_labels <- seqGetData(genotype_file, "annotation/info/QC_label")
seqClose(genotype_file)
```

`$dosage` 是 REF 拷贝数。表型子集处理采用 STAARpipelinePheWAS 的 `Genotype_sp_extraction`、`Missing_num.sp` 和逐表型 mean/minor 填补规则。当前验证使用 R `gdsfmt` 直接读取同一 GDS 节点作为独立解码参照。

## 5. 真实数据对照

2026-10-04 对一份已处理的真实染色体 GDS 验证。文件有 345,967 个样本、14,866,221 个位点，固定倍性为 2；基因型为 `Bit2 + ZIP_RA`，三个位置使用两层 Bit2 编码。选取 2,079 个样本和 35 个位点，使用逆序请求检验行列顺序，包含全部三个多层位置。

| 对照项 | 结果 |
|---|---|
| R `gdsfmt` 与 Python allele code | 逐元素完全一致 |
| REF dosage，包括 6,490 个缺失调用 | 逐元素完全一致，NaN 对齐 |
| 并集方向和 mean 填补后的 minor dosage | 最大绝对差 0 |
| 当前样本的 MAF | 最大绝对差 0 |
| variant ID、位置、allele、两种 QC 字段、CADD 和 GENCODE category | 均通过逐项对照；浮点注释按 R 文本导出精度比较 |
| Python 三次基因型读取及字段读取、私有参照存储的整体耗时 | 两次检查为 3.77 和 4.22 秒；未与 R 同等计时流程比较，不据此给出加速倍数 |

三个位点额外读取了全部 345,967 个样本；该真实数据的解码结果未出现大于 2 的非缺失 allele code，因此此结果证明多层索引和缺失编码读取，不能作为高 allele code 的专项精度验证。该输入没有变长 INFO 索引，FORMAT 为空；变长 INFO 实现遵循官方格式，尚未在该输入上获得真实字段对照。

此次核对还发现：`annotation/filter` 的检查位置为缺失标签，而 `annotation/info/QC_label` 为 `PASS`。采用实际有效的 QC 字段是运行正确的前提。

此解码检查与完整关联检验 benchmark 分开记录。它没有检验 null model、STAAR P 值、全基因组扫描或 GPU 加速。

## 6. 更新和验证记录

- 2026-10-04：增加原生只读 GDS 接口；完成真实 Bit2 层索引、任意样本/位点顺序、缺失、REF dosage、factor 和固定注释的 R 独立对照。
- 2026-10-04：增加并集 minor 方向、逐表型 MAF、mean/minor 缺失处理，以及保留显式 NaN 的 COO 输出。
- 2026-10-04：明确固定依赖来源与现有真实数据尚未覆盖的高 allele code、变长 INFO、FORMAT 验证边界。

## 7. 原实现和参考文献

- [CoreArray/pygds 原生 Python 接口](https://github.com/CoreArray/pygds)：GDS 文件解压和节点读取。
- [SeqArray 官方源代码](https://github.com/zhengxwen/SeqArray)：基因型层数、缺失码、REF dosage 和变长 INFO 格式。
- [PySeqArray 官方源代码](https://github.com/CoreArray/PySeqArray)：Python SeqArray 解码语义参考；本模块不依赖该预发行包。
- [STAARpipelinePheWAS](https://github.com/li-lab-genetics/STAARpipelinePheWAS)：并集方向与逐表型样本提取逻辑。
- Zheng X et al. SeqArray—a storage-efficient high-performance data format for WGS variant calls. *Bioinformatics* (2017). [doi:10.1093/bioinformatics/btx145](https://doi.org/10.1093/bioinformatics/btx145).
- Zheng X et al. A high-performance computing toolset for relatedness and principal component analysis of SNP data. *Bioinformatics* (2012). [doi:10.1093/bioinformatics/bts606](https://doi.org/10.1093/bioinformatics/bts606).
