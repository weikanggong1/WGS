# 可选 packed Bit2 读取

本页介绍从原 GDS 解压流读取的可选 SDK packed 入口。本版完整性能测量采用预先生成的无损缓存；这两种读取入口的准备成本分别记录，见 [完整 benchmark](tf32_benchmark.md)。
该入口从 GDS 的逻辑解压流读取 packed Bit2 字节，在 GPU 直接选取样本并解包。后续层合并、等位基因方向、缺失处理、频率、MAC、筛选及原生输出沿用现有实现。它减少 SDK 在 CPU 展开全部样本及扫描样本 mask 的工作；选定样本较少时，传输的字节数可能增加。

默认使用现有读取路径。启用 packed 必须显式构建适配器，使用与已安装 PyGDS 二进制完全对应的源码构建。CoreArray allocator 是内部 C++ 接口；当前支持经验证的 64 位小端 Linux，加载时核对二进制、源码、headers、Python SOABI 和布局。配置不匹配或读取失败会报错。

## 构建输入与输出

```bash
python -m staar_phewas.gds_packed \
  --source-dir private/sdk-source \
  --source-built-sdk private/sdk-build/ccall.so \
  --output-dir private/packed-reader-build
```

| 输入 | 含义和格式 |
|---|---|
| `source_dir` / `--source-dir` | 官方 PyGDS 对应源码树的目录，保留 `pygds/include` 和 `src/CoreArray` headers；不是另一个同版本源码副本的猜测配对。 |
| `source_built_sdk` / `--source-built-sdk` | 上述源码已有构建产生的 `pygds.ccall` 动态库文件；SHA-256 必须等于当前已安装的库。使用实际含 SOABI 的文件名。 |
| `output_dir` / `--output-dir` | 新的本地目录；保存本次编译扩展、编译绑定 header 和 `packed_binding.json`。构建需要已有 C++ 编译器、Python 开发 headers 及对应 SDK 构建依赖。 |
| `packed_reader_directory` | 运行配置中的可选路径字符串，指向上述构建目录；省略时使用现有 reader。绑定文件包含本地源码路径，应保留在私有目录。 |

构建不下载、不安装依赖，也不复制上游 headers。扩展和绑定文件用于本地执行，不加入代码库。返回的构建报告记录后端、二进制/header 哈希和布局验证状态，不包含样本、位点、基因或路径。

## Python 与 pipeline 调用

```python
from staar_phewas.gds_packed import build_packed_reader
from staar_phewas.gds import SeqArrayGDS
import numpy as np

# 使用已有官方源码构建；这些目录由运行者在本地指定。
reader_build_report = build_packed_reader(
    source_dir="private/sdk-source",
    source_built_sdk="private/sdk-build/ccall.so",
    output_dir="private/packed-reader-build",
)
# 接口示例索引；正式分析由已对齐的样本和区域筛选产生。
variant_indices = np.array([0, 1], dtype=np.int64)
sample_indices = np.array([0, 1, 2], dtype=np.int64)
with SeqArrayGDS("private/input.gds",
                 packed_reader_directory="private/packed-reader-build") as genotype_reader:
    # 样本和变异均为零基索引，保留调用者顺序。
    genotype_block = genotype_reader.minor_block(
        variant_indices, sample_indices, device="cuda", resident=True,
    )
```

`variant_indices` 和 `sample_indices` 是唯一、范围有效的一维整数索引；`device="cuda"` 选择 GPU，`resident=True` 保留设备剂量。`genotype_block` 是 `DeviceMinorBlock`：`dosage` 为样本×变异的 uint8 CUDA 矩阵，0–2 为 minor copy 数、3 为整个基因型缺失；对应的样本/变异索引、原等位基因 AF、初始 MAC、缺失率、reference AC 和 called allele 数为同序的主机数组。随后由模型按原插补规则生成 FP32 关联矩阵。

完整流程在私有 JSON 中设置 `"packed_reader_directory": "private/packed-reader-build"`，然后按 [TF32 pipeline](tf32_pipeline.md) 调用 `staar-phewas-chromosome`。GDS、表型、目录、注释及输出参数的含义不变。输出仍是原生 R 文件；读取报告另记 packed calls、返回字节及读取/H2D/解包阶段时间。GPU 只缓存样本索引，不生成基因型缓存。

原 R 流程使用 SeqArray 读取 GDS，再调用 STAAR；本适配器替换读取与解包步骤，没有独立关联检验命令。原分析调用见 [chromosome](chromosome.md)，原读取代码见 [CoreArray PyGDS](https://github.com/CoreArray/pygds) 和 [SeqArray](https://github.com/zhengxwen/SeqArray)。

## 验证记录

2026-10-05：可选入口完成真实 coding、noncoding、ncRNA 与短 Single 对照，使用同一已保存的零模型。包含全部 15 个原始 mask 槽位和一个全空 ncRNA 任务；4 个原生关联文件、377 个 P 值相对同版旧读取路径全字段逐位一致。相对原 R，文件结构通过，最大 `-log10(P)` 差为 `0.000158636111464494`，低于 `0.001`。短 Single 解码 12,000 个变异，27 个符合 MAC 门槛。

解压流的 SDK 长度可能返回 `-1`，表示长度未知。读取前仍检查逻辑数组、三维 `[layer, sample, ploidy]` 形状、`ploidy=2`、范围及字节边界；长度已知时另外检查流边界，实际短读与解压错误继续报错。零长度、奇数偏移和末尾读取已和 SDK 原字节对照。

2026-10-06：将块长度、偏移和读出数量作为 Triton 运行时整数参数。上述 174 次读取仅产生一个解包内核，全字段逐位一致与原 R 精度对照再次通过。分阶段观察中，解包主机时间从约 `10.19 s` 降到 `0.70 s`；新记录的读取主机时间约 `13.29 s`。返回的 `5,149,372,910` 字节是解压后的逻辑 packed 字节，不能解释为磁盘读取量。GPU 有其他任务且启用分阶段记录，这些数值不构成端到端速度 benchmark。

上述 SDK packed 对照属于 F 之前的短流程记录，没有重新拟合零模型或运行完整染色体。另一组固定矩阵的 248 个 P 中曾有 5 个超过全部 P 的旧诊断门槛；该轮未完成完整验收。

本版 F 的完整 chr21 benchmark 使用已经生成的无损转存缓存，795 项任务、19 份原生文件结构与严格零模型通过，478,082 个 P 有效、可比较，显著联合范围的 24,713 个 P 最大 logP 差为 `0.0003579714`。缓存流程进程墙钟 `429.097 s`，300 秒目标未达。首次转存另需 `10,854.84 s`，新增缓存空间约为原 GDS 的 `8.53%`；首次转存不包含在 429.097 秒内。完整缓存接口另见[六状态 CSR 缓存](sixstate_cache.md)，测量边界见 [TF32 benchmark](tf32_benchmark.md)。

数据与绑定细节保存在 private；匿名汇总见 [TF32 benchmark](tf32_benchmark.md)。上游许可见 [NOTICE](../NOTICE.md)。
