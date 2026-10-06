# 可选 packed Bit2 读取

本页介绍从原 GDS 解压流读取的可选 SDK packed 入口。本版完整性能测量采用预先生成的无损缓存；这两种读取入口的准备成本分别记录，见 [完整 benchmark](torchstaar.md)。
该入口从 GDS 的逻辑解压流读取 packed Bit2 字节，在 GPU 直接选取样本并解包。后续层合并、等位基因方向、缺失处理、频率、MAC、筛选及原生输出沿用现有实现。它减少 SDK 在 CPU 展开全部样本及扫描样本 mask 的工作；选定样本较少时，传输的字节数可能增加。

默认使用现有读取路径。启用 packed 必须显式构建适配器，使用与已安装 PyGDS 二进制完全对应的源码构建。CoreArray allocator 是内部 C++ 接口；当前支持经验证的 64 位小端 Linux，加载时核对二进制、源码、headers、Python SOABI 和布局。配置不匹配或读取失败会报错。

## 构建输入与输出

```bash
python -m torchstaar.gds_packed \
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
from torchstaar.gds_packed import build_packed_reader
from torchstaar.gds import SeqArrayGDS
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

完整流程在私有 JSON 中设置 `"packed_reader_directory": "private/packed-reader-build"`，然后按 [TF32 pipeline](torchstaar.md) 调用 `torchstaar-chromosome`。GDS、表型、目录、注释及输出参数的含义不变。输出仍是原生 R 文件；读取报告另记 packed calls、返回字节及读取/H2D/解包阶段时间。GPU 只缓存样本索引，不生成基因型缓存。

原 R 流程使用 SeqArray 读取 GDS，再调用 STAAR；本适配器替换读取与解包步骤，没有独立关联检验命令。原分析调用见 [chromosome](torchstaar.md)，原读取代码见 [CoreArray PyGDS](https://github.com/CoreArray/pygds) 和 [SeqArray](https://github.com/zhengxwen/SeqArray)。

## 当前验证与测量

packed入口用于原GDS逻辑流读取/解包，绑定当前SDK源码、headers、编译binary及SOABI；零长度、奇数偏移、跨块/末尾短读与损坏拒绝由接口合同检查。完整关联性能当前采用已经生成的六状态缓存，首次转存与分析分开计时，实际P/结构验收见 [主指南](torchstaar.md#5-最新真实精度与耗时)。不将packed读字节数解释为磁盘物理吞吐，也不以局部读出证明完整pipeline速度。

## 更新、原实现与文献

当前packed构建显式执行，分析沿用同一原状态/剂量规则，无自动安装/编译。原SDK与许可见 [CoreArray PyGDS](https://github.com/CoreArray/pygds)、[SeqArray](https://github.com/zhengxwen/SeqArray)、[NOTICE](../NOTICE.md)；方法文献见 [主指南](torchstaar.md#7-原实现许可与参考文献)。
