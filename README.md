# Torchstaar

# 本工具尚在开发验证阶段，未达到原软件精度，切不可作为真实分析使用的标准化工具

Torchstaar 用 PyTorch GPU 完成全基因组 Single、gene-based coding、noncoding 和 ncRNA 关联分析，支持原生 R 文件和 CSV 导出。独立多表型 PheWAS 共享基因型读取、缓存和设备传输，同时保留每个表型自己的可用样本、零模型与关联结果。

## 安装

```bash
git clone https://github.com/weikanggong1/WGS.git
cd WGS
conda env create -f environment.yml
conda activate torchstaar
python -m pip install -e . --no-deps
```

[使用说明](docs/torchstaar.md)
