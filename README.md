# Fudan WGS Toolkit

> 本工具尚在开发验证阶段，未达到原软件精度，切不可作为真实分析使用的标准化工具。

Fudan WGS Toolkit 使用 PyTorch GPU 完成全基因组单变异、gene-based coding、noncoding 和 ncRNA 关联分析。多个表型共享遗传数据读取、缓存和传输，每个表型使用自身的完整样本集合。`prepare_WGS_data` 从原始 PLINK BED/BIM/FAM 与独立功能注释准备数据；`run_WGS_all` 输出每个表型的最终 CSV。运行和数据准备均使用 Python。

## 安装

```bash
git clone https://github.com/weikanggong1/Fudan-WGS-Toolkit.git
cd Fudan-WGS-Toolkit
conda env create -f environment.yml
conda activate fudan-wgs-toolkit
```

[完整使用说明、输入格式、流程图与验证记录](docs/fudan_wgs_toolkit.md)
