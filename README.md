# Fudan WGS Toolkit

> 本工具尚在开发验证阶段，未达到原软件精度，切不可作为真实分析使用的标准化工具。

Fudan WGS Toolkit 使用 PyTorch GPU 完成全基因组单变异、gene-based coding、noncoding 和 ncRNA 关联分析。多个表型共享遗传数据读取、缓存和传输，每个表型使用自身的完整样本集合。`prepare_WGS_data` 从原始 PLINK BED/BIM/FAM 与独立功能注释准备数据；`run_WGS_all` 消费表型 CSV、协变量 CSV 和已完成遗传目录，输出四类最终 CSV。运行和数据准备均使用 Python。

遗传目录决定模型入口：新版 `dataset.json` 使用完整 FID/IID 和普通零模型；既有 `cache_dataset.json` 使用原数字身份轴、绑定的 GRM 和协变量 profile，执行混合模型 PheWAS。后者也可通过 `python -m fudan_wgs_toolkit.phewas_run` 从三个输入启动，或用私有冻结计划分配 worker。

## 安装

```bash
git clone https://github.com/weikanggong1/Fudan-WGS-Toolkit.git
cd Fudan-WGS-Toolkit
conda env create -f environment.yml
conda activate fudan-wgs-toolkit
```

[完整使用说明、输入格式、流程图与验证记录](docs/fudan_wgs_toolkit.md)

[已完成缓存与混合模型 PheWAS 调用](docs/phewas_cache_compatibility.md) · [已拟合模型的来源与复用](docs/phewas_model_reuse.md)
