# WGS GPU association analysis

`staar_phewas` 使用 PyTorch float64 复现 STAARpipelinePheWAS 的关联计算，并从原生 GDS 流式读取基因型。R 用于独立验证，生产分析不调用 R。

安装、Python/命令行调用、全部参数、原 R 示例与流程图见 [pipeline 文档](docs/pipeline.md)。真实数据对照范围和误差见 [benchmark](docs/benchmark.md)。

正式输出采用 STAAR 原生 `.Rdata` 文件，保留保存对象名、列表层次、混合矩阵、factor 与 `row.names`；[文件格式](docs/r_native_output.md)给出批次命名和 R 读回方式。[联合多表型](docs/multi.md)与[二分类 SPA](docs/binary.md)分别说明相关性模型、原包兼容问题和实际验证范围。

```bash
conda env create -f environments/staar-gpu.yml
conda activate staar-phewas-torch
LZMA_PREFIX="$CONDA_PREFIX" python -m pip install --no-deps --no-build-isolation \
  'git+https://github.com/CoreArray/pygds.git@b7a2dbbebf3b06ac4e97c806e36ec4e1a6af5bdd'
python -m pip install -e .
staar-phewas-torch examples/staar-analysis.json --device cuda --report runs/summary.json
```

示例配置使用占位路径。输入 GDS、表型、亲缘信息和零模型均保存在分析者的私有目录。

从表型表和原生 R 格式的稀疏 GRM 准备输入：见 [样本对齐](docs/prepare.md)。零模型输入、输出和原 R 调用见 [零模型](docs/null_model.md)。
