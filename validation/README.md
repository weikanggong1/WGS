# 独立 R 文件检查

先在冻结的原 R 包环境中运行相同的表型、样本、GDS、注释目录、集合和参数，按教程保存原生结果。再比较 PyTorch 生成的对应文件：

```bash
Rscript validation/compare_native_outputs.R \
    private/reference/Imaging_Coding_1.Rdata runs/Imaging_Coding_1.Rdata
```

两个输入必须采用相同的 PheWAS 列表层次和保存对象名。检查器逐字段检查 `typeof`、属性顺序、列名、维度、列表名称、factor levels、row.names、S4 slots 与 `validObject`；数值默认采用 `1e-10 + 1e-7 × |reference|`，超过时退出码为 1。可通过第三、第四个参数显式指定绝对与相对容差。整数、字符、形状和属性保持精确比较。

这一步属于独立验证，需安装 R 和 Matrix。生产分析的 GDS 读取、零模型、关联计算和 `.Rdata` 写入由 Python 完成。零模型含个体状态，验证文件应保存在私有目录。具体冻结版本与已通过范围见 [benchmark](../docs/benchmark.md)。
