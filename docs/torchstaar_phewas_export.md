# 原生关联结果导出为 CSV

`torchstaar_phewas.export_results` 把已经完成的 Single、coding、noncoding 和 ncRNA 原生 R 结果整理到每个表型自己的目录。每个输入文件同时保留字节一致的 R 文件，并产生同名 CSV。此步骤只转换输出格式，不拟合模型、不读取基因型，也不重新计算关联统计量。

```mermaid
flowchart LR
    A[表型及有序原生文件清单] --> B[读取原生 R 对象及重复 list 槽位]
    B --> C[按原行列顺序写临时 CSV]
    C --> D[逐标量读回校验及 SHA 核对]
    D --> E[results/表型名称/原生文件和 CSV]
```

## Python 调用

```python
from pathlib import Path
from torchstaar_phewas.export_results import export_results

# name 只用于私有输出目录；文件顺序与既有分析计划保持一致。
trait_manifest = [
    {
        "name": "trait_01",
        "native_files": [
            {
                "path": "analysis_outputs/single_segment_01.Rdata",
                "kind": "individual",
                "object_name": "results_individual_analysis",
            },
            {
                "path": "analysis_outputs/coding_batch_01.Rdata",
                "kind": "coding",
                "object_name": "results_coding",
            },
        ],
    },
]

export_report = export_results(
    trait_manifest,
    results_directory=Path("results"),
    exclude_columns=["source_label", "source_category", "source_code"],
    # 上述名字仅为示意；替换为私有输入中需排除的确切元数据列名。
    empty_columns=None,  # NULL-only 文件可由调用者明确提供实际空表列名。
)
print(export_report["native_files"], export_report["csv_files"])
```

`traits` 是非空、有序的字典列表。每个字典的 `name` 是表型的私有目录名，允许中文和空格，禁止路径分隔符、控制字符和 `.`、`..`。不同表型名称不能在 Unicode 规范化或忽略大小写后发生冲突。`native_files` 是该表型需要导出的全部文件，必须明确列出 Single 的每一段；导出器不会自动发现文件或推断缺失的分段。

每个文件字典包含以下参数：

| 参数 | 意义和格式 |
| --- | --- |
| `path` | 已完成的 `.Rdata`、`.rda` 或 `.rds` 文件路径；必须存在且是普通文件。 |
| `kind` | `individual`、`coding`、`noncoding` 或 `ncrna`；用于标明关联类型及 NULL-only 文件的默认空表结构。 |
| `object_name` | 可选的 R 工作空间对象名。只有一个对象时可以省略；多个对象时必须选择恰好一个对象。RDS 直接存储一个对象，不使用此选择参数。 |
| `empty_columns` | 可选的有序字符串列表，仅在文件没有任何可识别的表列时使用；覆盖函数级的空表列列表。不会生成虚构的结果行。 |

`results_directory` 是输出根目录。`exclude_columns` 是要删除的确切列名列表，可在函数级或每个表型字典中设置；两级列表合并，未出现的列忽略。模块不追加表型名称、字段编号或表型类别等元数据列。原文件已有的额外说明列由确切列名清单排除；gene 的原 `Category` mask 列仍属于关联输出。`empty_columns` 是函数级的有序空表列名，可省略。

标准 18 个原生文件的清单会产生 18 个 R 文件和 18 个 CSV；四段 Single 全部单独保存。例如：

```text
results/
└── trait_01/
    ├── single_segment_01.Rdata
    ├── single_segment_01.csv
    ├── single_segment_02.Rdata
    ├── single_segment_02.csv
    ├── coding_batch_01.Rdata
    ├── coding_batch_01.csv
    └── ...
```

## 命令行调用

把同一个 `trait_manifest` 保存为 UTF-8 JSON 列表，然后执行：

```bash
python -m torchstaar_phewas.export_results private_manifest.json \
    --results-directory results \
    --report private_export_report.json
```

`--exclude-column` 可以重复传入多个要移除的列。`--report` 是可选的私有 JSON 报告路径，包含目录标签、原始列名、输入输出路径、SHA 和校验结果，不应发布到公开仓库。报告不能与输入文件或关联输出重名。已有不同报告不会被覆盖；再次导出时可以使用新的报告文件名。

## CSV 结构和精度

CSV 采用 UTF-8、标准 CSV 引号规则和单行表头。顺序依次为原生 list 槽位、每个表的原始行顺序；重复命名的 list 槽位分别保留，不能转换成字典后覆盖。原列顺序保持不变；不同 mask 有额外统计列时按首次出现的顺序扩充表头，未提供的单元格留空。若两个表的共同列顺序矛盾则拒绝导出。重复列名按各自的出现顺序保留。

数值使用 17 位有效数字保存，写后读取须恢复相同浮点数。特别小的 `pvalue` 可以是科学计数法；原值为零时保留零，已有 `pvalue_log10` 按原值保存，不从零 P 值反推。报告分别记录正 P 值的 `-log10(P)` 读回误差和已有 `pvalue_log10` 的读回误差。这里的误差只衡量格式转换，与 GPU 和原软件的关联精度对照不同。

R factor 输出其可见文本；NULL 和缺失单元格留空，NaN、无穷数和布尔值分别写为 `NaN`、`Inf`、`-Inf`、`TRUE`、`FALSE`。CSV 保留数值和文本，不包含 R 对象类型、factor levels、matrix rownames 或 list 槽位名称属性；这些属性及重复槽位的完整结构仍保存在字节一致的原生文件中，私有报告也记录槽位顺序。CSV 中空字符串与缺失单元格都是空单元格。

同一 coding 文件可包含原 19 列普通 mask 和原 25 列 missense mask，逐表原行列及数值保持不变，CSV 使用首次出现的联合列头。全 NULL 的 `all_categories_incl_ptv` 文件应显式提供该类别能力对应的 25 列 `empty_columns`；也可从同一原输出类别的非空文件核对列头，不能为全 NULL 文件生成占位结果行。

NULL-only 文件仍产生只有表头的可读 CSV。未指定 `empty_columns` 时，`individual` 使用普通单表型非 SPA 的 13 列空表头，其余类型使用无注释权重的 19 列 gene 空表头。如果原任务使用 SPA、不同注释列或其他输出结构，应明确提供实际空表列名。

## 已完成的格式验证与 0.7.0 状态

已完成共享来源的 13 个表型分别写出 18 native + 18 CSV，总计 234 native、234 CSV 和 8,803,388 行，其中 Single 为 52 文件。独立 CPU 后处理实测 544.193 s，包含复制、CSV 写出、逐标量读回和最终 SHA 校验，不计入关联入口时间。所有原生字节保持一致，CSV 的 P 和已有 logP 读回最大误差均为 0。0.7.0 全部 mask 集成源码的完整回归、安装和新实际 CSV 输出已完成。全部 native 完成后，独立 CPU 阶段验收已激活的新输出：保留 234 native 与 234 CSV（52 Single），共 8,803,427 行；当前生产者绑定、SHA/列头/格式和逐标量读回核验 1,319.498 s。114,671,291 个标量、96,988,035 个数值读回通过，P 与已有 logP 序列化误差为 0；这项计时不含先前 CSV 初次写盘、关联、回归执行或打包构建，不重新计算关联、不启用 GPU。此前旧数字保留在[旧来源记录](../benchmarks/torchstaar_phewas_chr21_previous_source.json)。

## 原软件结果查看

原生文件可继续由 R 按原方式读取；CSV 用于通用表格工具：

```r
load("results/trait_01/single_segment_01.Rdata")
head(results_individual_analysis)
single_csv <- read.csv(
    "results/trait_01/single_segment_01.csv",
    check.names = FALSE,
    stringsAsFactors = FALSE
)
```

此模块对应原生结果的后处理，没有替代关联检验命令。关联模型和分析参数沿用 `torchstaar_phewas` 的原调用。

## 保存、复用和验证

每个文件先写在输出根目录的私有临时目录内，校验完成后原子发布。同一路径已有相同 SHA 的文件直接复用，保留原时间戳；存在不同内容、符号链接或同名输出冲突时拒绝覆盖。整批任务不是单一事务：异常可能留下此前已经完成的文件，重新执行同一清单会核对并复用这些文件。目录锁阻止两个导出任务同时写同一输出根目录；异常断电后遗留的锁应由调用者确认没有导出进程后再移除。

返回报告的每文件记录包含 `rows`、`columns`、`slots`、`source_sha256`、`native_sha256`、`csv_sha256`、`native_status`、`csv_status`；以及 `roundtrip_passed`、`numeric_roundtrip_passed`、`excluded_columns_absent`、`pvalue_log10_present`、逐值计数和 P/logP 误差。`association_computed=False` 明确表示这是输出后处理。

CPU 格式测试覆盖成熟原生 writer 的 data.frame/factor、各种 matrix、重复命名 mask、NULL-only、极小 P、已有 logP、列排除、全部分段保存、重复运行及冲突拒绝。这些测试证明序列化行为，不能当作真实数据的速度 benchmark 或关联方法验证。真实文件的导出耗时应单独报告，并放在已完成的 GPU 关联计时之外。

当前更新增加原生文件到 CSV 的独立转换与逐值读回核对；保持原关联计算入口和原生输出文件不变。

## 代码和参考

- [STAARpipeline](https://github.com/li-lab-genetics/STAARpipeline)：原生单表型输出。
- [STAARpipelinePheWAS](https://github.com/li-lab-genetics/STAARpipelinePheWAS)：多表型结果组织。
- [rdata](https://github.com/vnmabus/rdata)：原生 R 序列化对象解析。模块直接遍历解析对象，以保留重复命名的 list 槽位。
