# 真实数据验收：0.1.0

2026-10-04，以真实脑影像表型和已注释的 chr22 GDS 验证 PyTorch 改写。正式单表型流程从原始表型表与 R 格式的 GRM 开始，使用纯 Python 样本准备、GPU RINT/零模型/关联计算及原生 `.Rdata` 写入。原 R 包独立运行相同数据与参数，随后读取 Python 正式文件。

## 单表型完整流程和文件

输入 GDS 为 345,967 人、14,866,221 个位点；表型有 64,840 行，最终分析样本 42,652 人。已排除缺失与名单，并保留真实非单位 GRM 对角。Python 原生 GRM/表型准备的 8 个关键数组与独立 R 导出完全一致；RINT 全部样本与 R `qnorm` 逐项一致。

11 项正式文件的递归 R 回读检查全部通过：保存对象名、类型、属性顺序、列名、列表层次、混合矩阵、factor levels、row.names 与空 `NULL` 零结构差异；1,243 个数值元素均满足 `abs=1e-10 + rel=1e-7 × |reference|`。最大绝对差为 **1.12218e-8**，最大相对差为 **3.59737e-8**。包含三基因全部编码类别、COMT 全部非编码类别、ncRNA、单滑窗、五个重叠滑窗、18 个单变异，以及三个单独类别入口。

正式 `obj_nullmodel.Rdata` 的 20 字段结构零差异，数值最大绝对差 1.78e-14。原 `STAAR_sp` 可直接使用这个 Python 写出的零模型；真实 pLoF 三变异检验最大 P 差 1.92e-15。文件内容按类型及数值比较，不要求 gzip 字节相同。

| 同一真实任务 | 原 R 函数耗时（秒） | GPU pipeline 函数耗时（秒） |
| --- | ---: | ---: |
| OR11H1 全编码 | 29.765 | 39.508 |
| COMT 全编码 | 25.249 | 35.574 |
| APOBEC3A 全编码 | 25.001 | 22.462 |
| OR11H1 单滑窗 | 11.606 | 14.339 |
| OR11H1 单变异 | 6.511 | 13.578 |
| COMT 全非编码 | 138.010 | 264.666 |
| POTEH-AS1 ncRNA | 20.696 | 141.297 |
| OR11H1 五重叠滑窗 | 16.512 | 72.694 |
| COMT 单独 missense | 15.479 | 14.674 |
| COMT 单独 disruptive missense | 10.514 | 6.367 |
| COMT 单独 UTR | 19.504 | 172.286 |

GPU CLI 实测端到端 **817.508 秒**，包括零模型、GDS 初始化和读取、集合选择、统计及正式文件写入；其中零模型阶段 4.699 秒，峰值 CUDA 分配 **984.203 MiB**。原 R 零模型 4.582 秒，表中 11 个函数加零模型的阶段耗时合计 323.429 秒；这是阶段合计，未作为完整 R CLI 总耗时。函数计时都包含该函数的数据提取和检验，不含随后保存文件。

当前端到端耗时主要来自 GDS 提取和字符串注释筛选，尚无整体 GPU 加速结论。注释已按 250,000 个位点分块，仅对入选位点读取 PHRED；同一任务的旧全量注释版本非编码/ncRNA为 478.106/348.508 秒，当前为 264.666/141.297 秒。CUDA 内核和 I/O 时间不能互相替代。

## 多个表型与其他分析

| 验证分支 | 真实范围 | 结果 |
| --- | --- | --- |
| 两个独立 Gaussian 模型的 PheWAS | 42,652/42,418 人，COMT 全编码、OR11H1 滑窗和单变异 | 832 个数值元素全部严格通过；正式 Rdata 零结构差异；最大绝对差 1.122e-8 |
| 单表型关联核心 | 42,652 人，七个真实集合，每集 40 个字段 | 原生 GPU 零模型→G→score/covariance→检验的 280 字段全部严格通过，最大 P 差 2.052e-9；使用冻结转换向量，完整原始输入结论以上方 11 文件为准 |
| 二项 SPA 完整 coding wrapper | 原拟合状态的 288,554 人，实际 pLoF 11/19 个变异 | 原 GDS→GPU→正式 Rdata 全部严格通过；最大 P 差 1.33e-15/4.44e-16 |
| 普通 logistic 零模型 | 288,554 人、41 固定效应 | 与 R glm 同 7 次迭代；系数最大差 2.22e-13，拟合概率/残差 4.87e-15，协方差 4.78e-12 |
| 两/三性状联合单变异 | 42,418/42,411 人，各 11 变异 | 原 PheWAS 分支与正式 Rdata 严格通过；209 数值元素；最大 P 差 1.50e-11 |
| 两/三性状联合集合检验 | 每组 3 个真实集合，每集 40 字段 | 三性状最大 P 差 4.71e-12；两性状 COMT missense 的一个 SKAT 差 1.14e-7，尚未达到上述严格标准 |

两独立表型 GPU CLI 端到端 110.081 秒、峰值约 255.90 MiB；三项函数为 39.016/17.792/21.165 秒，具体记录见 [独立多表型](phewas_multiple.md)。原 R 对应三项 32.811/15.903/8.167 秒。联合单变异原 R 8.596 秒、GPU 函数 13.776 秒，GPU 含加载和保存 19.389 秒。

原推荐依赖缺少 `MultiSTAAR_sp`，其 joint gene wrapper 无法直接运行；联合集合对照使用真实官方 `MultiSTAAR`。原 GMMAT 1.3.2 对此两/三性状 GRM 在边界更新报错；严格模式保留该失败，显式 `joint_mode="robust"` 提供因子 REML 修复。固定其拟合协方差后的原 R/GPU 联合检验最大 P 差 5.62e-12/1.08e-13。详细模型与已知原包问题见 [联合多表型](multi.md)。

二项混合零模型尚未从原始数据实现 GPU 拟合；SPA 的实际完整 wrapper 验证使用明确提供的既有拟合状态。缺少完整状态时，二项正式零模型导出明确拒绝，不生成部分 `obj_nullmodel`。见 [二项零模型](binary_null.md)与 [SPA](binary.md)。本轮未运行全部染色体和所有基因；也未实现不属于主 PheWAS 包的 SCANG 动态窗口。

## 冻结环境与重现

- GPU：共享 A100-SXM4-80GB；Python 3.12、PyTorch 2.5.1+cu118、NumPy 1.26.4、SciPy 1.14.1，数值运算 float64。计时为该共享资源的实测观察。
- R：原作者推荐镜像 `zilinli/staarpipeline:0.9.9`，digest `sha256:fa4df148cfe83097422a6982828e1e8ccc7d524b1a5ecb3f6ceab99eb4ec9faf`，R 3.6.1、Matrix 1.2-17、GMMAT 1.3.2，R BLAS 单线程；通过用户空间容器运行原包。
- STAAR 0.9.9：`yuxinyuanqt/STAAR@4bbf77ba8a90894a434f5eb4473d540e172dad05`。
- STAARpipeline 0.9.9：`yuxinyuanqt/STAARpipeline@fbce778bf14cc4f9e892989a194c64bae2670311`。
- STAARpipelinePheWAS 0.9.7.1：`yuxinyuanqt/STAARpipelinePheWAS@6b72cf9d1f5ef001887b37d1d77a0c5370c46be0`；R 函数与主参考 `li-lab-genetics/STAARpipelinePheWAS@7a2c49617b3791c35a20504e260c038c02a4c643` 相同。
- MultiSTAAR：`xihaoli/MultiSTAAR@c372e135d88d5537c43af2d0f3e935f47cafd11c`。
- CoreArray pygds：`b7a2dbbebf3b06ac4e97c806e36ec4e1a6af5bdd`，原生 GDS 精确解码检查含 2,079 人×35 位点和真实多层 Bit2 位点；掩码检查 285 次选择、77 个 PHRED 矩阵。

使用 [固定 Conda 环境](../environments/staar-gpu.yml)和主页的原生 GDS 编译命令，按 [输入准备](prepare.md)与[pipeline](pipeline.md)运行。正式文件的独立检查器是 [compare_native_outputs.R](../validation/compare_native_outputs.R)。本次全部 34 项数值/身份映射回归检查通过；固定小矩阵测试仅检验计算规则，未替代真实数据 benchmark。

0.1.0 的开发记录：完成原生 GDS 与掩码对照；修复零模型求逆、扩展精度方差、标量投影、列主序 AI 求解的舍入次序；补齐原生 R 输出及批次 append/rbind；最终以原始影像输入完成单表型和两个独立表型验收。完整代码以本版本 main 提交为准，未发布旧调试拟合文件或个体数据。
