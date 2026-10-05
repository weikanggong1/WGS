# 输入、参数与 Python 接口

## 输入与输出结构

1. 芯片/WGS：SNP-major PLINK `prefix.bed/.bim/.fam`。芯片输入须为未压缩BED；WGS也支持`.bed.gz`，在私有InputCache解压，源文件不改变，完成该染色体后默认删除缓存BED。BED读出BIM第5列A1的0/1/2剂量，缺失为NaN；BIM六列是染色体、ID、遗传距离、物理位置、A1、A2；FAM前两列为FID/IID，第5列是sex。
2. 表型：空白分隔表，第一行 `FID IID 表型列...`。按 `phenotype_column` 选一列，按FID/IID重排；`NA`/非数值表示缺失，兼容研究表里的`-9`。
3. discovery名单：一列IID或前两列FID/IID；兼容研究六列V1/V2...表头。排除名单格式相同。相同IID但不同FID不自动合并。
4. Step1变异名单：每行第一字段为变异ID；不存在的ID忽略，保持原BED顺序及染色体block边界。
5. annotation：`变异ID gene category`或`变异ID gene domain category`；域标签原样保留。setlist四列为`gene chr position ID1,ID2,...`；mask文件两列为`maskName category1,category2,...`。
6. Sub评分筛选：外部白名单每行第一列为变异ID。GERP/REVEL等评分文件及阈值的生成属于输入注释；可换白名单或构造`MaskDefinition.extract_variants`。研究PDF明确GERP>2、Gnocchi≥4、SpliceAI>.5、REVEL>.5；JARVIS条件和完整TableS24需与真实注释生成记录核对。
7. 输出：REGENIE原格式LOCO/pred.list、逐表型.regenie/.ids，gene附MASKS标头；paper默认输出masks BED/BIM/FAM/snplist，可关闭；独立JSON保存参数、输入身份、时间、显存、CPU尾积分回退计数。具体文件树见[完整流程](REGENIE.md#原格式输出)。

`load_phenotype(path, column, sample_ids, missing_values=(-9,))`按给定FID/IID列表返回CPU float64 `[N]` tensor；`column`为精确列名，`missing_values`可修改数值缺失标记，未匹配样本与非数值填NaN。重复FID/IID会报错。

## 表型处理

```python
from torchwgs import prepare_phenotype

# phenotype_values: [N]，confound_matrix: [N,C]，与同一discovery样本顺序一致。
prepared_phenotype = prepare_phenotype(
    phenotype_values, mode="raw", covariates=confound_matrix,
    outlier_sd=5.0, quantile_normalize=True, device="cuda",
)
residual_values = prepared_phenotype.values      # 仅有效样本，[N有效]
valid_sample_indices = prepared_phenotype.sample_indices
transformation_record = prepared_phenotype.metadata
```

raw模式排除5SD异常值，对选中样本做可选分位数正态化，然后回归传入混杂因素。调用者需提供论文规定的年龄、年龄²、性别、年龄×性别、年龄²×性别、采集中心、40个遗传主成分，以及适用的总体积协变量；分析该体积本身时不回归该协变量。相应成像表型还需运动与扫描床位置信息。采集中心等类别列需先编码。

`mode="residual"`只筛缺失，不重新回归。REGENIE阶段RINT由Step1/Step2各自的`apply_rint`控制，采用平均并列rank和Blom公式。`WGSConfig.paper(apply_rint=False)`统一关闭三个阶段，raw模式的分位数正态化也关闭；两个选项可单独修改。

`run_discovery`的raw模式从`DiscoveryInputs.covariates`读取与筛选后芯片样本对齐的`[N,C]`数值tensor，并将残差散射回原行；缺失行保留以复现原Step1折叠行为。

## Step1

```python
from torchwgs import BedReader, Step1Config, fit_null

array_reader = BedReader(array_prefix, keep=discovery_keep_file, remove=sample_remove_file)
null_model = fit_null(
    array_reader, phenotype_values,
    config=Step1Config(device="cuda", block_size=1000, apply_rint=True),
    variant_indices=qc_variant_indices, phenotype_name="trait_01",
    output_dir="/results/discovery/Step1",
)
null_model.export_regenie("/results/discovery/Step1/discovery")
```

`phenotype_values`为reader顺序的`[N]`，`qc_variant_indices`为原BIM索引。可传torch/NumPy `[N,M]`矩阵，另提供`chromosomes[M]`和`sample_ids`。返回`NullModel`有sample_ids、sample_indices、loco[N有效,22]、prs[N有效]、y_scale和metadata；`save/load`保存本地缓存，`from_regenie`读取既有原格式LOCO。使用导入模型时应同时核对原cohort、表型、RINT和软件版本，原LOCO文本本身不包含完整来源记录。

`fit_null`另有`covariates=None`（与原样本行对齐的`[N,C]`数值矩阵）和`progress_callback=None`；回调接收包含`stage`及已完成block/fold计数的dict，用于记录进度，返回值不参与拟合。默认memmap模式在`output_dir=None`时使用临时L0目录，完成后清理；指定目录便于保留模型与阶段统计。

`NullModel.export_regenie(prefix, phenotype_name=None, sort_ids=True, pheno_index=1, n_chromosomes=23, compressed=False)`返回LOCO与pred.list两个路径；表型名默认取模型metadata，`sort_ids`控制FID_IID排序，`pheno_index`控制LOCO文件数字后缀，`n_chromosomes`控制输出染色体行数，`compressed=True`压缩LOCO。`NullModel.from_regenie(path, phenotype_name="phenotype", sample_ids=None, chromosomes=None, y_scale=1.)`返回导入模型；`sample_ids`指定对齐顺序，`chromosomes`选择输入行，`y_scale`仅记录来源尺度，不缩放已标准化预测。

`BedReader(prefix, keep=None, remove=None, sample_ids=None, metadata_cache_size=100000)`只解码需要的样本/变异；`sample_ids`可指定FID/IID顺序。`metadata_cache_size`限制已查询BIM元数据与缺失ID的LRU条数，设0关闭。后续查询属于缓存子集时省去重扫；新ID仍扫描BIM，不建立整张BIM索引。BIM大小、时间或文件身份变化会清空缓存，返回顺序始终为原BIM顺序。

| Step1Config参数 | 默认值与作用 |
|---|---|
| block_size / folds | 1000 / 5；按染色体分块，按有效样本累计构造连续fold |
| ridge_l0 / ridge_l1 | (.01,.25,.5,.75,.99)；h网格，L0惩罚使用总M，L1使用预测列数 |
| apply_rint | True；可关闭，不在函数内固定执行 |
| preserve_masked_rows | True；保留原软件缺失表型零行和L1归一化行为 |
| device / dtype / tf32 | cuda / float32 / True；float64用于严格对照 |
| l0_storage / keep_l0 | memmap / False；磁盘存L0预测，完成后默认清理 |
| sample_chunk_size | 4096；L1 GPU样本分块 |
| max_gpu_gb | 20；Step1估计工作区的十进制GB预算（每单位10⁹字节），超出时提前报MemoryError；pipeline的分配器限制由execution另行设置 |
| variance_tolerance | 1e−6；拒绝无法标准化的变异/预测列 |
| covariate_eigen_tolerance | 1e−15；X'X相对特征值截断 |
| output_chromosomes | 1–22；export额外写原格式chr23=PRS |

## Single-variant

```python
from torchwgs import create_test_context, SingleVariantConfig, test_single_variant

association_context = create_test_context(
    phenotype_values, chromosome_loco_predictions,
    sample_ids=wgs_reader.sample_ids, covariates=None,
    apply_rint=False, device="cuda", dtype="float64", tf32=False,
)
single_result_file = test_single_variant(
    wgs_reader, association_context,
    config=SingleVariantConfig(maf_min=0., min_mac=20, genotype_reader="cuda_packed"),
    output_path="/results/discovery/single_trait_01.regenie",
)
```

context包含规范化残差、y尺度、LOCO后残差尺度、协变量basis、样本顺序与有效索引。QT score统计用原n−q归一化，输出χ²1尾概率，不改成普通OLS的Student-t。`SingleVariantConfig`有block_size=1000、maf_min=0（严格大于）、min_mac=20、apply_rint=True、device="cuda"、dtype="float32"、tf32=True、genotype_reader="cpu"。`genotype_reader="cuda_packed"`将原packed BED字节传入GPU解码，按有效样本行直接选取，保留缺失值与FAM顺序；`"cpu"`使用原CPU解码路径。MAC和频率先用未插补的有效基因型计数，在协变量投影前剔除不合格列；默认原结果保留所有MAC20位点，频率阈值留给汇总。低层函数收到已构建context时使用context的RINT/dtype设置；这些配置在pipeline负责构建context。无output_path时返回DataFrame，只适合有界结果，全基因组应流式写文件。

`test_single_variant(..., variant_indices=None)`默认扫描全部BIM位点；传原BIM索引序列可限定读取范围，仍执行配置的MAC/频率筛选。指定`output_path`时返回写出的文件路径。

## Gene-based

```python
from torchwgs import GeneConfig, test_gene_based
from torchwgs.output import RegenieWriter
from torchwgs.masks import load_annotations, load_mask_definitions, effective_mask_definitions

mask_definitions = load_mask_definitions(mask_definition_file)
gene_configuration = GeneConfig(aaf_bins=(.01,), vc_max_aaf=.01, min_mac=1)
annotation_records = load_annotations(annotation_file)  # 保留完整注释，不按setlist或基因筛选表头
candidate_ids = {record.variant_id for record in annotation_records}
if gene_configuration.extract_variants is not None:
    candidate_ids.intersection_update(gene_configuration.extract_variants)
variant_lookup = wgs_reader.find_variants(candidate_ids)
header_masks = effective_mask_definitions(mask_definitions, annotation_records, variant_lookup)
with RegenieWriter("/results/discovery/gene", "trait_01", masks=header_masks,
                   sample_ids=association_context.sample_ids,
                   write_samples=True, print_pheno_name=True) as result_writer:
    for result_row in test_gene_based(
        wgs_reader, association_context, annotation_records, setlist_file,
        mask_definitions, gene_configuration, variant_lookup=variant_lookup,
    ):
        result_writer.write(result_row)
```

`test_gene_based`的`genotypes`可为BedReader，也可为与context有效样本对齐的`[N,M]`矩阵；矩阵输入须传`variants`，每列对应一个`Variant(index, chrom, id, position, allele1, allele0)`。`annotation/setlist/masks`均接受原文本路径或解析后的记录。`variant_lookup=None`让reader自行查BIM；传`{ID: Variant}`可复用已完成的查询。`artifact_callback=None`可接收每个gene的`GeneArtifacts(gene, masks)`，其中mask保留成员与burden、移除VC矩阵，供另行写mask文件；函数自身仍逐行返回关联dict。

`MaskDefinition(name, categories, extract_variants=None, score=None, category_order=None)`的`categories`为注释类别集合，`extract_variants`限定成员白名单，`score`记录评分名称；`category_order`控制原格式表头类别顺序，未指定时按类别排序。`FrequencyDomain(lower=0., upper=1., lower_inclusive=True, upper_inclusive=True)`显式设置MAF区间与两端是否包含，放入`GeneConfig.domain_mapping`按域名使用。

| GeneConfig参数 | 默认值与作用 |
|---|---|
| apply_rint | True；pipeline据此构建gene context，gene函数不重复RINT |
| aaf_bins / vc_max_aaf | (.01,) / .01；burden和VC分别设置AAF上限 |
| min_mac | 1；mask最低MAC，不能据此删除所有构成变异 |
| collapse_mac | 10；冻结原软件的低ALT计数折叠阈值；不同于注释UR频率域 |
| include_singletons / singleton_carrier | True / False；单ALT副本定义，可选单携带者定义 |
| include_domains / include_overall | True / True；原annotation分层和整体mask |
| domain_mapping | None；允许显式FrequencyDomain(lower,upper及包含边界)，默认保留原标签 |
| extract_variants / extract_genes | None；白名单，Sub评分筛选或限定验证基因 |
| variant_block_size | 1000；大gene的流式读取块 |
| max_matrix_bytes | 8GiB；gene矩阵/协方差工作空间上限，并行时按每任务预算进一步限制 |
| vc_storage / vc_score_method | dense / residual；可设 sparse / crossproduct 进行稀疏分块交叉乘积计算，减少 N×M 投影副本 |
| genotype_reader | cpu；cuda_packed用PyTorch解码原BED位，按有效样本直接选行。paper预设cuda_packed |
| vc_score_block_size | 256；稀疏协方差乘积的列块大小 |
| genotype_orientation | allele1；同原REGENIE计BIM A1。variant_id可显式核对chr:pos:REF:ALT并翻转，alt用于已定向矩阵 |
| skato_rhos | (0,.01,.04,.09,.16,.25,.5,1)；内部按冻结原版clip最高端点 |
| tail_method | regenie；兼容强尾/回退策略，也可设exact进行数学审查 |
| davies_controller | auto；谱维数≥1024时用NumPy向量化误差界控制器，其余保留scalar。scalar/numpy可显式选择；Fourier积分始终在tensor设备执行，误差预算与故障路径保持 |
| eigen_backend | dense；逐个rho计算完整特征值，可设secular或auto。paper预设auto，仅CUDA且维数达到secular_min_size时启用秩一更新 |
| secular_min_size / secular_root_chunk / secular_iterations | 4096 / 256 / 64；auto切换维数、同时求根数量和二分次数。保持原正特征值筛选与rho端点定义 |
| skato_integral_backend | adaptive_x；独立PyTorch GK21自适应积分，使用原 REGENIE 的 χ² 坐标。可选adaptive_sqrt（平方根变量变换）或segmented；低层skato_logp数学接口默认segmented |
| skato_integral_epsabs | 1e-25；原版SKAT-O绝对误差预算 |
| skato_integral_epsrel | 2**-13（0.0001220703125）；原版相对误差预算 |
| skato_integral_max_intervals | 1000；自适应区间预算，耗尽时按原版Bonferroni规则回退 |
| acato_full / gene_p / run_sbat | True；开启完整组合、GENE-P和SBAT |
| gene_p_groups | None；可定义GENE-P组合mask分组 |
| rank_tolerance | 1e−7；共线mask rank处理 |
| qr_tie_tolerance | 0；用冻结SSE2范数顺序选pivot；正值可显式指定相对容差内按输入顺序选列 |
| genotype_scale_tolerance | 1e−6；排除投影后低于原numtol的常数或近常数mask |
| sbat_max_subsets / sbat_qmc_samples / sbat_seed | 10 / 8192 / 0；chi-bar子集数、正交积分Sobol样本数和seed。0个子集表示完整枚举。8192为本实现积分设置，论文未规定；可提高精度 |
| sbat_subset_sampling | unique；原3.4.1高维抽样为 with_replacement，可显式选择，sbat_seed 控制本实现重复性 |
| beta_a / beta_b | 1 / 25；Beta(MAF)权重，使用GPU密度公式，可修改 |

输出12类TEST：ADD、ADD-SKAT、ADD-SKATO、ADD-SKATO-ACAT、ADD-ACATO、ADD-ACATV、ADD-ACATV-ACAT、ADD-BURDEN-ACAT、ADD-BURDEN-SBAT、ADD-BURDEN-SBAT_POS、ADD-BURDEN-SBAT_NEG、GENE_P。

多rho的kernel检验按原版规则先检查burden消除后的残差谱与moment方差。失败的mask保留ACAT-V和ADD，联合SKATO-ACAT只计入有效kernel masks，其DF及GENE-P组成相应更新。单个位点与固定rho保留各自原版分支。低层`statistics.skato_logp(..., native_validity=True)`公开这一规则：返回`kernel_valid=False`和失败原因时没有数值kernel结果；默认False保留独立数学接口的退化核分布计算。

`native_validity=True`另按原版处理条件SF下溢：主动零值区间之外的float64概率下溢触发积分失败，再尝试Bonferroni回退；回退不可用或最终P>1时返回`SKATO=None`，其他有效kernel结果保留。积分目标及最终SKATO概率使用`10*DBL_MIN`地板；单站点、固定rho及默认`native_validity=False`的数学接口保留各自行为。

源mask定义全部读取。原格式`##MASKS`表头仅列出BIM与全局/评分白名单交集中的注释类别，混合定义按原顺序去掉未知类别，全部未知的定义省略。类别注册在setlist、所选gene、MAC和AAF筛选之前；pipeline自动处理并复用同一BIM lookup。chr21的42个源定义注册37个活跃定义，其中Pseudo为5/8、RNA为8/10。直接写文件时可按上例调用`effective_mask_definitions()`。

gene的稀疏小矩阵投影与推断保持GPU float64，使用context中未丢精度的y_float64和covariates_q_float64；single仍按请求dtype计算大块。SBAT在原基因型样本位置补零，再以独立Triton kernel复现冻结Eigen/SSE2的列范数累加顺序，减少共线mask的pivot差异。CPU模式用相同递推。该兼容顺序针对本次核对的3.4.1二进制，其他ISA构建可能选择不同的等范数列。

Sub的评分白名单与调用者的global白名单取交集，在BIM查找、BED解码和GPU传输前筛选。输出的df=1等效χ²在−log10(P)≤300时使用float64正态逆CDF，接近P=1时用expm1/erfinv；其他df和更强尾保留log域二分算法。该优化减少逐行GPU调用，不改变关联检验或尾积分预算。

## Pipeline和汇总

`DiscoveryInputs`指定一个表型的输入；同一FID/IID的芯片、WGS、表型和LOCO行可以不同顺序，程序按身份对齐。

| 输入字段 | 必填 / 默认 | 格式与作用 |
|---|---|---|
| array_prefix | 必填 | 未压缩芯片BED/BIM/FAM前缀，用于拟合Step1及建立样本身份 |
| phenotype_file / phenotype_column | 必填 | 表型表路径和精确列名；一次分析一个连续表型 |
| discovery_samples | 必填 | discovery样本名单，限定正式输入cohort |
| wgs_prefixes | 必填 | `{染色体字符串: BED前缀}`；只运行列出的染色体 |
| array_variant_include | None | 芯片QC变异白名单；None使用全部芯片变异 |
| sample_remove | None | 从discovery名单中排除样本 |
| gene_analyses | 空dict | `{染色体: [GeneAnalysis,...]}`；每条染色体列出完整Main/Sub入口 |
| covariates | None | `[N,C]`数值矩阵，N为keep/remove后的芯片行数；raw模式用于残差回归，residual模式作为关联协变量 |
| imported_loco | None | 已核对来源的`.loco`或`_pred.list`；导入时跳过Step1拟合，仍按芯片身份对齐 |

每个`GeneAnalysis`含`name`（同染色体唯一的输出组名）、`annotation_file`、`setlist_file`和`mask_definition_file`四个必填字符串，以及可选`variant_whitelist_file=None`。Sub评分白名单由最后一项指定，与`gene_based.extract_variants`取交集。

输入JSON含本地路径；若直接填covariates数组，还包含参与者协变量。配置和输入JSON应存放在自己的私有目录，不能作为公开示例发布。残差表型默认不需要协变量数组。

已有研究注释文件布局可调用`study_gene_analyses(annotation_root)`构造每条染色体14 Main和12 Sub入口（42个基础mask定义），返回`{染色体字符串: [GeneAnalysis,...]}`，默认检查所有文件存在。`main_types`、`sub_combinations`、`chromosomes`均可改；`require_files=False`允许生成配置时暂不检查文件，实际分析仍需提供输入。评分名字代表输入白名单，不推测JARVIS等未核对数值阈值。完整mask列表与Table S24的一致性应由注释生成来源确认。

`WGSConfig`含step1、single_variant、gene_based、significance和execution五组参数，以及phenotype_mode、phenotype_quantile_normalize、phenotype_outlier_sd、gzip_output、write_samples、print_pheno_name、split_by_pheno、write_masks、keep_uncompressed_inputs。`print_pheno_name=True`给.ids写原脚本所用的表型标签首行。`WGSConfig.paper()`默认`write_masks=True`，输出原脚本启用的BED/BIM/FAM/snplist；可显式关闭，直接`WGSConfig()`仍默认False。paper另预设single和gene的cuda_packed解码、gene的sparse/crossproduct、auto特征值后端和with_replacement抽样。配置生成器默认写mask，`--no-write-masks`关闭，旧`--write-masks`继续可用。切换CPU运行时将single_variant.genotype_reader和gene_based.genotype_reader设为"cpu"。

`configuration.to_dict()`返回全部配置字段的字典；`WGSConfig.from_dict(overrides)`以paper预设补全缺省字段，并合并各参数组的局部覆盖。`gene_based.domain_mapping`中的字典会恢复为`FrequencyDomain`对象；CLI的`--config`使用同一规则。

| 流程字段 | 默认值与作用 |
|---|---|
| phenotype_mode | residual；可设raw先处理异常值与协变量 |
| phenotype_quantile_normalize / phenotype_outlier_sd | True / 5；raw输入的分位数正态化及异常值SD阈值 |
| gzip_output | False；True压缩Step2结果为`.regenie.gz` |
| write_samples / print_pheno_name | True / True；输出`.ids`及其表型标签首行 |
| split_by_pheno | True；False使用Y1列后缀与`.regenie.Ydict` |
| write_masks | paper为True，直接WGSConfig为False；输出构建后的BED/BIM/FAM/snplist |
| keep_uncompressed_inputs | False；染色体完成后清理InputCache中展开的BED |

`run_discovery(inputs, config=None, output_dir=..., run_single=True, run_gene=True, resume=True)`返回与`run_manifest.json`相同的dict，包含配置、输入身份、每阶段报告、结果路径、汇总计数、数值诊断、总耗时和按CUDA设备记录的峰值。`config=None`采用paper预设；`resume=False`重新计算。两类关联可单独关闭；启用的分析仍需要拟合或导入Step1。gene使用`single_variant.device`建立context，其矩阵推断保持float64。

`summarize_results(single_files, gene_files, output_dir=..., config=...)`读取结果路径列表，返回`single_significant`（显著位点行数）、`gene_test_rows`（显著gene/TEST行数）和`loci`；后者不等于独立gene数。输出三个TSV的原关联字段，locus表额外包含`LOCUS_START/LOCUS_END/N_LEADS`。

`ExecutionConfig(parallel_level="serial", workers=1, max_gpu_gb=20.)`控制并行层级（serial/mask/chromosome）、并发数量和本GPU的PyTorch分配器GiB预算（每单位2³⁰字节）；CUDA上下文及其他进程占用需另行观测。pipeline入口即设置该预算，Step1估计工作区使用step1.max_gpu_gb与execution.max_gpu_gb数值的较小值，仍按十进制GB检查；每次调用都会设置本次预算。显存峰值按显式CUDA设备逐项报告。mask模式以每个Main/Sub组为独立任务；chromosome模式以每条染色体为独立任务。组内全部category/domain/AAF/singleton masks均保留。GPU任务使用独立CUDA streams、独立输出文件和独立数值诊断计数，结果按输入组的次序汇总。

每个gene任务的矩阵预算为`min(gene_based.max_matrix_bytes, int(max(.25, max_gpu_gb / concurrency - 1) * 1024**3))`字节，`concurrency`在serial时为1、其他模式为workers。在mask/chromosome模式中，仅`MemoryError`或`torch.cuda.OutOfMemoryError`触发等待全部并发任务结束后按总预算串行重试一次；失败尝试和重试都计入总耗时，染色体重试可复用已提交的兼容阶段。其他异常及重试再次失败会向调用者抛出。

`SignificanceConfig`公开effective_phenotypes=831.50、n_genes=17863、variant_alpha=5e−9、gene_alpha=.05、lead_window_bp=500000、locus_merge_bp=1000000。另有single_frequency_min=.001和single_frequency_field="a1freq"，在完整原格式输出之后筛选频率。`WGSConfig.paper()`选择field="maf"，按论文的minor allele frequency筛选；field="a1freq"匹配旧研究汇总代码。运行一个表型时仍用论文全研究阈值；修改这些参数只重汇总。缓存核对输入身份、运行参数和实现hash：改MAF/MAC重single；改annotation/mask/白名单重gene；改样本、表型、芯片QC、RINT或ridge重Step1。

`excluded_locus_regions=((6,25000000,34000000),)`在挑选lead之前排除闭区间内的候选位点；空tuple关闭，JSON可用空数组。默认chr6 MHC区间来自研究源脚本 `clump.py` 的额外规则，论文Methods只规定±500kb递归选lead与1Mb合并，没有规定这一排除区间。它只影响`single_loci.tsv`，完整`.regenie`和`single_significant.tsv`仍保留这些位点。该步骤按物理距离选择locus，没有使用LD或r²。

缓存还检查结果、ID、mask文件的存在和身份。≤64MiB文件核对SHA-256，大文件核对路径/大小/纳秒mtime；修改大输入后应保留mtime变化或使用resume=False。输出先写.partial，计算异常时删除未完成部分；完成后提交原文件名。Step1/Step2的`.log`和主`discovery.log`为可读文本，采用参数、样本/位点/test计数、耗时的布局，明确标注PyTorch引擎；实际软件标识、时间及事件不与原REGENIE程序生成的日志逐字节一致。`Step1/discovery.log`区分`fitted`、`cached`和`imported`，Elapsed time记录本次拟合或加载耗时；缓存日志另标原拟合时间，导入日志不推测源拟合时间。Step1缓存同时核对模型、LOCO、pred.list和日志；缓存加载后先更新日志，再更新其manifest身份。导入模式只引用外部LOCO/list，另生成本次导入日志和来源manifest，不另写本地模型或预测文件；`output_files.step1`列出本次使用的外部预测或生成的本地文件及日志。Step1进度仍记录在主日志，拟合参数与阶段统计另存`null_model.json`。`discovery.events.jsonl`保留结构化进度事件，manifest/progress继续使用独立JSON。Single阶段只有完整读到EOF后才登记`source_variants`和`variants_scanned`，其中包含MAC过滤掉的位点。

## 对应原软件验证命令

以下是开发验证入口，生产pipeline不执行这些命令。3.4.1不能同时传--keep与--remove，先生成已取差集的discovery_final.keep。

```bash
regenie --step 1 --bed /data/array --extract /data/qc.snplist \
  --keep /data/discovery_final.keep --phenoFile /data/residuals.txt \
  --phenoCol trait_01 --qt --apply-rint --bsize 1000 --lowmem \
  --threads 8 --out /reference/discovery_step1

regenie --step 2 --bed /data/wgs/chr5 --keep /data/discovery_final.keep \
  --phenoFile /data/residuals.txt \
  --phenoCol trait_01 --pred /reference/discovery_step1_pred.list \
  --qt --apply-rint --bsize 1000 --minMAC 20 --write-samples --print-pheno \
  --threads 8 --out /reference/discovery_single_c5

regenie --step 2 --bed /data/wgs/chr5 --keep /data/discovery_final.keep \
  --phenoFile /data/residuals.txt \
  --phenoCol trait_01 --pred /reference/discovery_step1_pred.list \
  --qt --apply-rint --bsize 1000 --minMAC 1 \
  --aaf-bins .01 --vc-maxAAF .01 --vc-tests skato,acato-full --rgc-gene-p \
  --anno-file /data/annotation/PTV_chr5.txt --set-list /data/annotation/chr5_PTV.setlist \
  --mask-def /data/annotation/Mask_PTV.txt --write-samples --print-pheno \
  --write-mask --write-mask-snplist \
  --threads 8 --out /reference/discovery_gene_c5
```

原软件关闭RINT时去掉`--apply-rint`；Sub加`--extract`评分白名单，另有全局白名单时先生成两者交集。single原脚本先做minMAC扫描，研究汇总脚本再筛A1FREQ>.001；论文正文描述MAF，Python接口可显式选择。原实现链接与参考文献见[完整流程](REGENIE.md#数值后端与参考)。
