# 输入、参数与 Python 接口

## 输入与输出结构

1. 芯片/WGS：SNP-major PLINK `prefix.bed/.bim/.fam`。芯片输入须为未压缩BED；WGS也支持`.bed.gz`，在私有InputCache解压，源文件不改变，完成该染色体后默认删除缓存BED。BED读出BIM第5列A1的0/1/2剂量，缺失为NaN；BIM六列是染色体、ID、遗传距离、物理位置、A1、A2；FAM前两列为FID/IID，第5列是sex。
2. 表型：空白分隔表，第一行 `FID IID 表型列...`。按 `phenotype_column` 选一列，按FID/IID重排；`NA`/非数值表示缺失，兼容研究表里的`-9`。
3. discovery名单：一列IID或前两列FID/IID；兼容研究六列V1/V2...表头。排除名单格式相同。相同IID但不同FID不自动合并。
4. Step1变异名单：每行第一字段为变异ID；不存在的ID忽略，保持原BED顺序及染色体block边界。
5. annotation：`变异ID gene category`或`变异ID gene domain category`；域标签原样保留。setlist四列为`gene chr position ID1,ID2,...`；mask文件两列为`maskName category1,category2,...`。
6. Sub评分筛选：外部白名单每行第一列为变异ID。GERP/REVEL等评分文件及阈值的生成属于输入注释；可换白名单或构造`MaskDefinition.extract_variants`。研究PDF明确GERP>2、Gnocchi≥4、SpliceAI>.5、REVEL>.5；JARVIS条件和完整TableS24需与真实注释生成记录核对。
7. 输出：REGENIE原格式LOCO/pred.list、逐表型.regenie/.ids，gene附MASKS标头；可选masks BED/BIM/FAM/snplist；独立JSON保存参数、输入身份、时间、显存、CPU尾积分回退计数。详情见README。

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

raw模式排除5SD异常值，对选中样本做可选分位数正态化，然后回归传入混杂因素。调用者需要提供论文的age、age²、sex、age×sex、age²×sex、center、40PC及total brain volume（Field26521例外）；fMRI另有motion/table position。数值矩阵中的center等类别列需先编码。

`mode="residual"`只筛缺失，不重新回归。REGENIE阶段RINT由Step1/Step2各自的`apply_rint`控制，采用平均并列rank和Blom公式。`WGSConfig.paper(apply_rint=False)`统一关闭三个阶段，raw模式的分位数正态化也关闭；两个选项可单独修改。

`run_discovery`的raw模式从`DiscoveryInputs.covariates`读取与筛选后芯片样本对齐的`[N,C]`数值tensor，并将残差散射回原行；缺失行保留以复现原Step1折叠行为。

## Step1

```python
from torchwgs import BedReader, Step1Config, fit_null

array_reader = BedReader(array_prefix, keep=discovery_keep_file, remove=sample_remove_file)
null_model = fit_null(
    array_reader, phenotype_values,
    config=Step1Config(device="cuda", block_size=1000, apply_rint=True),
    variant_indices=qc_variant_indices, phenotype_name="24485-2.0",
    output_dir="/results/discovery/Step1",
)
null_model.export_regenie("/results/discovery/Step1/discovery")
```

`phenotype_values`为reader顺序的`[N]`，`qc_variant_indices`为原BIM索引。可传torch/NumPy `[N,M]`矩阵，另提供`chromosomes[M]`和`sample_ids`。返回`NullModel`有sample_ids、sample_indices、loco[N有效,22]、prs[N有效]、y_scale和metadata；`save/load`保存本地缓存，`from_regenie`读取既有原格式LOCO。使用导入模型时应同时核对原cohort、表型、RINT和软件版本，原LOCO文本本身不包含完整来源记录。

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
| max_gpu_gb | 20；根据估计工作空间限制GPU分配 |
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
    config=SingleVariantConfig(maf_min=.001, min_mac=20),
    output_path="/results/discovery/single_24485-2.0.regenie",
)
```

context包含规范化残差、y尺度、LOCO后残差尺度、协变量basis、样本顺序与有效索引。QT score统计用原n−q归一化，输出χ²1尾概率，不改成普通OLS的Student-t。`SingleVariantConfig`有block_size、maf_min（严格大于）、min_mac、apply_rint、device、dtype、tf32。低层函数收到已构建context时使用context的RINT/dtype设置；这些配置在pipeline负责构建context。无output_path时返回DataFrame，只适合有界结果，全基因组应流式写文件。

## Gene-based

```python
from torchwgs import GeneConfig, test_gene_based
from torchwgs.output import RegenieWriter
from torchwgs.masks import load_mask_definitions

mask_definitions = load_mask_definitions(mask_definition_file)
gene_configuration = GeneConfig(aaf_bins=(.01,), vc_max_aaf=.01, min_mac=1)
with RegenieWriter("/results/discovery/gene", "24485-2.0", masks=mask_definitions,
                   sample_ids=association_context.sample_ids) as result_writer:
    for result_row in test_gene_based(
        wgs_reader, association_context, annotation_file, setlist_file,
        mask_definitions, gene_configuration,
    ):
        result_writer.write(result_row)
```

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
| max_matrix_bytes | 8GiB；gene矩阵/协方差工作空间上限 |
| genotype_orientation | allele1；同原REGENIE计BIM A1。variant_id可显式核对chr:pos:REF:ALT并翻转，alt用于已定向矩阵 |
| skato_rhos | (0,.01,.04,.09,.16,.25,.5,1)；内部按冻结原版clip最高端点 |
| tail_method | regenie；兼容强尾/回退策略，也可设exact进行数学审查 |
| acato_full / gene_p / run_sbat | True；开启完整组合、GENE-P和SBAT |
| gene_p_groups | None；可定义GENE-P组合mask分组 |
| rank_tolerance | 1e−7；共线mask rank处理 |
| qr_tie_tolerance | 0；用冻结SSE2范数顺序选pivot；正值可显式指定相对容差内按输入顺序选列 |
| genotype_scale_tolerance | 1e−6；排除投影后低于原numtol的常数或近常数mask |
| sbat_max_subsets / sbat_qmc_samples / sbat_seed | 10 / 8192 / 0；chi-bar子集数、正交积分Sobol样本数和seed。0个子集表示完整枚举。8192为本实现积分设置，论文未规定；可提高精度 |
| beta_a / beta_b | 1 / 25；Beta(MAF)权重，使用GPU密度公式，可修改 |

输出12类TEST：ADD、ADD-SKAT、ADD-SKATO、ADD-SKATO-ACAT、ADD-ACATO、ADD-ACATV、ADD-ACATV-ACAT、ADD-BURDEN-ACAT、ADD-BURDEN-SBAT、ADD-BURDEN-SBAT_POS、ADD-BURDEN-SBAT_NEG、GENE_P。

gene的稀疏小矩阵投影与推断保持GPU float64，使用context中未丢精度的y_float64和covariates_q_float64；single仍按请求dtype计算大块。SBAT在原基因型样本位置补零，再以独立Triton kernel复现冻结Eigen/SSE2的列范数累加顺序，减少共线mask的pivot差异。CPU模式用相同递推。该兼容顺序针对本次核对的3.4.1二进制，其他ISA构建可能选择不同的等范数列。

Sub的global白名单在BIM查找、BED解码和GPU传输前筛选。输出的df=1等效χ²在−log10(P)≤300时使用float64正态逆CDF，接近P=1时用expm1/erfinv；其他df和更强尾保留log域二分算法。该优化减少逐行GPU调用，不改变关联检验或尾积分预算。

## Pipeline和汇总

`DiscoveryInputs`逐项作用见README调用：array_prefix、phenotype_file/column、discovery_samples必填；wgs_prefixes指定运行染色体；array_variant_include/sample_remove可选；gene_analyses指定每条Main/Sub；covariates可选数值tensor；imported_loco可选导入已核对的LOCO。

输入JSON含本地路径；若直接填covariates数组，还包含参与者协变量。配置和输入JSON应存放在自己的私有目录，不能作为公开示例发布。残差表型默认不需要协变量数组。

已有研究Anno_New布局可调用`study_gene_analyses(annotation_root)`构造每条染色体11 Main和12 Sub入口，默认检查所有文件存在。`main_types`、`sub_combinations`、`chromosomes`均可改；评分名字代表输入白名单，不推测JARVIS等未核对数值阈值。完整mask列表与Table S24的一致性应由注释生成来源确认。

`WGSConfig`含step1、single_variant、gene_based、significance四组参数，以及phenotype_mode、phenotype_quantile_normalize、phenotype_outlier_sd、gzip_output、write_samples、split_by_pheno、write_masks、keep_uncompressed_inputs。

`SignificanceConfig`公开effective_phenotypes=831.50、n_genes=17863、variant_alpha=5e−9、gene_alpha=.05、lead_window_bp=500000、locus_merge_bp=1000000。运行一个表型时仍用论文全研究阈值；修改这些参数只重汇总。缓存核对输入身份、运行参数和实现hash：改MAF/MAC重single；改annotation/mask/白名单重gene；改样本、表型、芯片QC、RINT或ridge重Step1。

缓存还检查结果、ID、mask文件的存在和身份。≤64MiB文件核对SHA-256，大文件核对路径/大小/纳秒mtime；修改大输入后应保留mtime变化或使用resume=False。输出先写.partial，计算异常时删除未完成部分；完成后提交原文件名。`.log`与JSON记录PyTorch执行信息。

## 对应原软件验证命令

以下是开发验证入口，生产pipeline不执行这些命令。3.4.1不能同时传--keep与--remove，先生成已取差集的discovery_final.keep。

```bash
regenie --step 1 --bed /data/array --extract /data/qc.snplist \
  --keep /data/discovery_final.keep --phenoFile /data/residuals.txt \
  --phenoCol 24485-2.0 --qt --apply-rint --bsize 1000 --lowmem \
  --threads 8 --out /reference/discovery_step1

regenie --step 2 --bed /data/wgs/chr5 --phenoFile /data/residuals.txt \
  --phenoCol 24485-2.0 --pred /reference/discovery_step1_pred.list \
  --qt --apply-rint --bsize 1000 --minMAC 20 --write-samples \
  --threads 8 --out /reference/discovery_single_c5

regenie --step 2 --bed /data/wgs/chr5 --phenoFile /data/residuals.txt \
  --phenoCol 24485-2.0 --pred /reference/discovery_step1_pred.list \
  --qt --apply-rint --bsize 1000 --minMAC 1 \
  --aaf-bins .01 --vc-maxAAF .01 --vc-tests skato,acato-full --rgc-gene-p \
  --anno-file /data/annotation/PTV_chr5.txt --set-list /data/annotation/chr5_PTV.setlist \
  --mask-def /data/annotation/Mask_PTV.txt --write-samples \
  --threads 8 --out /reference/discovery_gene_c5
```

原软件关闭RINT时去掉`--apply-rint`；Sub加`--extract`评分白名单。single正文MAF筛选与初始minMAC扫描区分。原实现与参考文献见README。
