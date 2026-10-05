"""Paper defaults with independent, editable analysis parameters."""
from dataclasses import dataclass, field, asdict
from .step1 import Step1Config
from .single import SingleVariantConfig
from .masks import GeneConfig, FrequencyDomain
from .execution import ExecutionConfig


@dataclass
class SignificanceConfig:
    effective_phenotypes: float = 831.50
    n_genes: int = 17863
    variant_alpha: float = 5e-9
    gene_alpha: float = .05
    lead_window_bp: int = 500000
    locus_merge_bp: int = 1000000
    single_frequency_min: float = .001
    single_frequency_field: str = 'a1freq'
    excluded_locus_regions: tuple = ((6, 25000000, 34000000),)

    def __post_init__(self):
        if self.effective_phenotypes<=0 or self.n_genes<1 or not 0<self.variant_alpha<=1 or not 0<self.gene_alpha<=1:
            raise ValueError('Positive study multiplicities and alpha in (0,1] required')
        if self.lead_window_bp<0 or self.locus_merge_bp<0:raise ValueError('Genomic distances must be nonnegative')
        if not 0<=self.single_frequency_min<.5 or self.single_frequency_field not in ('a1freq','maf'):
            raise ValueError('Frequency threshold in [0,.5) and field a1freq/maf required')
        regions = []
        for region in self.excluded_locus_regions:
            if len(region) != 3:
                raise ValueError('Excluded locus regions require chromosome, start and end')
            chromosome, start, end = region
            if any(isinstance(value, bool) or not isinstance(value, int) for value in region):
                raise ValueError('Excluded locus region coordinates must be integers')
            if chromosome < 1 or start < 1 or end < start:
                raise ValueError('Excluded locus regions require a positive chromosome and 1 <= start <= end')
            regions.append((chromosome, start, end))
        self.excluded_locus_regions = tuple(regions)

    @property
    def single_threshold(self): return self.variant_alpha/self.effective_phenotypes
    @property
    def gene_threshold(self): return self.gene_alpha/(self.effective_phenotypes*self.n_genes)


@dataclass
class WGSConfig:
    step1: Step1Config = field(default_factory=Step1Config)
    single_variant: SingleVariantConfig = field(default_factory=SingleVariantConfig)
    gene_based: GeneConfig = field(default_factory=GeneConfig)
    significance: SignificanceConfig = field(default_factory=SignificanceConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    phenotype_mode: str = 'residual'
    phenotype_quantile_normalize: bool = True
    phenotype_outlier_sd: float = 5.
    gzip_output: bool = False
    write_samples: bool = True
    print_pheno_name: bool = True
    split_by_pheno: bool = True
    write_masks: bool = False
    keep_uncompressed_inputs: bool = False
    bim_index_enabled: bool = True

    def __post_init__(self):
        if not isinstance(self.bim_index_enabled, bool):
            raise ValueError('bim_index_enabled must be a boolean.')

    @classmethod
    def paper(cls, *, apply_rint=True):
        from dataclasses import replace
        result = cls()
        result.step1 = replace(result.step1,apply_rint=apply_rint)
        result.single_variant.apply_rint = apply_rint
        result.single_variant.genotype_reader = 'cuda_packed'
        result.gene_based.apply_rint = apply_rint
        result.gene_based.sbat_subset_sampling = 'with_replacement'
        result.gene_based.vc_storage = 'sparse'
        result.gene_based.vc_score_method = 'crossproduct'
        result.gene_based.genotype_reader = 'cuda_packed'
        result.gene_based.eigen_backend = 'auto'
        result.gene_based.skato_integral_backend = 'qags_x'
        result.significance.single_frequency_field = 'maf'
        result.phenotype_quantile_normalize = apply_rint
        result.write_masks = True
        return result

    def to_dict(self): return asdict(self)

    @classmethod
    def from_dict(cls, values):
        d = cls.paper().to_dict()
        parameter_groups={'step1','single_variant','gene_based','significance','execution'}
        for key,value in dict(values).items():
            d[key]={**d[key],**value} if key in parameter_groups and isinstance(value,dict) else value
        if isinstance(d.get('gene_based'),dict):
            gene=dict(d['gene_based'])
            if gene.get('domain_mapping') is not None:
                gene['domain_mapping']={name:FrequencyDomain(**domain) if isinstance(domain,dict) else domain
                                        for name,domain in gene['domain_mapping'].items()}
            d['gene_based']=gene
        for key, kind in [('step1',Step1Config),('single_variant',SingleVariantConfig),
                          ('gene_based',GeneConfig),('significance',SignificanceConfig),('execution',ExecutionConfig)]:
            if key in d: d[key] = kind(**d[key])
        return cls(**d)
