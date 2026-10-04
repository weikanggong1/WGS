"""Paper defaults with independent, editable analysis parameters."""
from dataclasses import dataclass, field, asdict
from .step1 import Step1Config
from .single import SingleVariantConfig
from .masks import GeneConfig, FrequencyDomain


@dataclass
class SignificanceConfig:
    effective_phenotypes: float = 831.50
    n_genes: int = 17863
    variant_alpha: float = 5e-9
    gene_alpha: float = .05
    lead_window_bp: int = 500000
    locus_merge_bp: int = 1000000

    def __post_init__(self):
        if self.effective_phenotypes<=0 or self.n_genes<1 or not 0<self.variant_alpha<=1 or not 0<self.gene_alpha<=1:
            raise ValueError('Positive study multiplicities and alpha in (0,1] required')
        if self.lead_window_bp<0 or self.locus_merge_bp<0:raise ValueError('Genomic distances must be nonnegative')

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
    phenotype_mode: str = 'residual'
    phenotype_quantile_normalize: bool = True
    phenotype_outlier_sd: float = 5.
    gzip_output: bool = False
    write_samples: bool = True
    split_by_pheno: bool = True
    write_masks: bool = False
    keep_uncompressed_inputs: bool = False

    @classmethod
    def paper(cls, *, apply_rint=True):
        from dataclasses import replace
        result = cls()
        result.step1 = replace(result.step1,apply_rint=apply_rint)
        result.single_variant.apply_rint = apply_rint
        result.gene_based.apply_rint = apply_rint
        result.phenotype_quantile_normalize = apply_rint
        return result

    def to_dict(self): return asdict(self)

    @classmethod
    def from_dict(cls, values):
        d = dict(values)
        if isinstance(d.get('gene_based'),dict):
            gene=dict(d['gene_based'])
            if gene.get('domain_mapping') is not None:
                gene['domain_mapping']={name:FrequencyDomain(**domain) if isinstance(domain,dict) else domain
                                        for name,domain in gene['domain_mapping'].items()}
            d['gene_based']=gene
        for key, kind in [('step1',Step1Config),('single_variant',SingleVariantConfig),
                          ('gene_based',GeneConfig),('significance',SignificanceConfig)]:
            if key in d: d[key] = kind(**d[key])
        return cls(**d)
