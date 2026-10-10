# Algorithm attribution and licensing

Fudan WGS Toolkit is distributed under GPL-3.0-only. Its Python implementations
of weighted rare-variant score tests, annotation masks, Cauchy combinations,
saddlepoint tails, and mixed-model arithmetic follow scientific algorithms by
Xihao Li, Zilin Li, Yuxin Yuan, and their collaborators.

Original algorithm sources and GPL attribution:

- [STAAR](https://github.com/li-lab-genetics/STAAR)
- [STAARpipeline](https://github.com/li-lab-genetics/STAARpipeline)
- [STAARpipelinePheWAS](https://github.com/li-lab-genetics/STAARpipelinePheWAS)
- [Frozen statistical reference](https://github.com/yuxinyuanqt/STAAR/tree/4bbf77ba8a90894a434f5eb4473d540e172dad05)

These names identify the scientific sources. They are not installed namespaces
or runtime dependencies. The distribution contains Python source only; it does
not bundle original software source, participant data, downloaded annotations,
or computed association tables.

PLINK file formats are documented by the [PLINK project](https://www.cog-genomics.org/plink/1.9/formats).
The converter reads binary files with Python and NumPy and does not invoke PLINK.
The independent FAVOR Essential annotation dataset is CC0-1.0, according to its
[Dataverse record](https://doi.org/10.7910/DVN/1VGTJI). Downloaded files remain external
inputs and must retain their own provenance and checksums.
