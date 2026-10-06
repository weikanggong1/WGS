The optional GDS adapters are original code distributed under GPL-3.0-only.
They use the user's existing official PyGDS/CoreArray installation; no upstream
source tree, headers, shared libraries, genotype data or analysis output is bundled.

PyGDS, copyright Xiuwen Zheng, is distributed under GNU GPL version 3:
https://github.com/CoreArray/pygds

The CoreArray headers used by the optional packed adapter, copyright Xiuwen
Zheng, declare GNU LGPL version 3:
https://github.com/CoreArray/pygds/tree/master/src/CoreArray

The packed adapter uses internal C++ allocator layout, outside the stable
public capsule interface. It requires an explicitly configured local build
bound to the installed SDK binary, its corresponding source-build binary,
official headers, platform and compiler layout. A matching package version
alone does not establish compatibility. Build manifests and compiled adapters
are local artifacts and are not included in this source distribution.
