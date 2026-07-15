"""RDT2 action-expert runtime derived from the official Apache-2.0 release.

The upstream source is https://github.com/thu-ml/RDT2. Local changes only
replace repository-relative imports so the released weights can be loaded as
part of the installed ``forge`` package. See ``LICENSE`` in this directory.
"""

from forge.vendor.rdt2.rdt_runner import RDTRunner

__all__ = ["RDTRunner"]
