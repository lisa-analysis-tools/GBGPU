"""JAX implementations of GB-specific waveform / WDM / heterodyne machinery.

Absorbed from ``fastlisaresponse.jax`` at Phase 3F of the sprint reorg.
Subpackages:

* :mod:`gbgpu.jax.sources` — JAX source classes (`JaxUCBSource`).
* :mod:`gbgpu.jax.wdm`     — WDM-domain JAX kernels (`gb_wdm_*`, `gb_wdm_het_*`).

Generic LISA-physics JAX (response, orbits, WDM lookup tables) lives in
:mod:`lisatools.jax` (LISAanalysistools). Subpackage import is gated on
``import jax``.
"""
from __future__ import annotations

try:
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401
    _HAS_JAX = True
except (ImportError, ModuleNotFoundError):
    _HAS_JAX = False

if _HAS_JAX:
    from . import sources, wdm

    __all__ = ["sources", "wdm"]
else:
    __all__ = []
