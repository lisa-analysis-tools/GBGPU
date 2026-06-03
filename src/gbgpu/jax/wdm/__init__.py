"""GB-specific WDM-domain JAX kernels.

Absorbed from ``fastlisaresponse.jax.wdm`` at Phase 3F (2026-06-02).
Exposes:

* :func:`gb_wdm_get_ll_jax` / :func:`gb_wdm_fill_global_jax` /
  :func:`gb_wdm_swap_ll_jax`         — direct WDM kernels.
* :func:`gb_wdm_het_get_ll_jax` / :func:`gb_wdm_het_fill_global_jax` /
  :func:`gb_wdm_het_swap_ll_jax`     — chunked-heterodyne kernels.
* :func:`gb_chunk_fd_to_wdm_jax`,
  :func:`fast_wdm_inner_heterodyne_jax`,
  :data:`ALPHA_AUTO`                 — fast_inner_heterodyne helpers.

Generic WDM infrastructure (WaveletLookupTableWrapJAX,
WDMSettingsWrapJAX, WDMDomainWrapJAX, fast_wdm_inner_jax) lives in
:mod:`lisatools.jax.wdm`. Subpackage import is gated on ``import jax``.
"""
from __future__ import annotations

from .kernels import (
    gb_wdm_get_ll_jax,
    gb_wdm_fill_global_jax,
    gb_wdm_swap_ll_jax,
)
from .heterodyne_kernels import (
    gb_wdm_het_get_ll_jax,
    gb_wdm_het_fill_global_jax,
    gb_wdm_het_swap_ll_jax,
)
from .fast_inner_heterodyne import (
    ALPHA_AUTO,
    fast_wdm_inner_heterodyne_jax,
    gb_chunk_fd_to_wdm_jax,
)

__all__ = [
    "gb_wdm_get_ll_jax",
    "gb_wdm_fill_global_jax",
    "gb_wdm_swap_ll_jax",
    "gb_wdm_het_get_ll_jax",
    "gb_wdm_het_fill_global_jax",
    "gb_wdm_het_swap_ll_jax",
    "fast_wdm_inner_heterodyne_jax",
    "gb_chunk_fd_to_wdm_jax",
    "ALPHA_AUTO",
]
