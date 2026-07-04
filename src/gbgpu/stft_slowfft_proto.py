"""FastGB-STFT prototype: heterodyned slow part on the STFT sub-grid -> narrow
per-segment DFT -> time-resolved inner product. Phase-1 prototype (design
docs/specs/2026-07-04-stft-gb-slowpart-fft-design.md). CPU/GPU via the `xp` of the
computation group; validated vs the injected brute STFT, benchmarked vs Fresnel."""

def get_ll_stft_slowfft_proto(grp, gb, params, n_sub=32):
    xp = gb.xp
    num_bin = int(params.shape[0])
    d_h = xp.zeros(num_bin, dtype=xp.complex128)
    h_h = xp.zeros(num_bin, dtype=xp.complex128)
    return d_h, h_h  # stub; filled in Tasks 2-4
