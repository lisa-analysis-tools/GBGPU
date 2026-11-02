#ifndef __GB_PHYSICS_HH__
#define __GB_PHYSICS_HH__

// gb_physics.hh -- phase, frequency, frequency derivative and amplitude of a galactic binary as
// functions of the time t_diff = t - t_ref from its reference epoch. GBTDIonTheFly (the
// TDI-on-the-fly response) and GBDFFRSource (the DFFR response) both evaluate these, so the two
// responses share one model.
// LDC convention: phi0 enters with a leading minus sign, and the amplitude grows as
// A0 (1 + (2/3) (fdot / f0) t_diff).

#include "gbt_global.h"  // CUDA_CALLABLE_MEMBER

#include <cmath>

CUDA_CALLABLE_MEMBER
inline double gb_phase(double f0, double fdot, double fddot, double phi0, double t_diff) {
  return -phi0 + 2 * M_PI * (f0 * t_diff
                             + 0.5 * fdot * t_diff * t_diff
                             + (1.0 / 6.0) * fddot * t_diff * t_diff * t_diff);
}

CUDA_CALLABLE_MEMBER
inline double gb_frequency(double f0, double fdot, double fddot, double t_diff) {
  return f0 + fdot * t_diff + 0.5 * fddot * t_diff * t_diff;
}

CUDA_CALLABLE_MEMBER
inline double gb_frequency_derivative(double fdot, double fddot, double t_diff) {
  return fdot + fddot * t_diff;
}

// The amplitude over its value A0 at the reference epoch.
CUDA_CALLABLE_MEMBER
inline double gb_amplitude_ratio(double f0, double fdot, double t_diff) {
  return 1.0 + (2.0 / 3.0) * fdot / f0 * t_diff;
}

// d ln A / dt at t_diff.
CUDA_CALLABLE_MEMBER
inline double gb_amplitude_log_slope(double f0, double fdot, double t_diff) {
  const double chirp_fraction = (2.0 / 3.0) * fdot / f0;
  return chirp_fraction / (1.0 + chirp_fraction * t_diff);
}

#endif  // __GB_PHYSICS_HH__
