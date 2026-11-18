#ifndef Constants_h
#define Constants_h



/* --------------  MATHEMATICAL CONSTANTS  -------------- */
/* Square root of 3 */
#define SQ3   1.7320508075688772

/* Pi's and freinds */
//use math.h (M_PI) for PI
#define PI2   6.283185307179586
#define PIon2 1.5707963267948966
#define PIon4 0.7853981633974483

/* Natural log of 2 */
#define LN2   0.6931471805599453



/* ----------------  NATURAL CONSTANTS  ----------------- */

/* Speed of light (m/s) */
#define Clight 299792458.

/* Mass of the Sun (s) */
#define TSUN  4.925490947641158e-06

/* Number of meters in a parsec */
#define PC 3.085677581491367e16

/* Number of seconds in a year */
#define YEAR 31558149.7635456

/* Astronomical unit (meters) */
#define AU 1.49597870700e11

#define MSUN 1.9884754153381438e30
#define Mjup 1.89819e27
#define G 6.67408e-11


/* Photon shot noise power */
#define Sps 8.321000e-23

/* Acceleration noise power */
#define Sacc 9.000000e-30

/* Mean arm length of constellation (m) */
#define Larm 2.5e9

 /* LISA orbital eccentricity */
#define ec 0.004824185218078991

 /* Initial azimuthal position of the guiding center */
#define kappa 0.000000

 /* Initial orientation of the LISA constellation */
#define lambda0 0.000000

 /* LISA modulation frequency */
#define fm 3.168753578738106e-08

/* transfer frequency (Hz) */
#define fstar 0.01908538063694777

/* MLDC sampling rate */
//#define dt 15.000000

/* Observation period */
//#define Tobs 31457280.000000// 125829120.00000//31457280.000000


//void instrument_noise(double f, double *SAE, double *SXYZ);

//void spacecraft(double t, double *x, double *y, double *z);





#endif /* Constants_h */
