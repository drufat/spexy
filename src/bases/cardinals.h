// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#include "utils.h"

constexpr double pi = M_PI;
constexpr double half = 0.5;

/*
###########################
# Chebyshev Polynomials
###########################
*/

double T(long n, double x);

double U(long n, double x);

double Uclamp(long n, double x);

double Tclamp(long n, double x);

/*
###########################
# First Derivatives
###########################
*/

double dT(long n, double x);

double dU(long n, double x);

double dUclamp(long n, double x);

double dTclamp(long n, double x);

/*
###########################
# Second Derivatives
###########################
*/

double ddT(long n, double x);

double ddU(long n, double x);

double ddUclamp(long n, double x);

double ddTclamp(long n, double x);

/*
###########################
# Roots
###########################
*/

double xT(long N, double n);

double xU(long N, double n);

double xUclamp(long N, double n);

double xTclamp(long N, double n);

/*
###########################
# Vertex Cardinal Functions
###########################
*/

template <decltype(T) P, decltype(dT) dP, decltype(xT) xP>
double C(long N, long n, double x) {
  auto xn = xP(N, n);
  if (Eq(x, xn)) return 1.0;
  return P(N, x) / (x - xn) / dP(N, xn);
}

#define CT C<T, dT, xT>
#define CU C<U, dU, xU>
#define CUclamp C<Uclamp, dUclamp, xUclamp>
#define CTclamp C<Tclamp, dTclamp, xTclamp>

/*
###########################################
# Derivative of Vertex Cardinal Functions
###########################################
*/

template <decltype(T) P, decltype(dT) dP, decltype(ddT) ddP, decltype(xT) xP>
double dC(long N, long n, double x) {
  auto xn = xP(N, n);
  if (Eq(x, xn)) return ddP(N, xn) / 2 / dP(N, xn);
  return (dP(N, x) * (x - xn) - P(N, x)) / pow(x - xn, 2) / dP(N, xn);
}

#define dCT dC<T, dT, ddT, xT>
#define dCU dC<U, dU, ddU, xU>
#define dCUclamp dC<Uclamp, dUclamp, ddUclamp, xUclamp>
#define dCTclamp dC<Tclamp, dTclamp, ddTclamp, xTclamp>

/*
###########################
# Edge Cardinal Functions
###########################
*/

template <decltype(dC<T, dT, ddT, xT>) dCP>
double D(long N, long m, double x) {
  auto sum = 0.0;
  for (long n = 0; n < m + 1; n++) sum += dCP(N, n, x);
  return -sum;
}

#define DT D<dCT>
#define DU D<dCU>
#define DUclamp D<dCUclamp>
#define DTclamp D<dCTclamp>

/*
######################################
# Normalized Edge Cardinal Functions
######################################
*/

template <decltype(D<dC<T, dT, ddT, xT>>) DP, decltype(xT) xP>
double Dn(long N, long n, double x) {
  return DP(N, n, x) * (xP(N, n + 1) - xP(N, n));
}

#define DnT Dn<DT, xT>
#define DnU Dn<DU, xU>
#define DnUclamp Dn<DUclamp, xUclamp>
#define DnTclamp Dn<DTclamp, xTclamp>
