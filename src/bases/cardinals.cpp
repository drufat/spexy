// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#include "cardinals.h"

using namespace std;

/*
###########################
# Chebyshev Polynomials
###########################
*/

double T(long n, double x) {
  auto theta = acos(x);
  return cos(n * theta);
}

double U(long n, double x) {
  if (x >= +1.0) return n + 1;
  if (x <= -1.0) return (n + 1) * pow(-1, n);
  auto theta = acos(x);
  return sin((n + 1) * theta) / sin(theta);
}

double Uclamp(long n, double x) { return (T(n - 2, x) - T(n, x)) / 2.0; }

double Tclamp(long n, double x) {
  return T(n - 2, x) / 2.0 - (T(fabs(n - 4), x) + T(n, x)) / 4.0;
}

/*
###########################
# First Derivatives
###########################
*/

double dT(long n, double x) { return n * U(n - 1, x); }

double dU(long n, double x) {
  auto b = [](double n) -> double { return n * (n + 1) * (n + 2) / 3; };
  if (Eq(x, -1.0)) return b(n) * pow(-1, n + 1);
  if (Eq(x, +1.0)) return b(n);
  return ((n + 1) * T(n + 1, x) - x * U(n, x)) / (x * x - 1);
}

double dUclamp(long n, double x) { return (dT(n - 2, x) - dT(n, x)) / 2.0; }

double dTclamp(long n, double x) {
  return dT(n - 2, x) / 2.0 - (dT(fabs(n - 4), x) + dT(n, x)) / 4.0;
}

/*
###########################
# Second Derivatives
###########################
*/

double ddT(long n, double x) { return n * dU(n - 1, x); }

double ddU(long n, double x) {
  auto b = [](long n) -> double {
    return (n - 1) * n * (n + 1) * (n + 2) * (n + 3) / 15.0;
  };
  if (Eq(x, -1.0)) return b(n) * pow(-1, n);
  if (Eq(x, +1.0)) return b(n);
  return (n * (n + 2) * U(n, x) - 3 * x * dU(n, x)) / (x * x - 1);
}

double ddUclamp(long n, double x) { return (ddT(n - 2, x) - ddT(n, x)) / 2; }

double ddTclamp(long n, double x) {
  return ddT(n - 2, x) / 2 - (ddT(fabs(n - 4), x) + ddT(n, x)) / 4;
}

/*
###########################
# Roots
###########################
*/

double xT(long N, double n) { return -cos(pi * (n + half) / N); }

double xU(long N, double n) { return -cos(pi * (n + 1) / (N + 1)); }

double xUclamp(long N, double n) {
  if (n <= 0) return -1;
  if (n >= N - 1) return +1;
  return -cos(pi * n / (N - 1));
}

double xTclamp(long N, double n) {
  if (n <= 0) return -1;
  if (n >= N - 1) return +1;
  return -cos(pi * (n - half) / (N - 2));
}
