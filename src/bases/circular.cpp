// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#include "circular.h"
#include "utils.h"

using namespace std;

/*
#########################
# Chebyshev Polynomials
#########################
*/

double
T(long N, double x)
{
  return cos(N * acos(x));
}

double
U(long N, double x)
{
  if (Eq(x, +1.0))
    return N + 1;
  if (Eq(x, -1.0))
    return (N + 1) * pow(-1, N);
  return sin((N + 1) * acos(x)) / sin(acos(x));
}

/*
########################
# Fourier Coefficients
########################
*/

double
h(long N)
{
  return 2 * pi / N;
}

double
coef_f(long N, double n)
{
  if (Eq(n, 0.0))
    return h(N);
  else
    return 2 * sin(n * h(N) / 2) / n;
}

double
coef_a(long N, double n)
{
  return 1.0 / N;
}

double
coef_a_star(long N, double n)
{
  return coef_a(N, n) / coef_f(N, n);
}

double
coef_k(long N, double n, double m)
{
  return coef_a(2 * N, m) * 2 * cos(pi * m * n / N);
}

double
coef_k_star(long N, double n, double m)
{
  return coef_a_star(2 * N, m) * 2 * cos(pi * m * n / N);
}

double
coef_p_star(long N, double n, double m)
{
  return coef_a_star(2 * N, m) * 2 * sin(pi * m * n / N);
}

/*
############
# Points
############
*/

double
points_periodic(long N, double i)
{
  double h = 2 * pi / N;
  return i * h;
}

double
points_regular(long N, double i)
{
  double h = pi / N;
  return i * h;
}

double
points_regnew(long N, double i)
{
  return points_regular(N + 1, i + half);
}

double
points_chebyshev(long N, double i)
{
  double x = points_regular(N, i);
  return -cos(x);
}

double
points_chebnew(long N, double i)
{
  double x = points_regnew(N, i);
  return -cos(x);
}

double
clamp(double xmin, double xmax, double x)
{
  if (x < xmin)
    return xmin;
  if (x > xmax)
    return xmax;
  return x;
}

double
points_regular_clamped(long N, double i)
{
  return points_regular(N, clamp(0, N, i));
}

double
points_chebyshev_clamped(long N, double i)
{
  return points_chebyshev(N, clamp(0, N, i));
}

/*
########################################
# Mapping between semi-circle and line #
########################################
*/

double
varphi(double x)
{
  return acos(-x);
}

double
varphi_inv(double x)
{
  return -cos(x);
}

/*
########################################
# Periodic Basis Functions
########################################
*/

double
phi_compact(long N, double x)
{
  if (fmod(x, 2 * pi) == 0)
    return 1;
  else if (fmod(N, 2) == 0)
    return (sin(N * x / 2) / tan(x / 2)) / N;
  else
    return (sin(N * x / 2) / sin(x / 2)) / N;
}

double
phi(long N, double x)
{
  double sum = 0.0;
  for (long k = -floor(N / 2); k < N - floor(N / 2); k++)
    sum += coef_a(N, k) * cos(k * x);
  return sum;
}

double
phi_grad(long N, double x)
{
  double sum = 0.0;
  for (long k = -floor(N / 2); k < N - floor(N / 2); k++)
    sum += coef_a(N, k) * (-k * sin(k * x));
  return sum;
}

double
phi_star(long N, double x)
{
  double sum = 0.0;
  for (long k = -floor(N / 2); k < N - floor(N / 2); k++)
    sum += coef_a_star(N, k) * cos(k * x);
  return sum;
}

/*
############################################################
# Regular Basis Functions
############################################################
*/

double
gamma(long N, double n)
{
  double sum = 0.0;
  for (long m = -N; m < N; m++)
    sum += sin(pi * n * m / N) * tan(pi * m / 4 / N) / 2 / N;
  return sum;
}

double
delta(long N, double x)
{
  // return N * (1 + cos(x)) * sin(N * x) / 2 + sin(x) * cos(N * x) / 2;
  return N * sin(N * x) / 2 + (N + 1) * sin((N + 1) * x) / 4 +
         (N - 1) * sin((N - 1) * x) / 4;
}

double
deltapsi(long N, double x)
{
  return N * U(N - 1, -x) / 2 + (N + 1) * U(N, -x) / 4 +
         (N - 1) * U(N - 2, -x) / 4;
}

double
correction0(long N, double n)
{
  if (n == (double)0)
    return 0.5;
  else if (n == (double)N)
    return 0.5;
  else
    return 1.0;
}

double
correctiond1(long N, double n, double x)
{
  if (n == (double)0)
    return delta(N, x);
  else if (n == (double)N)
    return delta(N, pi - x);
  else
    return (-gamma(N, n) * delta(N, x) - gamma(N, N - n) * delta(N, pi - x));
}

double
correctionpsid1(long N, double n, double x)
{
  if (n == (double)0)
    return deltapsi(N, x);
  else if (n == (double)N)
    return deltapsi(N, -x);
  else
    return (-gamma(N, n) * deltapsi(N, x) - gamma(N, N - n) * deltapsi(N, -x));
}

double
kappa(long N, double n, double x)
{
  double sum = 0.0;
  for (long m = -N; m < N; m++)
    sum += coef_k(N, n, m) * cos(m * x);
  return sum;
}

double
kappa_grad(long N, double n, double x)
{
  double sum = 0.0;
  for (long m = -N; m < N; m++)
    sum += coef_k(N, n, m) * (-m * sin(m * x));
  return sum;
}

double
kappa_A_star(long N, double n, double x)
{
  double sum = 0.0;
  for (long m = -N; m < N; m++)
    sum += coef_p_star(N, n, m) * sin(m * x);
  return sum;
}

double
kappa_star(long N, double n, double x)
{
  double sum = 0.0;
  for (long m = -N; m < N; m++)
    sum += coef_k_star(N, n, m) * cos(m * x);
  return sum;
}

/*
############################################################
# Chebyshev Basis Functions
############################################################
*/

double
psi(long N, double n, double x)
{
  double sum = 0.0;
  for (long m = -N; m < N; m++)
    sum += coef_k(N, n, m) * T(fabs(m), -x);
  return sum;
}

double
psi_grad(long N, double n, double x)
{
  double sum = 0.0;
  for (long m = -N; m < N; m++)
    sum += coef_k(N, n, m) * (-fabs(m) * U(fabs(m) - 1, -x));
  return sum;
}

double
psi_star(long N, double n, double x)
{
  double sum = 0.0;
  for (long m = -N; m < N; m++)
    sum += coef_p_star(N, n, fabs(m)) * U(fabs(m) - 1, -x);
  return sum;
}
