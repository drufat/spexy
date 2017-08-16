// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#ifndef BASESNATIVE
#define BASESNATIVE

#include <cmath>

constexpr double pi = M_PI;
constexpr double half = 0.5;

/*
#########################
# Chebyshev Polynomials
#########################
*/

double T(long N, double x);

double U(long N, double x);

/*
########################
# Fourier Coefficients
########################
*/

double h(long N);

double coef_f(long N, double n);

double coef_a(long N, double n);

double coef_a_star(long N, double n);

double coef_k(long N, double n, double m);

double coef_k_star(long N, double n, double m);

double coef_p_star(long N, double n, double m);

/*
############
# Points
############
*/

double points_periodic(long N, double i);

double points_regular(long N, double i);

double points_chebyshev(long N, double i);

double points_regnew(long N, double i);

double points_chebnew(long N, double i);

double clamp(double xmin, double xmax, double x);

double points_regular_clamped(long N, double i);

double points_chebyshev_clamped(long N, double i);

/*
########################################
# Mapping between semi-circle and line #
########################################
*/

double varphi(double x);

double varphi_inv(double x);

/*
########################################
# Periodic Basis Functions
########################################
*/

double phi_compact(long N, double x);

double phi(long N, double x);

double phi_grad(long N, double x);

double phi_star(long N, double x);

/*
############################################################
# Regular Basis Functions
############################################################
*/

double gamma(long N, double n);

double delta(long N, double x);

double deltapsi(long N, double x);

double correction0(long N, double n);

double correctiond1(long N, double n, double x);

double correctionpsid1(long N, double n, double x);

double kappa(long N, double n, double x);

double kappa_grad(long N, double n, double x);

double kappa_A_star(long N, double n, double x);

double kappa_star(long N, double n, double x);

/*
############################################################
# Chebyshev Basis Functions
############################################################
*/

double psi(long N, double n, double x);

double psi_grad(long N, double n, double x);

double psi_star(long N, double n, double x);

#endif // BASESNATIVE
