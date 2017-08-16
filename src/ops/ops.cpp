// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#include "ops.h"

#include <fftw3.h>

constexpr double pi(M_PI);
constexpr std::complex<double> J(0, 1);

/*
##############################
# Converting to Fourier Space
##############################
*/

std::function<void(int, int, const double*, double*)>
fourier(void (*F)(int, const std::complex<double>*, std::complex<double>*))
{

  return [F](int M, int N, const double* X, double* Y) -> void {

    auto plan_forw =
      fftw_plan_dft_r2c_1d(N, NULL, NULL, FFTW_ESTIMATE); // | FFTW_UNALIGNED);
    auto plan_back =
      fftw_plan_dft_c2r_1d(N, NULL, NULL, FFTW_ESTIMATE); // | FFTW_UNALIGNED);

    std::vector<std::complex<double>> W(N);

#pragma omp parallel for firstprivate(W) if (M > MINPARALLEL)
    for (int j = 0; j < M; j++) {
      fftw_execute_dft_r2c(  //
        plan_forw,           //
        (double*)&X[j * N],  //
        (fftw_complex*)&W[0] //
        );

      F(N, &W[0], &W[0]);
      for (int i = 0; i < N; i++) {
        W[i] /= N;
      }

      fftw_execute_dft_c2r(   //
        plan_back,            //
        (fftw_complex*)&W[0], //
        (double*)&Y[j * N]    //
        );
    }

    fftw_destroy_plan(plan_forw);
    fftw_destroy_plan(plan_back);
  };
}

/*
################
# Fourier Ops
################
*/

int
freq(int N, int i)
{
  if (i < (N + 1) / 2) {
    return i;
  } else {
    return i - N;
  }
}

double
H_hat(int N, int i)
{
  auto k = freq(N, i);
  if (k == 0) {
    return 2 * pi / N;
  }
  return sin(pi * k / N) * 2 / k;
}

std::complex<double>
Q_hat(int N, int i)
{
  auto k = freq(N, i);
  if (k == 0) {
    return 2 * pi / N;
  }
  return (exp(2.0 * J * (double)k * pi / (double)N) - 1.0) / (J * (double)k);
}

std::complex<double>
S_hat(int N, int i, int s)
{
  auto k = freq(N, i);
  return exp((double)s * J * (double)k * pi / (double)N);
}

std::complex<double>
G_hat(int N, int i)
{
  auto k = freq(N, i);
  return J * (double)k;
}

/*
################
# H
################
*/

void
H(int N, const std::complex<double>* X, std::complex<double>* Y)
{
  for (int i = 0; i < N; i++) {
    Y[i] = X[i] * H_hat(N, i);
  }
}

void
Hinv(int N, const std::complex<double>* X, std::complex<double>* Y)
{
  for (int i = 0; i < N; i++) {
    Y[i] = X[i] / H_hat(N, i);
  }
}

/*
################
# Q
################
*/

void
Q(int N, const std::complex<double>* X, std::complex<double>* Y)
{
  for (int i = 0; i < N; i++) {
    Y[i] = X[i] * Q_hat(N, i);
  }
}

void
Qinv(int N, const std::complex<double>* X, std::complex<double>* Y)
{
  for (int i = 0; i < N; i++) {
    Y[i] = X[i] / Q_hat(N, i);
  }
}

/*
################
# S
################
*/

void
S(int N, const std::complex<double>* X, std::complex<double>* Y)
{
  for (int i = 0; i < N; i++) {
    Y[i] = X[i] * S_hat(N, i, +1);
  }
}

void
Sinv(int N, const std::complex<double>* X, std::complex<double>* Y)
{
  for (int i = 0; i < N; i++) {
    Y[i] = X[i] * S_hat(N, i, -1);
  }
}

/*
################
# G
################
*/

void
G(int N, const std::complex<double>* X, std::complex<double>* Y)
{
  for (int i = 0; i < N; i++) {
    Y[i] = X[i] * G_hat(N, i);
  }
}
