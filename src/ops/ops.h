// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#ifndef DECOPS
#define DECOPS

#include <cmath>
#include <complex>
#include <functional>
#include <vector>

#define MINPARALLEL 32

/*
################
# Helpers
################
*/

// positive modulo
inline int mod(int k, int n) { return (k % n + n) % n; }

template <class T>
void diff(int Nin, const T* in, int Nout, T* out) {
  for (int i = 0; i < Nout; i++) {
    out[i] = in[i + 1] - in[i];
  }
}

template <class T>
void roll(int n, int Nin, const T* in, int Nout, T* out) {
  for (int i = 0; i < Nin; i++) {
    out[mod(i + n, Nout)] = in[i];
  }
}

template <class T>
void slice_(int begin, int step, int Nin, const T* in, int Nout, T* out) {
  for (int i = 0; i < Nout; i++) {
    out[i] = in[begin + i * step];
  }
}

template <class T>
void weave(int N0, const T* in0, int N1, const T* in1, int N, T* out) {
  for (int i = 0; i < N0; i++) {
    out[2 * i] = in0[i];
  }
  for (int i = 0; i < N1; i++) {
    out[2 * i + 1] = in1[i];
  }
}

template <class T>
void concat(int N0, const T* in0, int N1, const T* in1, int N, T* out) {
  for (int i = 0; i < N0; i++) {
    out[i] = in0[i];
  }
  for (int i = 0; i < N1; i++) {
    out[i + N0] = in1[i];
  }
}

/*
##############################
# Converting to Fourier Space
##############################
*/

std::function<void(int, int, const double*, double*)> fourier(
    void (*F)(int, const std::complex<double>*, std::complex<double>*));

/*
################
# Fourier Ops
################
*/

int freq(int N, int i);

double H_hat(int N, int i);

std::complex<double> S_hat(int N, int i);

std::complex<double> Q_hat(int N, int i);

/*
################
# H
################
*/

void H(int N, const std::complex<double>* X, std::complex<double>* Y);

void Hinv(int N, const std::complex<double>* X, std::complex<double>* Y);

/*
################
# Q
################
*/

void Q(int N, const std::complex<double>* X, std::complex<double>* Y);

void Qinv(int N, const std::complex<double>* X, std::complex<double>* Y);

/*
################
# S
################
*/

void S(int N, const std::complex<double>* X, std::complex<double>* Y);

void Sinv(int N, const std::complex<double>* X, std::complex<double>* Y);

/*
################
# G
################
*/

void G(int N, const std::complex<double>* X, std::complex<double>* Y);

#endif  // DECOPS
