// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#ifndef UTILS_H
#define UTILS_H

#include <cmath>

template <class T>
bool Eq(T a, T b) {
  constexpr T rtol = 1e-05;
  constexpr T atol = 1e-08;
  return fabs(a - b) <= (atol + rtol * fabs(b));
}

#endif  // UTILS_H
