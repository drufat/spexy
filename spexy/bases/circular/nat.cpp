// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#include <bases/circular.h>
#include <pybindcpp/module.h>
#include <pybindcpp/numpy.h>

using namespace pybindcpp;

void
basesnative(ExtModule& m)
{
  m.var("half", half);
  m.var("pi", pi);

  m.fun("S", [](PyObject* o) -> PyObject* {
    Py_IncRef(o);
    return o;
  });

  ufunc(m, "points_periodic", points_periodic);
  ufunc(m, "points_regular", points_regular);
  ufunc(m, "points_regular_clamped", points_regular_clamped);
  ufunc(m, "points_chebyshev", points_chebyshev);
  ufunc(m, "points_chebyshev_clamped", points_chebyshev_clamped);
  ufunc(m, "points_regnew", points_regnew);
  ufunc(m, "points_chebnew", points_chebnew);

  ufunc(m, "phi", phi);
  ufunc(m, "phi_grad", phi_grad);
  ufunc(m, "phi_star", phi_star);

  ufunc(m, "correction0", correction0);

  ufunc(m, "kappa", kappa);
  ufunc(m, "kappa_grad", kappa_grad);
  ufunc(m, "kappa_star", kappa_star);
  ufunc(m, "kappa_A_star", kappa_A_star);

  ufunc(m, "psi", psi);
  ufunc(m, "psi_grad", psi_grad);
  ufunc(m, "psi_star", psi_star);

  ufunc(m, "correctiond1", correctiond1);
  ufunc(m, "correctionpsid1", correctionpsid1);
}

PyMODINIT_FUNC
PyInit_nat(void)
{
  import_array();
  import_ufunc();
  return module_init("nat", basesnative);
}
