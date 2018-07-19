// Copyright (C) 2010-2018 Dzhelil S. Rufat. All Rights Reserved.
#include <bases/circular.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#define vectorize(name) m.def(#name, py::vectorize(name))

PYBIND11_MODULE(nat, m) {
  m.attr("half") = py::float_(half);
  m.attr("pi") = py::float_(pi);

  m.def("S", [](py::object o) -> py::object { return o; });

  vectorize(points_periodic);
  vectorize(points_regular_clamped);
  vectorize(points_chebyshev);
  vectorize(points_chebyshev_clamped);
  vectorize(points_regnew);
  vectorize(points_chebnew);

  vectorize(phi);
  vectorize(phi_grad);
  vectorize(phi_star);

  vectorize(correction0);

  vectorize(kappa);
  vectorize(kappa_grad);
  vectorize(kappa_star);
  vectorize(kappa_A_star);

  vectorize(psi);
  vectorize(psi_grad);
  vectorize(psi_star);

  vectorize(correctiond1);
  vectorize(correctionpsid1);
}
