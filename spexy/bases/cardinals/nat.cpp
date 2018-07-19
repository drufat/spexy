// Copyright (C) 2010-2018 Dzhelil S. Rufat. All Rights Reserved.
#include <bases/cardinals.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#define vectorize(name) m.def(#name, py::vectorize(name))

PYBIND11_MODULE(nat, m) {
  m.attr("half") = py::float_(half);
  m.attr("pi") = py::float_(pi);

  m.def("S", [](py::object o) -> py::object { return o; });

  vectorize(T);
  vectorize(U);
  vectorize(Uclamp);
  vectorize(Tclamp);

  vectorize(dT);
  vectorize(dU);
  vectorize(dUclamp);
  vectorize(dTclamp);

  vectorize(ddT);
  vectorize(ddU);
  vectorize(ddUclamp);
  vectorize(ddTclamp);

  vectorize(xT);
  vectorize(xU);
  vectorize(xUclamp);
  vectorize(xTclamp);

  vectorize(CT);
  vectorize(CU);
  vectorize(CUclamp);
  vectorize(CTclamp);

  vectorize(dCT);
  vectorize(dCU);
  vectorize(dCUclamp);
  vectorize(dCTclamp);

  vectorize(DT);
  vectorize(DU);
  vectorize(DUclamp);
  vectorize(DTclamp);

  vectorize(DnT);
  vectorize(DnU);
  vectorize(DnUclamp);
  vectorize(DnTclamp);
}
