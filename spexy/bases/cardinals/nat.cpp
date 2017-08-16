// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#include <bases/cardinals.h>
#include <pybindcpp/module.h>
#include <pybindcpp/numpy.h>

using namespace pybindcpp;

void
cardinalsnative(ExtModule& m)
{
  m.var("half", half);
  m.var("pi", pi);

  m.fun("S", [](PyObject* o) -> PyObject* {
    Py_IncRef(o);
    return o;
  });

  ufunc(m, "T", T);
  ufunc(m, "U", U);
  ufunc(m, "Uclamp", Uclamp);
  ufunc(m, "Tclamp", Tclamp);

  ufunc(m, "dT", dT);
  ufunc(m, "dU", dU);
  ufunc(m, "dUclamp", dUclamp);
  ufunc(m, "dTclamp", dTclamp);

  ufunc(m, "ddT", ddT);
  ufunc(m, "ddU", ddU);
  ufunc(m, "ddUclamp", ddUclamp);
  ufunc(m, "ddTclamp", ddTclamp);

  ufunc(m, "xT", xT);
  ufunc(m, "xU", xU);
  ufunc(m, "xUclamp", xUclamp);
  ufunc(m, "xTclamp", xTclamp);

  ufunc(m, "CT", CT);
  ufunc(m, "CU", CU);
  ufunc(m, "CUclamp", CUclamp);
  ufunc(m, "CTclamp", CTclamp);

  ufunc(m, "dCT", dCT);
  ufunc(m, "dCU", dCU);
  ufunc(m, "dCUclamp", dCUclamp);
  ufunc(m, "dCTclamp", dCTclamp);

  ufunc(m, "DT", DT);
  ufunc(m, "DU", DU);
  ufunc(m, "DUclamp", DUclamp);
  ufunc(m, "DTclamp", DTclamp);

  ufunc(m, "DnT", DnT);
  ufunc(m, "DnU", DnU);
  ufunc(m, "DnUclamp", DnUclamp);
  ufunc(m, "DnTclamp", DnTclamp);
}

PyMODINIT_FUNC
PyInit_nat(void)
{
  import_array();
  import_ufunc();
  return module_init("nat", cardinalsnative);
}
