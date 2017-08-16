// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#include <pybindcpp/module.h>
#include <pybindcpp/numpy.h>

#include <ops/ops.h>

using namespace pybindcpp;

#define PARALLEL _Pragma("omp parallel for if (M > MINPARALLEL)")

void
native(ExtModule& m)
{
  m.fun("freq", freq);

  auto py = [](const char* name) {
    return py_function<PyObject*(PyObject*)>("spexy.ops.nat_helper", name);
  };

  {
    auto wrap = py("wrap_op");
    auto _ = [&](const char* name, decltype(H) F) {
      m.add(name, wrap(fun2obj(fourier(F))));
    };

    _("H", H);
    _("Hinv", Hinv);

    _("S", S);
    _("Sinv", Sinv);

    _("Q", Q);
    _("Qinv", Qinv);

    _("G", G);
  };

  m.add("diff",                  //
        py("wrap_diff")(fun2obj( //
          [](int M, int Nin, const double* in, int Nout, double* out) {
            PARALLEL
            for (int i = 0; i < M; i++)
              diff<double>(          //
                Nin, in + i * Nin,   //
                Nout, out + i * Nout //
                );
          })));

  m.add("roll",                  //
        py("wrap_roll")(fun2obj( //
          [](int n, int M, int Nin, const double* in, int Nout, double* out) {
            PARALLEL
            for (int i = 0; i < M; i++)
              roll<double>(          //
                n,                   //
                Nin, in + i * Nin,   //
                Nout, out + i * Nout //
                );
          })));

  m.add("slice_",                  //
        py("wrap_slice_")(fun2obj( //
          [](int begin, int step, int M, int Nin, const double* in, int Nout,
             double* out) {
            PARALLEL
            for (int i = 0; i < M; i++)
              slice_<double>(        //
                begin, step,         //
                Nin, in + i * Nin,   //
                Nout, out + i * Nout //
                );
          })));

  m.add("weave",                  //
        py("wrap_weave")(fun2obj( //
          [](int M, int Nin0, const double* in0, int Nin1, const double* in1,
             int Nout, double* out) {
            PARALLEL
            for (int i = 0; i < M; i++)
              weave<double>(          //
                Nin0, in0 + i * Nin0, //
                Nin1, in1 + i * Nin1, //
                Nout,
                out + i * Nout //
                );
          })));

  m.add("concat",                  //
        py("wrap_concat")(fun2obj( //
          [](int M, int Nin0, const double* in0, int Nin1, const double* in1,
             int Nout, double* out) {
            PARALLEL
            for (int i = 0; i < M; i++)
              concat<double>(         //
                Nin0, in0 + i * Nin0, //
                Nin1, in1 + i * Nin1, //
                Nout,
                out + i * Nout //
                );
          })));
}

PyMODINIT_FUNC
PyInit_nat(void)
{
  import_array();
  return module_init("nat", native);
}
