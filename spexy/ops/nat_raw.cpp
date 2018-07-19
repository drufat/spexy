// Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
#include <ops/ops.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#define PARALLEL _Pragma("omp parallel for if (M > MINPARALLEL)")

using arr_t = py::array_t<double, py::array::c_style | py::array::forcecast>;
using op_t = decltype(H);

auto wrap_op(op_t F) {
  auto FF = fourier(F);
  auto func = [=](int M, int N, arr_t f, arr_t fout) {
    FF(M, N, f.data(), fout.mutable_data());
  };
  return func;
}

auto wrap_diff(int M, int Nin, arr_t f, int Nout, arr_t fout) {
  auto in = f.data();
  auto out = fout.mutable_data();
  PARALLEL
  for (int i = 0; i < M; i++)
    diff<double>(             //
        Nin, in + i * Nin,    //
        Nout, out + i * Nout  //
    );
}

auto wrap_roll(int n, int M, int Nin, arr_t f, int Nout, arr_t fout) {
  auto in = f.data();
  auto out = fout.mutable_data();
  PARALLEL
  for (int i = 0; i < M; i++)
    roll<double>(             //
        n,                    //
        Nin, in + i * Nin,    //
        Nout, out + i * Nout  //
    );
}

auto wrap_slice_(int begin, int step, int M, int Nin, arr_t f, int Nout,
                 arr_t fout) {
  auto in = f.data();
  auto out = fout.mutable_data();
  PARALLEL
  for (int i = 0; i < M; i++)
    slice_<double>(           //
        begin, step,          //
        Nin, in + i * Nin,    //
        Nout, out + i * Nout  //
    );
}

auto wrap_weave(int M, int Nin0, arr_t fin0, int Nin1, arr_t fin1, int Nout,
                arr_t fout) {
  auto in0 = fin0.data();
  auto in1 = fin1.data();
  auto out = fout.mutable_data();
  PARALLEL
  for (int i = 0; i < M; i++)
    weave<double>(             //
        Nin0, in0 + i * Nin0,  //
        Nin1, in1 + i * Nin1,  //
        Nout,
        out + i * Nout  //
    );
}

auto wrap_concat(int M, int Nin0, arr_t fin0, int Nin1, arr_t fin1, int Nout,
                 arr_t fout) {
  auto in0 = fin0.data();
  auto in1 = fin1.data();
  auto out = fout.mutable_data();
  PARALLEL
  for (int i = 0; i < M; i++)
    concat<double>(            //
        Nin0, in0 + i * Nin0,  //
        Nin1, in1 + i * Nin1,  //
        Nout,
        out + i * Nout  //
    );
}

PYBIND11_MODULE(nat_raw, m) {
  m.def("freq", freq);

  m.def("H", wrap_op(H));
  m.def("Hinv", wrap_op(Hinv));

  m.def("S", wrap_op(S));
  m.def("Sinv", wrap_op(Sinv));

  m.def("Q", wrap_op(Q));
  m.def("Qinv", wrap_op(Qinv));

  m.def("G", wrap_op(G));

  m.def("diff", wrap_diff);
  m.def("roll", wrap_roll);
  m.def("slice_", wrap_slice_);
  m.def("weave", wrap_weave);
  m.def("concat", wrap_concat);
}
