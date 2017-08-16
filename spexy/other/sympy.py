# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import builtins

from sympy.printing.lambdarepr import NumPyPrinter
from sympy.core.compatibility import range
import sympy


class Printer(NumPyPrinter):
    def _print_Relational(self, expr):
        "Relational printer for Equality and Unequality"
        if expr.rel_op == '==':
            lhs = self._print(expr.lhs)
            rhs = self._print(expr.rhs)
            return 'equal({0}, {1})'.format(lhs, rhs)
        elif expr.rel_op == '!=':
            lhs = self._print(expr.lhs)
            rhs = self._print(expr.rhs)
            return 'not_equal({0}, {1})'.format(lhs, rhs)
        return super(Printer, self)._print_Relational(expr)

    def _print_Sum(self, expr):
        e, (k, k0, kN) = expr.args
        template = '(builtins.sum({expr} for {k} in range({k0}, {kN}+1)))'
        return template.format(expr=self._print(e),
                               k=self._print(k),
                               k0=self._print(k0),
                               kN=self._print(kN))


def lambdify(args, expr, modules=None, **kwargs):
    namespace = {'range': range,
                 'builtins': builtins}

    if modules is None:
        modules = namespace
    elif isinstance(modules, dict):
        modules.update(namespace)
    elif isinstance(modules, list):
        modules.insert(0, namespace)
    elif isinstance(modules, str):
        modules = [namespace, modules]
    else:
        raise NotImplementedError(type(modules))

    return sympy.lambdify(args, expr, modules=modules, printer=Printer, **kwargs)
