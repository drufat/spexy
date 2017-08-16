# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import importlib


class BasesImp(object):
    def __init__(self, N, imp='nat'):
        self.N = N
        self.imp = importlib.import_module(
            '{module}.{imp}'.format(
                module=self.module(),
                imp=imp
            )
        )

    def cells(self):
        p = self.points
        (i0, i1), (id0, id1) = self.cells_index()
        c0 = lambda n: tuple(p(_) for _ in i0(n))
        c1 = lambda n: tuple(p(_) for _ in i1(n))
        cd0 = lambda n: tuple(p(_) for _ in id0(n))
        cd1 = lambda n: tuple(p(_) for _ in id1(n))
        return (c0, c1), (cd0, cd1)

    def delta(self):
        (c0, c1), (cd0, cd1) = self.cells()
        d0 = lambda n: 1
        d1 = lambda n: c1(n)[1] - c1(n)[0]
        d0d = lambda n: 1
        d1d = lambda n: cd1(n)[1] - cd1(n)[0]
        return (d0, d1), (d0d, d1d)

    def module(self):
        raise NotImplemented

    def numbers(self):
        raise NotImplemented

    def points(self):
        raise NotImplemented

    def bases(self):
        raise NotImplemented

    def boundary(self):
        raise NotImplemented

    def cells_index(self):
        raise NotImplemented
