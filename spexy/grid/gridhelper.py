# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
class GridHelper:
    def verts(self):
        return self.enum(self.cells[0], self.N[0])

    def edges(self):
        return self.enum(self.cells[1], self.N[1])

    def faces(self):
        return self.enum(self.cells[2], self.N[2])

    def points(self):
        return self.refine().verts()

    def numbers(self):
        return self.N + self.dual.N

    def basis_fn(self):
        return self.B + self.dual.B

    def projection(self):
        return (
            tuple(self.proj_num(d) for d in range(self.dimension + 1)) +
            tuple(self.dual.proj_num(d) for d in range(self.dimension + 1))
        )

    def reconstruction(self):
        return (
            tuple(self.reconst(d) for d in range(self.dimension + 1)) +
            tuple(self.dual.reconst(d) for d in range(self.dimension + 1))
        )

    def boundary_cond(self):
        return (
            tuple(self.bndry_cond_num(d) for d in range(self.dimension + 1)) +
            tuple(self.dual.bndry_cond_num(d) for d in range(self.dimension + 1))
        )

    def derivative(self):
        return (
            tuple(self.deriv(d) for d in range(self.dimension + 1)) +
            tuple(self.dual.deriv(d) for d in range(self.dimension + 1))
        )

    def hodge_star(self):
        return (
            tuple(self.hodge(d) for d in range(self.dimension + 1)) +
            tuple(self.dual.hodge(d) for d in range(self.dimension + 1))
        )

    def laplacian(self):
        D0, D1, D2, D0d, D1d, D2d = self.derivative()
        H0, H1, H2, H0d, H1d, H2d = self.hodge_star()

        L0 = lambda f: H2d(D1d(H1(D0(f))))
        L0d = lambda f: H2(D1(H1d(D0d(f))))
        L1 = lambda f: (H1d(D0d(H2(D1(f)))) +
                        D0(H2d(D1d(H1(f)))))
        L1d = lambda f: (H1(D0(H2d(D1d(f)))) +
                         D0d(H2(D1(H1d(f)))))

        return L0, L1, L0d, L1d
