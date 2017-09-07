# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import numpy as np

rand = lambda n, m: np.random.random((n, m))
bmat = lambda blocks: np.asarray(np.bmat(blocks))


def schur(Ainv, B, C, D):
    """
    Suppose we are given a matrix M in block diagonal form
    M = [[A, B],
         [C, D]]
    and we are interested in computing its inverse. Normally that would be 
    a very slow operation, but we can make it really fast using 
    Schur's complement if we have a fast implementation of Ainv. 
    
    Minv = schur(Ainv, B, C, D)
    """
    n, m = B.shape
    assert C.shape == (m, n) and D.shape == (m, m)

    # Apply Ainv to columns of B
    CAinvB = np.array([C.dot(Ainv(b)) for b in B.T]).T

    # inverse of Schur's Complement
    Sinv = np.linalg.inv(CAinvB - D)

    def Minv(f, g):
        λ = Sinv.dot(C.dot(Ainv(f)) - g)
        x = Ainv(f - B.dot(λ))
        return x, λ

    return Minv


def schurλ(Ainv, B, C, D, n, m):
    """
    Variation of schur where Ainv, B, C, D, are 
    functions rather than methods
    and the dimensions of the problem are (n+m, n+m)
    where presumably n>>m.
    """

    # iterate over the columns of B
    CAinvB = np.array([C(Ainv(B(i))) for i in np.eye(m)]).T
    D = np.array([D(i) for i in np.eye(m)]).T

    # inverse of Schur's Complement
    Sinv = np.linalg.inv(CAinvB - D)

    def Minv(f, g):
        λ = Sinv.dot(C(Ainv(f)) - g)
        x = Ainv(f - B(λ))
        return x, λ

    return Minv


def test_schur():
    np.random.seed(1)

    for (n, m) in ((0, 1), (0, 2), (1, 1), (5, 2), (10, 3)):
        f, g = rand(n, 1), rand(m, 1)

        A = rand(n, n)
        B = rand(n, m)
        C = rand(m, n)
        D = rand(m, m)

        Ainv = np.linalg.inv(A)
        M = bmat([[A, B], [C, D]])
        Minv = np.linalg.inv(M)

        λ = lambda A: lambda x: A.dot(x)

        Mschur = schur(λ(Ainv), B, C, D)
        assert np.allclose(Minv.dot(np.vstack([f, g])), np.vstack(Mschur(f, g)))

        Mschur = schurλ(λ(Ainv), λ(B), λ(C), λ(D), n, m)
        assert np.allclose(Minv.dot(np.vstack([f, g])), np.vstack(Mschur(f, g)))
