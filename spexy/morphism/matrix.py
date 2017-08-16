import numpy as np
import sympy as sy


def matrix():
    m00, m01, m10, m11, v0, v1 = sy.symbols('m00, m01, m10, m11, v0, v1')
    M = sy.Matrix(
        [[m00, m01],
         [m10, m11]]
    )
    V = sy.Matrix(
        [[v0],
         [v1]]
    )
    W = M * V
    w0, w1 = W

    λ = sy.lambdify((m00, m01, m10, m11, v0, v1),
                    (w0, w1), 'numpy')

    def _(m):
        def f(v):
            return λ(*m, *v)

        return f

    return _


def matrixlambdify(st, M):
    """
    >>> s, t = sy.symbols('s, t')
    >>> M = sy.Matrix([[s, t], [t, s]])
    >>> m = matrixlambdify((s, t), M)
    >>> m(1, 2)([3, 4])
    (11, 10)
    >>> m(1, 0)([3, 4])
    (3, 4)
    """
    m = sy.lambdify(st, (M[0, 0], M[0, 1], M[1, 0], M[1, 1]), 'numpy')
    _ = matrix()
    return lambda *args: _(m(*args))


def matrixnum(st, M):
    """
    >>> s, t = sy.symbols('s, t')
    >>> M = sy.Matrix([[s, t], [t, s]])
    >>> m = matrixnum((s, t), M)
    >>> m(1, 2)
    (array(1), array(2), array(2), array(1))
    """
    return sy.lambdify(st, (M[0, 0], M[0, 1], M[1, 0], M[1, 1]), 'numpy')


def rotation():
    """
    >>> R = rotation()
    >>> R(90)([1, 0])
    (6.123233995736766e-17, 1.0)
    """
    theta, v0, v1 = sy.symbols('theta, v0, v1')
    cos, sin, π = sy.cos, sy.sin, sy.pi

    θ = theta * π / 180
    R = sy.Matrix([[cos(θ), -sin(θ)],
                   [sin(θ), cos(θ)]])
    V = sy.Matrix([[v0],
                   [v1]])
    W = R * V
    w0, w1 = W

    λ = sy.lambdify((theta, v0, v1),
                    (w0, w1), 'numpy')

    def num(θ):
        def f(v):
            v0, v1 = v
            return λ(θ, v0, v1)

        return f

    return num


def mat(f, **kwargs):
    def _(M):
        a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
        a, b, c, d = f(a, b, c, d, **kwargs)
        return sy.Matrix([[a, b], [c, d]])

    return _


def sqrtmat(
        a, b,
        c, d,
        module=sy
):
    """
    >>> sqrt = mat(sqrtmat)
    >>> M = sy.Matrix([[1, 2], [3, 4]])
    >>> M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    (1, 2, 3, 4)
    >>> assert sy.simplify(sqrt(M)**2) == M
    >>> M = sy.Matrix([[1, 0], [0, 1]])
    >>> sqrt(M)
    Matrix([
    [1, 0],
    [0, 1]])
    >>> assert sqrt(M)**2 == M
    >>> M = sy.Matrix([[4, 0], [0, 1]])
    >>> sqrt(M)
    Matrix([
    [2, 0],
    [0, 1]])
    >>> assert sqrt(M)**2 == M
    >>> M = sy.Matrix([[1, 4], [0, 1]])
    >>> sqrt(M)
    Matrix([
    [1, 2],
    [0, 1]])
    >>> assert sqrt(M)**2 == M
    >>> A, B, C, D = sy.symbols('A, B, C, D')
    >>> M = sy.Matrix([[A, B], [C, D]])
    >>> sy.simplify(sqrt(M)**2)
    Matrix([
    [A, B],
    [C, D]])
    >>> M = sy.Matrix([[A, B], [B, C]])
    >>> sy.simplify(sqrt(M))
    Matrix([
    [(A + sqrt(A*C - B**2))/sqrt(A + C + 2*sqrt(A*C - B**2)),                      B/sqrt(A + C + 2*sqrt(A*C - B**2))],
    [                     B/sqrt(A + C + 2*sqrt(A*C - B**2)), (C + sqrt(A*C - B**2))/sqrt(A + C + 2*sqrt(A*C - B**2))]])
    """
    τ = a + d
    δ = a * d - b * c
    s = module.sqrt(δ)
    t = module.sqrt(τ + 2 * s)
    a, b, c, d = (a + s), b, c, (d + s)
    a, b, c, d = a / t, b / t, c / t, d / t
    return a, b, c, d


def invmat(
        a, b,
        c, d
):
    """
    >>> inv = mat(invmat)
    >>> M = sy.Matrix([[1, 2], [3, 4]])
    >>> assert sy.simplify(inv(M)) == M.inv()
    """
    δ = a * d - b * c
    a, b, c, d = d, -b, -c, a
    a, b, c, d = a / δ, b / δ, c / δ, d / δ
    return a, b, c, d


def transmat(
        a, b,
        c, d
):
    """
    >>> trans = mat(transmat)
    >>> M = sy.Matrix([[1, 2], [3, 4]])
    >>> assert sy.simplify(trans(M)) == M.T
    """
    return (
        a, c,
        b, d,
    )


def detmat(
        a, b,
        c, d
):
    """
    >>> a, b, c, d = sy.symbols('a, b, c, d')
    >>> sy.Matrix([[a, b], [c, d]]).det()
    a*d - b*c
    >>> assert detmat(a, b, c, d) == sy.Matrix([[a, b], [c, d]]).det()
    """
    return a * d - b * c


def rot90(v0, v1):
    """
    >>> rot90(1, 0)
    (0, 1)
    >>> rot90(0, 1)
    (-1, 0)
    """
    return -v1, v0


def normalize(v):
    """
    >>> normalize([2, 0])
    (1.0, 0.0)
    """
    v0, v1 = v
    l = np.sqrt(v0 ** 2 + v1 ** 2)
    return v0 / l, v1 / l


def cartesian_product(a, b):
    """
    >>> assert (
    ... cartesian_product([1,2,3], ['a', 'b']) ==
    ... np.array([['1', 'a'],
    ...           ['1', 'b'],
    ...           ['2', 'a'],
    ...           ['2', 'b'],
    ...           ['3', 'a'],
    ...           ['3', 'b']],
    ...          dtype='<U21')
    ...     ).all()
    """
    return np.vstack([np.repeat(a, len(b)), np.tile(b, len(a))]).T


def smooth_step(a, b, x):
    """
    >>> smooth_step(0, 1, 0.0)
    0.0
    >>> smooth_step(0, 1, 0.5)
    0.5
    >>> smooth_step(0, 1, 1.0)
    1.0
    """
    if x < a: return 0.0
    if x > b: return 1.0
    x = (x - a) / (b - a)
    return x * x * x * (x * (x * 6 - 15) + 10)


def dot(ux, uy):
    return ux[0] * uy[0] + ux[1] * uy[1]
