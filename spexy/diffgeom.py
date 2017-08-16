from inspect import signature

import sympy as sy


def nargs(fn):
    """
    >>> nargs(lambda: None)
    0
    >>> nargs(lambda x: None)
    1
    >>> nargs(lambda x, y: None)
    2
    >>> nargs(lambda x, y, z: None)
    3
    """
    return len(signature(fn).parameters)


def Jac(M):
    """
    >>> x, y = sy.symbols('x, y')
    >>> M = lambda x, y: (x + 1, y + 1)
    >>> Jac(M)(x, y)
    [1, 0, 0, 1]
    >>> M = lambda x, y: (x*y, x - y)
    >>> Jac(M)(x, y)
    [y, x, 1, -1]
    >>> M = lambda x, y: (x*y, 0)
    >>> Jac(M)(x, y)
    [y, x, 0, 0]

    """
    x = [sy.Dummy() for _ in range(nargs(M))]
    y = M(*x)
    J = [sy.diff(yi, xi) for yi in y for xi in x]
    return sy.lambdify(x, J, 'sympy')


def Inv(M):
    """
    >>> M = lambda x, y: (x + 1, y + 1)
    >>> Minv = Inv(M)
    >>> x, y = sy.symbols('x, y')
    >>> Minv(x, y)
    [x - 1, y - 1]
    >>> Minv(*M(x, y))
    [x, y]
    """
    x = [sy.Dummy() for _ in range(nargs(M))]
    y = M(*x)
    z = [sy.Dummy() for _ in range(len(y))]
    sln = sy.solve(
        [yi - zi for (yi, zi) in zip(y, z)],
        x
    )
    return sy.lambdify(
        z,
        [sln[xi] for xi in x],
        'sympy'
    )


def PullBack(M):
    """
    >>> x, y = sy.symbols('x, y')
    >>> M = lambda x, y: (x + 1, y + 1)
    >>> s = lambda x, y: x + y
    >>> PullBack(M)(s)(x, y)
    x + y + 2
    >>> s = lambda x, y: (x * y, x - y)
    >>> PullBack(M)(s)(x, y)
    ((x + 1)*(y + 1), x - y)
    """
    x = [sy.Dummy() for _ in range(nargs(M))]

    def pull(s):
        return sy.lambdify(x, s(*M(*x)), 'sympy', )

    return pull


def PushForw(M):
    """
    >>> M = lambda x, y: (x + y, x - y)
    >>> s = lambda x, y: x * y
    >>> x, y = sy.symbols('x, y')
    >>> PushForw(M)(s)(x, y)
    (x/2 - y/2)*(x/2 + y/2)
    >>> PullBack(M)(PushForw(M)(s))(x, y)
    x*y
    """
    return PullBack(Inv(M))


def Tang(M):
    """
    >>> x, y, vx, vy = sy.symbols('x, y, vx, vy')
    >>> M = lambda x, y: (x + y, x - y)
    >>> Tang(M)(x, y, vx, vy)
    [x + y, x - y, vx + vy, vx - vy]
    >>> M = lambda x, y: (-y, x)
    >>> Tang(M)(x, y, vx, vy)
    [-y, x, -vy, vx]
    >>> M = lambda x, y: (x*y, x + y)
    >>> Tang(M)(x, y, vx, vy)
    [x*y, x + y, vx*y + vy*x, vx + vy]
    >>> ux, uy, uvx, uvy = sy.symbols('ux, uy, uvx, uvy')
    >>> Tang(Tang(M))(x, y, vx, vy, ux, uy, uvx, uvy)
    [x*y, x + y, vx*y + vy*x, vx + vy, ux*y + uy*x, ux + uy, uvx*y + uvy*x + ux*vy + uy*vx, uvx + uvy]
    """
    x = [sy.Dummy() for _ in range(nargs(M))]
    vx = [sy.Dummy() for _ in range(len(x))]

    y = list(M(*x))
    J = Jac(M)(*x)
    J = sy.Matrix(J).reshape(len(y), len(x))

    vy = list(J @ sy.Matrix(vx))
    return sy.lambdify(
        x + vx,
        y + vy,
        'sympy',
    )


def CoTang(M):
    """
    >>> x, y, fx, fy = sy.symbols('x, y, fx, fy')
    >>> M = lambda x, y: (x + y, x - y)
    >>> CoTang(M)(x, y, fx, fy)
    [x + y, x - y, fx/2 + fy/2, fx/2 - fy/2]
    >>> M = lambda x, y: (-y, x)
    >>> CoTang(M)(x, y, fx, fy)
    [-y, x, -fy, fx]
    """
    x = [sy.Dummy() for _ in range(nargs(M))]
    fx = [sy.Dummy() for _ in range(len(x))]

    y = list(M(*x))
    J = Jac(M)(*x)
    J = sy.Matrix(J).reshape(len(y), len(x))

    fy = list(J.T.inv() @ sy.Matrix(fx))
    return sy.lambdify(
        x + fx,
        y + fy,
        'sympy',
    )
