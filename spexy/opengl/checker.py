# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import numpy as np


def checkerboard(lenx, leny, Nx, Ny):
    nx, ny = len(lenx), len(leny)
    image = np.zeros((ny, nx))
    image[0::2, 0::2] = 1
    image[1::2, 1::2] = 1

    lenx = Nx * lenx / sum(lenx)
    leny = Ny * leny / sum(leny)

    cumx = np.cumsum(lenx)
    cumy = np.cumsum(leny)

    Nx = np.rint(cumx[-1]).astype(np.int)
    Ny = np.rint(cumy[-1]).astype(np.int)

    cumx = np.rint(cumx).astype(np.int)
    cumy = np.rint(cumy).astype(np.int)

    rslt = np.empty((Nx, Ny))
    for x, (x0, x1) in enumerate(zip(np.r_[0, cumx[:-1]], cumx)):
        for y, (y0, y1) in enumerate(zip(np.r_[0, cumy[:-1]], cumy)):
            rslt[y0:y1, x0:x1] = image[y, x]

    return rslt


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    np.random.seed(1)

    nx, ny = 9, 13
    # lenx = np.ones(nx); lenx[0] *= 2
    # leny = np.ones(ny)
    lenx = np.random.rand(nx) + .3
    leny = np.random.rand(ny) + .3
    Nx, Ny = 1000, 1000
    rslt = checkerboard(lenx, leny, Nx, Ny)
    plt.matshow(rslt)

    plt.show()
