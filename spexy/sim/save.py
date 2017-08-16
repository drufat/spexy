from functools import wraps

import os
import h5py
import numpy as np

from spexy.helper import peek


def read(source, Const, Var):
    f = h5py.File(source, 'r')

    const = {_: f[_][...] for _ in Const._fields}
    yield Const(**const)

    frames = f['frames'][...]
    for i in range(frames):
        var = {_: f[_][i] for _ in Var._fields}
        yield Var(**var)

    f.close()


def write(sim, dest, frames, Const, Var):
    f = h5py.File(dest, 'w')

    f.create_dataset('frames', data=frames)

    const = next(sim)
    for _ in Const._fields:
        data = getattr(const, _)
        f.create_dataset(_, data=data)

    sim, step = peek(sim)
    for _ in Var._fields:
        data = np.array(getattr(step, _))
        f.create_dataset(_, (frames,) + data.shape, dtype=data.dtype)

    for i in range(frames):
        step = next(sim)
        for _ in Var._fields:
            f[_][i] = getattr(step, _)

    f.close()


def write_guarded(sim, dest, frames, Const, Var):
    temp = f'{dest}.part'
    write(sim, temp, frames, Const, Var)
    os.rename(temp, dest)


def save(Const, Var):
    def decorator(sim):

        class sim_functor:
            def __init__(self, *args, **kwargs):
                args_t = tuple(type(_) for _ in args)
                if args_t == (str,):
                    self.read(*args)
                else:
                    self.__sim__ = sim(*args, **kwargs)
                self.Const = Const
                self.Var = Var

            def read(self, source):
                self.__sim__ = read(source, Const, Var)

            def write(self, dest, frames):
                write_guarded(self.__sim__, dest, frames, Const, Var)

            def __iter__(self):
                return self

            def __next__(self):
                return next(self.__sim__)

        return sim_functor

    return decorator


def saverun(window_glpy):
    from glumpy import app
    from glumpy.app.movie import record

    @wraps(window_glpy)
    def _(*args, **kwargs):
        w = window_glpy(*args, **kwargs)

        def run():
            app.run()

        def save(dest):
            fps = 30
            with record(w, dest, fps=fps):
                app.run(framerate=fps)

        return run, save

    return _
