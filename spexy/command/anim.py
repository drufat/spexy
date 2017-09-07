# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
from glumpy import app
from importlib.machinery import SourceFileLoader

from glumpy.app.movie import record


def matplotanim(module, dest):
    module.anim.save(dest)


def glumpyanim(module, dest):
    duration = getattr(module, 'duration', 10.0)
    framerate = getattr(module, 'framerate', 60.0)
    with record(module.window, dest, fps=framerate):
        app.run(framerate=framerate, framecount=framerate * duration)


def anim(source, dest):
    module = SourceFileLoader('anim', source).load_module()
    if hasattr(module, 'window'):
        glumpyanim(module, dest)
    if hasattr(module, 'anim'):
        matplotanim(module, dest)
