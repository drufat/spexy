from importlib.machinery import SourceFileLoader
from itertools import chain
from pathlib import Path

import matplotlib

from spexy.command.ffmpeg import snapshot

matplotlib.use('Agg')


def mk(fun, *args, file_dep=None, targets=None, name=None):
    if file_dep is None:
        file_dep = args[:-1]
    if targets is None:
        targets = [args[-1]]
    if name is None:
        name = '_'.join([str(t) for t in targets])
    return {
        'name': name,
        'actions': [lambda: fun(*args)],
        'file_dep': file_dep,
        'targets': targets,
        'clean': True
    }


def task_adv():
    from spexy.adv.command import img, math

    dir = 'spexy/adv/samples'
    for src in Path(dir).glob('*.py'):
        if src.name in ['__init__.py']:
            continue
        yield mk(img, src, f'{src}.png')
        yield mk(math, src, f'{src}.txt')


def task_vectorfields():
    from spexy.command.vf import arr, lic

    dir = 'spexy/vectorfields'
    for src in Path(dir).glob('*.py'):
        if src.name in ['__init__.py', 'transform.py']:
            continue
        yield mk(arr, src, src.with_suffix('.arr.png'))
        yield mk(lic, src, src.with_suffix('.lic.png'))


def task_morphism():
    from spexy.command.morphism import plots, movies, gen

    dir = 'spexy/morphism/maps'
    out = 'spexy/maps'
    target_plots = plots()
    target_movies = movies()

    for source in Path(dir).glob('*.py'):
        if source.name in ['__init__.py', 'common.py']:
            continue
        for trgt in chain(
                target_plots,
                target_movies
        ):
            target = f'{out}/{source.stem}/{trgt}'
            yield mk(gen, source, target)


def task_converge():
    from spexy.command.converge import sim_save, plt_save

    dir = 'spexy/sim/converge'

    for sim in Path(dir).glob('**/sim.py'):
        hdf5 = sim.with_suffix('.hdf5')

        yield mk(sim_save, sim, hdf5)

        plt = sim.parent / 'plt.py'
        if not plt.exists():
            continue

        Figs = SourceFileLoader('_', str(plt)).load_module().Figs
        for fig in Figs._fields:
            png = plt.parent / f'{fig}.png'
            yield mk(
                plt_save, str(sim), str(hdf5), str(plt), str(fig), str(png),
                file_dep=[sim, hdf5, plt],
                targets=[png],
            )


def task_vorticity():
    from spexy.sim.vorticity.vorticity import (
        vort_save, vort_ww, vort_vv
    )

    dir = 'spexy/sim/vorticity/plots'

    for src in Path(dir).glob('**/*.py'):
        if src.name in ['__init__.py']:
            continue

        for (ext, primal) in [('', 'primal'), ('.dual', 'dual')]:
            hdf5 = src.with_suffix(f'{ext}.hdf5')
            yield mk(vort_save, str(src), str(hdf5), primal,
                     file_dep=[src], targets=[hdf5])
            png = src.with_suffix(f'{ext}.ww.png')
            yield mk(vort_ww, str(hdf5), str(png))
            png = src.with_suffix(f'{ext}.vv.png')
            yield mk(vort_vv, str(hdf5), str(png))


def task_embed():
    from spexy.embed.opengl import (
        cmd, flat_vector_sqrt, surface_vector_id, surface_vector_sqrt, surface_glumpy,
    )
    dir = 'spexy/embed/surface'
    for src in Path(dir).glob('**/*.py'):
        if src.name in ['__init__.py']:
            continue

        mp4 = src.with_suffix('.flat.mp4')
        yield mk(cmd(flat_vector_sqrt), str(src.stem), str(mp4),
                 file_dep=[src], targets=[mp4])
        mp4 = src.with_suffix('.id.mp4')
        yield mk(cmd(surface_vector_id), str(src.stem), str(mp4),
                 file_dep=[src], targets=[mp4])
        mp4 = src.with_suffix('.sqrt.mp4')
        yield mk(cmd(surface_vector_sqrt), str(src.stem), str(mp4),
                 file_dep=[src], targets=[mp4])
        mp4 = src.with_suffix('.surf.mp4')
        yield mk(surface_glumpy, str(src.stem), str(mp4), str(50),
                 file_dep=[src], targets=[mp4])
        mp4 = src.with_suffix('.above.mp4')
        yield mk(surface_glumpy, str(src.stem), str(mp4), str(00),
                 file_dep=[src], targets=[mp4])

        frame = '00'
        mp4 = src.with_suffix('.surf.mp4')
        png = src.with_suffix('.surf.png')
        yield mk(snapshot, str(mp4), str(png), str(frame),
                 file_dep=[mp4], targets=[png])

        mp4 = src.with_suffix('.above.mp4')
        png = src.with_suffix('.above.png')
        yield mk(snapshot, str(mp4), str(png), str(frame),
                 file_dep=[mp4], targets=[png])

        if src.stem in [
            'id', 'gauss',
        ]:
            mp4 = src.with_suffix('.flat.mp4')
            png = src.with_suffix('.flat.png')
            yield mk(snapshot, str(mp4), str(png), str(frame),
                     file_dep=[mp4], targets=[png])
            mp4 = src.with_suffix('.id.mp4')
            png = src.with_suffix('.id.png')
            yield mk(snapshot, str(mp4), str(png), str(frame),
                     file_dep=[mp4], targets=[png])
            mp4 = src.with_suffix('.sqrt.mp4')
            png = src.with_suffix('.sqrt.png')
            yield mk(snapshot, str(mp4), str(png), str(frame),
                     file_dep=[mp4], targets=[png])
