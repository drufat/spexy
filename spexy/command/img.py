# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import os
import subprocess
from importlib.machinery import SourceFileLoader

import matplotlib.pyplot as plt


def img(source, dest):
    m = SourceFileLoader('plot', source).load_module()
    if not os.path.exists(dest):
        os.makedirs(dest)

    fignums = plt.get_fignums()
    assert len(fignums) == m.__figcount__, f'figcount != {len(fignums)}'

    for i in fignums:
        plt.figure(i)
        plt.savefig('{}/{}.png'.format(dest, i))
        plt.savefig('{}/{}.pdf'.format(dest, i))

    plt.close('all')


if __name__ == '__main__':
    import argh

    parser = argh.ArghParser()
    parser.add_commands([
        img,
    ])
    parser.dispatch()
