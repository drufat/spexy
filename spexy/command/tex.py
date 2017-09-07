# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import importlib.machinery
import os
import re
import subprocess


def removepreamble(i, o):
    with open(i, 'r') as f:
        tex = f.read()

    _, tex = tex.split(r'\begin{document}')
    tex, _ = tex.split(r'\end{document}')
    tex = tex.strip()

    with open(o, 'w') as f:
        f.write(tex)


def removebib(i, o):
    with open(i, 'r') as f:
        tex = f.read()

    tex = re.sub(r'\\bibliography[a-z]*\{.*\}', '', tex)
    tex = tex.strip()

    with open(o, 'w') as f:
        f.write(tex)


def tex(source, dest):
    srcmod = importlib.machinery.SourceFileLoader('tex', source).load_module()
    if not os.path.exists(dest):
        os.makedirs(dest)

    def gentex(name, latex):
        # print(latex)
        with open('{}/{}.tex'.format(dest, name), 'w') as f:
            f.write(latex)

    srcmod.gentex(gentex)
    subprocess.run(['touch', dest])
