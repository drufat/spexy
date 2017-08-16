# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.

import json
import subprocess


def removetext(fromfile, tofile):
    with open(fromfile, 'r') as f:
        j = json.loads(f.read())

    for c in j['cells']:
        if 'outputs' not in c:
            continue
        for o in c['outputs']:
            if 'data' not in o:
                continue
            if 'text/plain' not in o['data']:
                continue
            if 'matplotlib' in o['data']['text/plain'][0]:
                del o['data']['text/plain']

    with open(tofile, 'w') as f:
        json.dump(j, f)


def nbconvert(fromfile, tofile, to='notebook', execute=False):
    command = ['jupyter', 'nbconvert', '--to', to, '--stdin', '--stdout']
    if execute:
        command += ['--execute', '--ExecutePreprocessor.timeout=300']

    with open(fromfile, 'r') as fin:
        sin = fin.read()

    sout = subprocess.check_output(command, input=sin.encode())

    with open(tofile, 'w') as fout:
        fout.write(sout.decode())


cmds = [removetext, nbconvert]

if __name__ == '__main__':
    import argh

    parser = argh.ArghParser()
    parser.add_commands(cmds)
    parser.dispatch()
