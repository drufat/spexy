# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.

def insert(notice, *args):
    for filename in args:
        with open(filename, 'r') as f:
            lines = f.readlines()
            if not lines:
                continue
        preamble = []
        if lines[0][:2] == '#!':
            preamble += lines[0]
            lines = lines[1:]
        line = '{notice}\n'.format(notice=notice)
        if lines[0] == line:
            continue
        with open(filename, 'w') as f:
            f.write(''.join(preamble + [line] + lines))
