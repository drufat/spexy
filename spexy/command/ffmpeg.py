import subprocess as sp

fps = 30.0


def snapshot(source, dest, frame):
    time = float(frame) / fps
    sp.check_call([
        'ffmpeg', '-y',
        '-ss', str(time),
        '-i', source,
        '-vframes', '1', dest
    ])

if __name__ == '__main__':
    import argh

    parser = argh.ArghParser()
    parser.add_commands([
        snapshot,
    ])
    parser.dispatch()
