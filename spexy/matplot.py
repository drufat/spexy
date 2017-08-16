# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import base64
import tempfile

import matplotlib.pyplot as plt
import numpy as np


class HTMLAnimation(object):
    VIDEO_TAG = """
        <video loop autoplay>
            <source src="{src}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    """

    def __init__(self, filename, inline=True):
        if inline:
            with open(filename, 'rb') as video:
                encoded = base64.b64encode(video.read()).decode()
                src = "data:video/mp4;base64,{data}".format(data=encoded)
                self.html = HTMLAnimation.VIDEO_TAG.format(src=src)
        else:
            self.html = HTMLAnimation.VIDEO_TAG.format(src=filename)

    def _repr_html_(self):
        return self.html


class MatplotAnimation(object):
    def __init__(self, animation):
        self.animation = animation

    def save(self, filename, **kwargs):
        assert (filename[-4:] == '.mp4')
        extra_args = [
            '-vcodec', 'libx264',
            '-pix_fmt', 'yuv420p',  # needed by Firefox
            '-preset', 'slow',
            '-movflags', '+faststart'
        ]
        self.animation.save(filename, extra_args=extra_args, **kwargs)

    def _repr_html_(self):
        with tempfile.NamedTemporaryFile(suffix='.mp4') as f:
            self.save(f.name)
            return HTMLAnimation(f.name, inline=True)._repr_html_()


def grey_save(name, tex):
    tex = tex[:, ::-1]
    tex = tex.T

    M, N = tex.shape
    texture = np.empty((M, N, 4), np.float32)
    texture[:, :, 0] = tex
    texture[:, :, 1] = tex
    texture[:, :, 2] = tex
    texture[:, :, 3] = 1

    plt.imsave(name, texture)
