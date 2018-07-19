# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import glob, os

import numpy
import pybind11
from setuptools import setup, Extension

include_dirs = [
    'src',
    numpy.get_include(),
    pybind11.get_include(True),
    pybind11.get_include(False),
]

headers = glob.glob(
    'include/python/*.h'
)

depends = [
              'setup.py',
          ] + headers

extra_compile_args = [
    '-std=c++11',
]

libraries = []

if ('CC' in os.environ) and ('gcc' in os.environ['CC'] or 'g++' in os.environ['CC']):
    extra_compile_args += ['-fopenmp']
    libraries += ['gomp']

ext_modules = [

    Extension(
        'spexy.ops.nat_raw',
        sources=[
            'spexy/ops/nat_raw.cpp',
            'src/ops/ops.cpp',
        ],
        depends=depends + [
            'src/ops/ops.h',
        ],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        language="c++",
        libraries=libraries + ['fftw3'],
    ),

    Extension(
        'spexy.bases.circular.nat',
        sources=[
            'spexy/bases/circular/nat.cpp',
            'src/bases/circular.cpp',
        ],
        depends=[
                    'src/bases/circular.h',
                ] + depends,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        language="c++",
        libraries=libraries,
    ),

    Extension(
        'spexy.bases.cardinals.nat',
        sources=[
            'spexy/bases/cardinals/nat.cpp',
            'src/bases/cardinals.cpp',
        ],
        depends=[
                    'src/bases/cardinals.h',
                ] + depends,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        language="c++",
        libraries=libraries,
    ),

]

setup(
    name='spexy',
    packages=['spexy'],
    package_dir={'spexy': 'spexy'},
    package_data={'spexy': ['spexy/data/memoize/*.json']},
    ext_modules=ext_modules,
    version='0.2',
    description='Spectral Exterior Calculus',
    author='Dzhelil Rufat',
    author_email='drufat@fastmail.com',
    license='GNU GPLv3',
    url='http://dzhelil.info/spexy',
    requires=[
        'numpy',
        'sympy',
        'scipy',
        'pybind11',
        'matplotlib',
    ],
)
