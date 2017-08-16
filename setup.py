# Copyright (C) 2010-2016 Dzhelil S. Rufat. All Rights Reserved.
import glob, os

import numpy
from setuptools import setup, Extension

include_dirs = [
    'src',
    '../pybindcpp/include',
    '../pybind11/include',
    numpy.get_include(),
]

headers = glob.glob(
    'include/python/*.h'
)

depends = [
              'setup.py',
          ] + headers

extra_compile_args = [
    '-std=c++14',
]

libraries = []

if ('CC' in os.environ) and ('gcc' in os.environ['CC'] or 'g++' in os.environ['CC']):
    extra_compile_args += ['-fopenmp']
    libraries += ['gomp']

ext_modules = [

    Extension(
        'dec.ops.nat',
        sources=[
            'dec/ops/nat.cpp',
            'src/dec/ops/ops.cpp',
        ],
        depends=depends + [
            'src/dec/ops/ops.h',
        ],
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        language="c++",
        libraries=libraries + ['fftw3'],
    ),

    Extension(
        'dec.bases.circular.nat',
        sources=[
            'dec/bases/circular/nat.cpp',
            'src/dec/bases/circular.cpp',
        ],
        depends=[
                    'src/dec/bases/circular.h',
                ] + depends,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        language="c++",
        libraries=libraries,
    ),

    Extension(
        'dec.bases.cardinals.nat',
        sources=[
            'dec/bases/cardinals/nat.cpp',
            'src/dec/bases/cardinals.cpp',
        ],
        depends=[
                    'src/dec/bases/cardinals.h',
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
    version='0.0',
    description='Spectral Exterior Calculus',
    author='Dzhelil Rufat',
    author_email='drufat@fastmail.com',
    license='GNU GPLv3',
    url='http://github.com/drufat/spexy.git',
    requires=[
        'numpy',
        'sympy',
        'pybindcpp',
    ],
)
