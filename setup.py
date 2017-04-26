## -------------------------- ##
## ------- Build with ------- ##
## python setup.py build_ext --inplace
## -------------------------- ##

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np


extensions = [
    Extension('plag._plag', ['cython/plag.pyx'],
        include_dirs = [np.get_include()],
        libraries = [],
        library_dirs = [],
        extra_compile_args = [],
        ),
]
setup(
    name = "plag",
    ext_modules = cythonize(extensions),
)
