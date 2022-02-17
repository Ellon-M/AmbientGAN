import Cython
from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("enc_messages.pyx"),
    include_dirs=[numpy.get_include()]
)
