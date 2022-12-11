from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(ext_modules=cythonize(Extension(
    'vein',
    sources=['vein_mask.pyx'],
    language='c++',
    include_dirs=[numpy.get_include()],
    library_dirs=[],
    libraries=[],
    extra_compile_args=['-o3'],
    extra_link_args=[]
)))
