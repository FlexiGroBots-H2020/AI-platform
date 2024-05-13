try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
	

from Cython.Build import cythonize


ext_modules = [
    Extension(
        "cython_function",
        ["cython_function.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp']
    )
]


setup(ext_modules=cythonize(ext_modules))

#setup(
#    name='cython_function-parallel-world',
#    ext_modules=cythonize(ext_modules),
#)


#cython cython_function.pyx
#python setup_cython_function.py build_ext --inplace
