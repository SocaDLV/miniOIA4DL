# --- INICIO BLOQUE GENERADO CON IA ---
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import sys

# Argumentos de compilación para activar OpenMP dependiendo del sistema operativo
if sys.platform.startswith('win'):
    # En Windows con MSVC
    openmp_args = ['/openmp']
elif sys.platform.startswith('darwin'):
    # En Mac (requiere libomp, a veces da problemas así que lo dejamos seguro)
    openmp_args = ['-Xpreprocessor', '-fopenmp']
else:
    # En Linux con GCC
    openmp_args = ['-fopenmp']

extensions = [
    Extension(
        "*",
        ["*.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=openmp_args,
        extra_link_args=openmp_args,
    )
]

setup(
    ext_modules=cythonize(extensions, language_level=3)
)
# --- FIN BLOQUE GENERADO CON IA ---