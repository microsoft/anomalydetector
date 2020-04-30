from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np

__version__ = "can't find version.py"
exec(compile(open('version.py').read(),
             'version.py', 'exec'))

extensions = [
    Extension("msanomalydetector._anomaly_kernel_cython", ["msanomalydetector/_anomaly_kernel_cython.pyx"],
              define_macros=[('CYTHON_TRACE', '1')])
]

cmdclass = {'build_ext': build_ext}

install_requires = [
    'Cython>=0.29.2',
    'numpy==1.18.1',
    'pandas==0.25.3'
]

setup(
    name="msanomalydetector",
    description='Microsoft Anomaly Detector Package Based On Saliency Detection',
    packages=find_packages(),
    include_dirs=[np.get_include()],
    cmdclass=cmdclass,
    ext_modules=cythonize(extensions),
    version=__version__,
    setup_requires=['Cython>=0.29.2', 'numpy==1.18.1'],
    install_requires=install_requires,
    requires=['Cython', 'numpy', 'pandas'],
    python_requires='>=3.6.0',
    package_data={'': ['*.txt']}
)
