from setuptools import setup, find_packages
import numpy

__version__ = "can't find version.py"
exec(compile(open('version.py').read(),
             'version.py', 'exec'))

install_requires = [
    'numpy==1.16.0',
    'pandas==0.23.4'
]

setup(
    name="msanomalydetector",
    description='Microsoft Anomaly Detector Package Based On Saliency Detection',
    packages=find_packages(),
    version=__version__,
    install_requires=install_requires,
    include_dirs=[numpy.get_include()],
    requires=['numpy', 'pandas'],
    python_requires='>=3.6.0',
    package_data={'': ['*.txt']}
)
