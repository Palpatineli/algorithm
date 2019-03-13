from setuptools import setup
from setuptools.extension import Extension
import numpy as np
import subprocess
from Cython.Build import cythonize

version = "0.1.2"

try:  # Try to create an rst long_description from README.md
    ARGS = "pandoc", "--to", "rst", "README.md"
    LONG_DESC = subprocess.check_output(ARGS)
    LONG_DESC = LONG_DESC.decode()
except Exception as error:  # pylint: disable=broad-except
    print("README.md conversion to reStructuredText failed. Error:\n",
          error, "Setting long_description to None.")
    LONG_DESC = None

extensions = [
    Extension("algorithm.array._main", ["algorithm.array._main.pyx"], include_dirs=[np.get_include()]),
    Extension("algorithm.time_series._utils", ["algorithm.time_series._utils.pyx"], include_dirs=[np.get_include()]),
    Extension("algorithm.utils.first_x", ["algorithm.utils.first_x.pyx"], include_dirs=[np.get_include()])
]

setup(
    name='algorithm',
    version=version,
    install_requires=['numpy', 'scipy', 'numba', 'noformat', 'Cython'],
    packages=['algorithm'],
    author="Keji Li",
    author_email="mail@keji.li",
    url='https://github.com/Palpatineli/algorithm',
    download_url='https://github.com/Palpatineli/algorithm/archive/{}.tar.gz'.format(version),
    ext_modules=cythonize(extensions),
    tests_require=["pytest"],
    extras_require=["pandas"],
    long_description=LONG_DESC,
)
