from setuptools import setup

setup(
    name='algorithm',
    version='0.1',
    install_requires=['numpy', 'scipy', 'numba', 'noformat'],
    packages=['algorithm'],
    extras_require={'test': ["pytest"]}
)
