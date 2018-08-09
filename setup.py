from setuptools import setup

setup(
    name='algorithm',
    version='0.1',
    install_requires=['numpy', 'scipy', 'numba', 'noformat'],
    dependency_links=['git+ssh://git@github.com:Palpatineli/noformat.git']
    packages=['algorithm'],
    extras_require={'test': ["pytest"]}
)
