import setuptools

with open('requirements-noversion.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name='plenoxels',
    version='0.1.0',
    install_requires=requirements
)

