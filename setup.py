from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='gtfstk',
    version='6.1.0',
    author='Alex Raichev',
    url='https://github.com/araichev/gtfstk',
    license=license,
    description='A Python 3.5 tool kit for processing General Transit Feed Specification (GTFS) data',
    long_description=readme,
    install_requires=[
        'Shapely>=1.5.1,<1.6',
        'pandas>=0.18.1,<0.20',
        'utm>=0.3.1',    
    ],
    packages=find_packages(exclude=('tests', 'docs'))
)

