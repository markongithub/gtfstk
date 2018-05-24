from setuptools import setup, find_packages


# Import ``__version__`` variable
exec(open('gtfstk/_version.py').read())

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE.txt') as f:
    license = f.read()

setup(
    name='gtfstk',
    version=__version__,
    author='Alex Raichev',
    author_email='alex@raichev.net',
    url='https://github.com/araichev/gtfstk',
    description='A Python 3.5+ tool kit that analyzes GTFS data',
    long_description=readme,
    license=license,
    install_requires=[
        'Shapely >= 1.5.1',
        'pandas >= 0.20.0',
        'utm >= 0.3.1',
        'pycountry == 17.1.8',
        'json2html >= 1.2.1',
    ],
    packages=find_packages(exclude=('tests', 'docs'))
)
