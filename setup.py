from setuptools import setup, find_packages


# Import ``version`` variable
exec(open('gtfstk/_version.py').read())

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE.txt') as f:
    license = f.read()

setup(
    name='gtfstk',
    version=version,
    author='Alex Raichev',
    url='https://github.com/araichev/gtfstk',
    data_files = [('', ['LICENSE.txt'])],
    description='A Python 3.4+ tool kit that analyzes General Transit'
      'Feed Specification (GTFS) data',
    long_description=readme,
    license=license,
    install_requires=[
        'Shapely>=1.5.1',
        'pandas>=0.18.1',
        'utm>=0.3.1',
        'pycountry==17.1.8',
    ],
    packages=find_packages(exclude=('tests', 'docs'))
)

