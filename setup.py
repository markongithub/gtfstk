from distutils.core import setup

setup(
    name='gtfstk',
    version='0.12.1',
    author='Alexander Raichev',
    author_email='alex@raichev.net',
    packages=['gtfstk', 'tests'],
    url='https://github.com/araichev/gtfstk',
    license='LICENSE',
    description='A Python 3.4 toolkit for processing General Transit Feed Specification (GTFS) data',
    long_description=open('README.rst').read(),
    install_requires=[
        'Shapely>=1.5.1',
        'pandas>=0.15.2',
        'utm>=0.3.1',    
    ],
)

