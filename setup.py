from distutils.core import setup

setup(
    name='gtfstk',
    version='2.0.1',
    author='Alexander Raichev',
    author_email='alex@raichev.net',
    packages=['gtfstk', 'tests'],
    url='https://github.com/araichev/gtfstk',
    license='LICENSE',
    description='A Python 3.4 tool kit for processing General Transit Feed Specification (GTFS) data',
    long_description=open('README.rst').read(),
    install_requires=[
        'Shapely>=1.5.1',
        'pandas>=0.17',
        'utm>=0.3.1',    
    ],
)

