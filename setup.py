from distutils.core import setup

setup(
    name='gtfs-tk',
    version='0.11.4',
    author='Alexander Raichev',
    author_email='alex@raichev.net',
    packages=['gtfs_tk', 'tests'],
    url='https://github.com/araichev/gtfs-tk',
    license='LICENSE',
    description='A Python 3.4 toolkit for processing General Transit Feed Specification (GTFS) data',
    long_description=open('README.rst').read(),
    install_requires=[
        'Shapely>=1.5.1',
        'pandas>=0.15.2',
        'utm>=0.3.1',    
    ],
)

