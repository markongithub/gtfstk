from distutils.core import setup

setup(
    name='gtfs-toolkit',
    version='0.1.1',
    author='Alexander Raichev',
    author_email='alex@raichev.net',
    packages=['gtfs_toolkit', 'gtfs_toolkit.tests'],
    url='https://github.com/araichev/gtfs-toolkit',
    license='LICENSE',
    description='Python tools for processing `General Transit Feed Specification (GTFS) <https://en.wikipedia.org/wiki/GTFS>`_ data',
    long_description=open('README.rst').read(),
    install_requires=[
        'Shapely==1.3.2',
        'pandas==0.13.1',
        'utm==0.3.1',    
    ],
)

