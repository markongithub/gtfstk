from distutils.core import setup

setup(
    name='gtfs-toolkit',
    version='0.7',
    author='Alexander Raichev',
    author_email='alex@raichev.net',
    packages=['gtfs_toolkit', 'tests'],
    url='https://github.com/araichev/gtfs-toolkit',
    license='LICENSE',
    description='A set of Python 3.4 tools for processing General Transit Feed Specification (GTFS) data',
    long_description=open('README.rst').read(),
    install_requires=[
        'Shapely>=1.3.2',
        'pandas>=0.14.1',
        'utm>=0.3.1',    
    ],
)

