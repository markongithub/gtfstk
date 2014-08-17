from distutils.core import setup

setup(
    name='gtfs-toolkit',
<<<<<<< HEAD
    version='0.1.12',
=======
    version='0.2.1',
>>>>>>> 57d7fbb8d668f0644d7e5e77120b07f57375968f
    author='Alexander Raichev',
    author_email='alex@raichev.net',
    packages=['gtfs_toolkit', 'gtfs_toolkit.tests'],
    url='https://github.com/araichev/gtfs-toolkit',
    license='LICENSE',
    description='A set of Python 3.4 tools for processing General Transit Feed Specification (GTFS) data',
    long_description=open('README.rst').read(),
    install_requires=[
        'Shapely==1.3.2',
        'pandas==0.14.1',
        'utm==0.3.1',    
    ],
)

