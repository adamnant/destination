from setuptools import setup

__version__ = "0.1"

core_libs = ['numpy',
                 'networkx',
                 'matplotlib',
                 'begins',
                 'deap']              

setup(
    name='destination_problem',
    version=__version__,
    author='ag',
    packages=['ea'],
    include_package_data=True,
    description='some experiments on solving destination problem',
    entry_points={
        'console_scripts': [
            'run_ea = main.cli:main.start',
            ]
    },
    long_description=open('README.rst').read(),
    keywords='mining',
    classifiers=['Development Status :: Incomplete',
                 'Environment :: Console',
                 'Operating System :: POSIX :: Linux',
                 'Programming Language :: Python',
                 'Topic :: Optimisation'],
    install_requires=core_libs,
    platforms=['any']
    )
