import setuptools
import os.path
from setuptools.command.install import install

executables = {'preprocessing': 'Preprocessing',
               'mcr_als': 'MCR-ALS',
               'clustering': 'Clustering'}

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="octavvs",
    version="0.1.14",
    author="Syahril Siregar, Carl Troein, Michiel Op De Beeck et al.",
    author_email="carl@thep.lu.se",
    description="Open Chemometrics Toolkit for Analysis and Visualization of Vibrational Spectroscopy data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ctroein/octavvs",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy', 'scipy', 'matplotlib', 'sklearn', 'pyshortcuts',
                      'opencv-python', 'pandas', 'pillow', 'pymatreader', 'dill',
                      'threadpoolctl', 'statsmodels'],
    package_data={ '': ['*.ui', '*.mat', '*.ico', '*.icns'] },
    entry_points={'console_scripts':
        ['octavvs = octavvs.launcher:main',
        'oct_preprocessing = octavvs.preprocessing:main',
        'oct_decomposition = octavvs.decomposition:main',
        'oct_make_icons = octavvs.make_icons:main'],
    },

)
