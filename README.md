# OCTAVVS: Open Chemometrics Toolbox for Analysis and Visualization of Vibrational Spectroscopy data

OCTAVVS is a set of graphical tools for high-throughput preprocessing and
analysis of vibrational spectroscopy data. Currently, the preprocessing is
primarily geared towards images from infrared absorption spectroscopy with
focal plane array detectors.

There are three separate tools in the current version:

**preprocessing** deals with atmospheric correction, resonant Mie scattering
correction, baseline correction and normalization.

**mcr_als** decomposes observed spectra into nonnegative concentrations and
spectra using the MCR-ALS algorithm.

**clustering** performs K-means clustering on the concentrations inferred by
MCR-ALS.

## Installation on Windows/Mac

To install OCTAVVS on Windows or Mac, start by downloading Python 3.7 (or newer) from
[Python.org](https://www.python.org/downloads/). (It should also be possible
to use Conda as described for Linux.)

When you have installed Python, get a command prompt:

* On Windows: Windows key + "r", type "cmd"
* On Mac: ???

Then ask ``pip`` to download and install octavvs and its requirements:
``pip install -U --extra-index-url https://test.pypi.org/simple octavvs-ctroein[noconda]``


## Installation on Linux (or with Conda on other systems)

Alternatively, and possibly required on Linux because of an issue with
PyQt5: Install the Python 3.7 (or newer) version of
[Miniconda](https://docs.conda.io/en/latest/miniconda.html) or
[Anaconda](https://www.anaconda.com/distribution/). In the last installation
step, conda will want to add its path to $PATH in your .bashrc; doing so
could potentially break things (on OpenSUSE it's been known to conflict with
KDE), so you may want to manually control the $PATH instead.

Thus to install the PyQt5 package: ``PATH=~/miniconda3/bin:$PATH conda install pyqt``

Then install Octavvs without pulling in the incompatible pyqt5 package with pip:
``pip install -U --extra-index-url https://test.pypi.org/simple octavvs-ctroein``

If you accidentally install the pip pyqt5 package, the easiest way to get
rid of it is to ``pip uninstall pyqt5`` and then
``conda install --force-reinstall pyqt``

## Finding and using octavvs

The easiest way to access the Octavvs scripts is through desktop shortcuts
which will be created by running the ``oct_make_icons`` script in the
console.

The location of the octavvs scripts will depend on your operating system and
where you installed Python. The files will be located in the directory
``lib/python3.7/site-packages/octavvs`` but the executable scripts
``oct_preprocessing``, ``oct_mcr_als_`` and ``oct_clustering`` will be
located in ``bin`` and should be possible to run straight from the console.
