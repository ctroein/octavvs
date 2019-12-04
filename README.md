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

## Installation on Windows and Mac

To install OCTAVVS on Windows or Mac, start by downloading Python version
3.8 (or any version from 3.6 and onwards) from
[Python.org](https://www.python.org/downloads/). (It is also possible to use
Conda as described for Linux, below.) When installing Python, check the box
for adding Python to your path.

When you have installed Python, get a command prompt:

* On Windows: Windows key + "r", type "cmd"
* On Mac: open Terminal

Then use ``pip`` to download and install pyqt5 and octavvs and its requirements:  
``pip install pyqt5``  
``pip install octavvs``

## Upgrading to the latest version

Information about the most recent version of OCTAVVS can be found on
[its PyPI project page](https://pypi.org/project/octavvs/).
To upgrade to the latest version:  
``pip install -U octavvs``

## Installation on Linux (or with Conda on other systems)

Alternatively (and apparently required on at least some Linux distributions because
of issues with PyQt5): Install the Python 3.7 (or newer) version
of [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or
[Anaconda](https://www.anaconda.com/distribution/). In the last installation
step, conda will want to add its path to $PATH in your .bashrc; doing so
could potentially break things (on OpenSUSE it's been known to conflict with
KDE), so you may want to manually control the $PATH instead.

Then install PyQt5 using conda: ``conda install pyqt``  
Install/upgrade Octavvs using pip as above: ``pip install -U octavvs``

If you accidentally install the pip pyqt5 package, the easiest way to get
rid of it is to ``pip uninstall pyqt5`` and then
``conda install --force-reinstall pyqt``

## Finding and using octavvs

The easiest way to access the Octavvs scripts is through desktop shortcuts
which may be created by running the ``oct_make_icons`` script in the
console.

The location of the octavvs scripts will depend on your operating system and
where you installed Python. Within the Python (or Conda) directory, the files will be located in
``lib/python3.7/site-packages/octavvs`` but the executable scripts
``oct_preprocessing``, ``oct_mcr_als`` and ``oct_clustering`` will be
located in ``bin`` and should be possible to run straight from the console.

## Test data

Test data from two 64x64 images of Paxillus hyphae growing on lignin can be
[downloaded here](http://cbbp.thep.lu.se/~carl/octavvs/octavvs_test_data.zip) (zip archive, 47 MB).

## Bug reports and code repository

Developers may want to access the OCTAVVS code through the [OCTAVVS GitHub
page](https://github.com/ctroein/octavvs), where bugs and other issues can
also be reported.

Non-technical users may prefer to send questions, bug reports and other
requests by email to corresponding author Carl Troein <carl@thep.lu.se>.

