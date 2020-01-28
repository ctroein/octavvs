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

## Installation on Windows, Mac or Linux

OCCTAVS needs a working Python 3 environment with various packages. The
easiest way to get this is through the Conda package management system.

Download and install the Python 3.7 version of [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
There are some platform-specific differences during installation:

* On **Windows**: The default options are good (no need to have Conda modify your path).  
After installation, start a Conda console (found in the Start menu).

* On **Mac OS X**: The default options are good. (But _if_ you choose to not
let Conda modify your path, you can specify the full path to the commands
in the following steps.)  
After installation, start Terminal.  

* On **Linux**: Make the Miniconda .sh package executable and run it. When it asks about
running conda init, it wants to modify your $PATH in .bashrc. This can be fine, but on some
Linux distributions (notably OpenSUSE) it breaks KDE when logging in. A suggested workaround
is to change $PATH manually when needed. An alias in .bashrc can be convenient:  
``alias startconda='export PATH=~/miniconda3/bin:"$PATH"'``

From the console, install OCTAVVS and its dependencies: ``conda install -c ctroein octavvs``

## Finding and using OCTAVVS

The easiest way to access the OCTAVVS tools is through desktop shortcuts
which are created by running the ``oct_make_icons`` script from the command prompt.
(We are not entirely sure if this works on all Mac OS X versions)

Regardless of whether you created icons, the three scripts
``oct_preprocessing``, ``oct_mcr_als`` and ``oct_clustering``
can be run straight from the console (Conda console or terminal window, as described above).

The location of the OCTAVVS scripts will depend on where you installed Conda.
Within the Conda directory, the three scripts will be in ``bin`` whereas
the actual Python code will be located in ``lib/python3.7/site-packages/octavvs``.

## Test data

Test data from two 64x64 images of _Paxillus_ hyphae growing on lignin can be
[downloaded here](http://cbbp.thep.lu.se/~carl/octavvs/octavvs_test_data.zip) (zip archive, 47 MB).

## Upgrading to the latest version

Information about the most recent release of OCTAVVS can be found on its
[PyPI page](https://pypi.org/project/octavvs), as well as on its
[Anaconda Cloud page](https://anaconda.org/ctroein/octavvs).

To upgrade to the most recent version, do ``conda update octavvs`` in the Conda console / Terminal.

Information about released versions can be found [here](https://github.com/ctroein/octavvs/blob/master/HISTORY.md).

## Bug reports and code repository

The main project homepage is its [GitHub page](https://github.com/ctroein/octavvs),
where developers can access the OCTAVVS code and submit bug reports and patches etc.

Questions, bug reports and other feedback may be sent to corresponding author Carl Troein <carl@thep.lu.se>.


## Installation through pip

Users familiar with Python could also install OCTAVVS through pip as an alternative to Anaconda,
as new releases will be made to PyPI in parallel with releases to Anaconda Cloud.
Note that pyqt and opencv sometimes don't work when installed through pip, depending on your system etc.

