import os
import os.path
import fnmatch
import traceback
from pkg_resources import resource_filename

from PyQt5.QtWidgets import QWidget, QInputDialog
from PyQt5.QtWidgets import QMessageBox, QFileDialog
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt
from PyQt5 import uic

from . import NoRepeatStyle

FileLoaderUi = uic.loadUiType(resource_filename(__name__, "imagevisualizer.ui"))[0]

class ImageVisualizerWidget(QWidget, FileLoaderUi):
    """
    Widget with a bunch of graphical components related to ...
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
