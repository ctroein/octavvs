#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 11:41:29 2019

@author: EelkeSpaak
"""

import io
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QImage


def add_clipboard_to_figures():
    oldfig = plt.figure
    def newfig(*args, **kwargs):
        fig = oldfig(*args, **kwargs)
        def clipboard_handler(event):
            if event.key == 'ctrl+c':
                buf = io.BytesIO()
                fig.savefig(buf)
                QApplication.clipboard().setImage(QImage.fromData(buf.getvalue()))
                buf.close()
        fig.canvas.mpl_connect('key_press_event', clipboard_handler)
        return fig
    plt.figure = newfig

