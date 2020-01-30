#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 21:57:05 2020

@author: carl
"""

from PyQt5.QtWidgets import QStyle, QProxyStyle

class NoRepeatStyle(QProxyStyle):
    def styleHint(self, hint, option=None, widget=None, returnData=None):
        if hint == QStyle.SH_SpinBox_ClickAutoRepeatThreshold:
            return 10000
        else:
            return super().styleHint(hint, option, widget, returnData)
