#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 03:28:27 2019

@author: carl
"""

import os
import sys
from pyshortcuts import make_shortcut
import argparse

def main():
    parser = argparse.ArgumentParser(
            description='Creates icons for the OCTAVVS scripts.')
    _ = parser.parse_args()

    executables = {'oct_preprocessing': 'Preprocessing',
#                   'oct_mcr_als': 'MCR-ALS',
#                   'oct_clustering': 'Clustering',}
                   'oct_decomposition': 'Decomposition'}
    proj = 'OCTAVVS '

    for cmd, nom in executables.items():
        icon = os.path.join(os.path.split(__file__)[0], 'prep', 'octavvs_prep')
        script = os.path.join(os.path.split(sys.argv[0])[0], cmd)

        make_shortcut(script, name=proj+nom, icon=icon,
                      startmenu=False, terminal=True, desktop=True)

if __name__ == '__main__':
    main()


