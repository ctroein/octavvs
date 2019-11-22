#!/bin/bash

rm -r dist build &&
PATH=~/anaconda3/bin/:$PATH python setup.py bdist_wheel  &&
PATH=~/anaconda3/bin/:$PATH pip install -U dist/*.whl
