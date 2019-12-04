#!/bin/bash

rm -rf dist build &&
PATH=~/anaconda3/bin/:$PATH python setup.py bdist_wheel  &&
PATH=~/anaconda3/bin/:$PATH pip install -U dist/*.whl
