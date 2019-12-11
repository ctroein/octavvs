#!/bin/bash

rm -rf dist build &&
python setup.py bdist_wheel  &&
pip install -U dist/*.whl
