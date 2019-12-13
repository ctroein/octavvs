#!/bin/bash

rm -rf dist build &&
python setup.py sdist bdist_wheel &&
echo pip install -U dist/*.whl
