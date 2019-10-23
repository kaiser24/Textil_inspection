#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 18:04:02 2019

@author: felipe
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("fast_localEnergy.pyx")
)