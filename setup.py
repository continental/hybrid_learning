#!/usr/bin/env python
"""Setup installation script for the package hybrid_learning."""
#  Copyright (c) 2022 Continental Automotive GmbH

from setuptools import setup, find_packages

setup(
    name='hybrid_learning',
    version='2.0',
    description='Library for concept analysis and enforcement of DNNs',
    license='MIT',
    maintainer='Gesina Schwalbe',
    maintainer_email='gesina.schwalbe@continental-corporation.com',
    packages=find_packages(exclude=['test*', '*rules*']),
    install_requires=[
        'torch (>=1.6)',
        'torchvision (>=0.7.0)',
        'tensorboard',
        'numpy',
        'pandas',
        'matplotlib',
        'pillow',
        'pyparsing (>=2.0.2)',
        'tqdm',
        'pycocotools',  # For the COCO dataset handles
    ],
    provides=['hybrid_learning'],
)
