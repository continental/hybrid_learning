#!/usr/bin/env python
#  Copyright (c) 2020 Continental Automotive GmbH

from setuptools import setup, find_packages

setup(
    name='hybrid_learning',
    version='0.1',
    description='Library for concept analysis and enforcement of DNNs',
    license='MIT',
    maintainer='Gesina Schwalbe',
    maintainer_email='gesina.schwalbe@continental-corporation.com',
    packages=find_packages(exclude=['test*', '*rules*']),
    install_requires=[
        'torch (>=1.6)',
        'torchvision (>=0.7.0)',
        'numpy',
        'pandas',
        'matplotlib',
        'pillow',
        'tqdm',
        'pycocotools',  # For the COCO dataset handles
    ],
    provides=['hybrid_learning'],
)
