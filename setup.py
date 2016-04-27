#!/usr/bin/env python

from setuptools import setup, find_packages

args = dict(
    name='arm_trajectories',
    version='0.1',
    description='Create and publush trajectories for SCARA arms.',
    packages=['arm_trajectories'],
    install_requires=[],
    author='Florian Reinhard',
    author_email='florian.reinhard@epfl.ch',
    url='https://github.com/cvra',
    license='BSD'
)

setup(**args)
