# -*- coding: utf-8 -*-
import os
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

version = os.environ.get("CI_BUILD_TAG", "local")

setup(
    name="CANdas",
    version=version,
    author="Julian Wagenschütz",
    author_email="julian@wagenschuetz.com",
    description="Manage CAN Data elegantly.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/lionsracing/candas",
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "numpy",
        "cantools",
        "scipy",
        "python-can",
        "pandas",
    ],
    extras_require={
        # 'dev': [],
        "sql": ["SQLAlchemy", "mysqlclient"],
        'test': ['coverage'],
        "docs": ["sphinx", "alabaster", "m2r"],
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',

        'Intended Audience :: Science/Research',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    python_requires='>=3.5',
)
