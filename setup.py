from setuptools import find_packages, setup

import numpy as np


with open("README.md", 'r') as f:
    long_description = f.read()


setup(
    name="ConfirmatoryLDA",
    version="0.0.4",
    packages=find_packages(),
    author="JongHo Im",
    description="A summarizing method for topic model",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/YD3SL/CLDA",
    include_dirs=np.get_include(),
    install_requires=[
        'numpy>=1.19.2'
    ]
)
