# -*- coding: utf-8 -*-
import os
import re
from setuptools import find_packages, setup, Command
from codecs import open

# Get the long description from the README file


def my_setup():
    setup(
        name="auto_uncertainties",
        packages = ['auto_uncertainties'],
        author="Varchas Gopalaswamy",
        license='GPLv3',
        author_email="vgop@lle.rochester.edu",
        description="Linear Uncertainty Propagation with Auto-Differentiation",
        url = 'https://github.com/varchasgopalaswamy/AutoUncertainties',
        download_url = 'https://github.com/varchasgopalaswamy/AutoUncertainties/archive/refs/tags/0.1.1.tar.gz',
        package_dir={"": ".\\" if os.name == "nt" else "./"},
        # include_package_data=True,
        python_requires=">=3.8",
        install_requires=["numpy >= 1.18.1", "hypothesis", "jax"],
    )


if __name__ == "__main__":
    my_setup()
