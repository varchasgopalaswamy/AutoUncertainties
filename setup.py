# -*- coding: utf-8 -*-
import os
import re
from setuptools import find_packages, setup, Command
from codecs import open

# Get the long description from the README file


def my_setup():
    setup(
        name="uncert",
        author="Varchas Gopalaswamy",
        author_email="vgop@lle.rochester.edu",
        description="Linear Uncertainty Propagation",
        packages=find_packages(exclude=["tests"]),
        package_dir={"": ".\\" if os.name == "nt" else "./"},
        # include_package_data=True,
        python_requires=">=3.8",
        install_requires=["numpy >= 1.18.1", "pint>=0.10.1", "hypothesis", "jax"],
    )


if __name__ == "__main__":
    my_setup()
