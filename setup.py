import os
import pathlib
import pkg_resources
from setuptools import setup, find_packages


PKG_NAME = "secant"
VERSION = "1.0"


def _read_file(fname):
    with pathlib.Path(fname).open() as fp:
        return fp.read()


def _read_install_requires():
    with pathlib.Path("requirements.txt").open() as fp:
        return [
            str(requirement) for requirement in pkg_resources.parse_requirements(fp)
        ]


setup(
    name=PKG_NAME,
    version=VERSION,
    author=f"SECANT Authors",
    description="SECANT ICML 2021",
    long_description=_read_file("README.md"),
    long_description_content_type="text/markdown",
    keywords=["Deep Learning", "Machine Learning"],
    license="MIT License",
    packages=find_packages(include=PKG_NAME),
    include_package_data=True,
    zip_safe=False,
    install_requires=_read_install_requires(),
    python_requires="==3.7.*",
    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Environment :: Console",
        "Programming Language :: Python :: 3",
    ],
)
