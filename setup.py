from setuptools import setup
from numpy import get_include
from setuptools import setup, Extension

module = Extension('C_StatTools',
                   include_dirs=[get_include()],
                   sources=['StatTools_C_API.cpp'],
                   language='c++')

requirements = [line.strip() for line in open("requirements.txt").readlines()]

setup(
    ext_modules=[module],
    packages=["StatTools", "StatTools.analysis",
              "StatTools.generators"],
    include_package_data=True,
    install_requires=requirements,
    description="A set of tools which allows to generate and process long-term dependent datasets",
)
