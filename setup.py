from setuptools import setup
from numpy import get_include
from setuptools import setup, Extension

module = Extension('C_StatTools',
    include_dirs=[get_include()],
    sources=['StatTools_C_API.cpp'],
    language='c++')

with open('requirements.txt') as f:
    requirements = f.read.splitlines()

setup(
    name="StatTools",
    version="1.6.0",
    ext_modules=[module],
    author="Alexandr Kuzmenko",
    author_email="alexander.k.spb@gmail.com",
    packages=["StatTools", "StatTools.analysis", "StatTools.generators", "StatTools.tests"],
    include_package_data=True,
    install_requires=requirements,
    description="A set of tools which allows to generate and process long-term dependent datasets",
)
