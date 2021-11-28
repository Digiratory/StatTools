from setuptools import setup
from numpy import get_include
from setuptools import setup, Extension

module = Extension('C_StatTools',
    include_dirs=[get_include()],
    sources=['StatTools_C_API.cpp'],
    language='c++')

setup(
    name="StatTools",
    version="1.2.1",
    ext_modules=[module],
    author="Alexandr Kuzmenko",
    author_email="alexander.k.spb@gmail.com",
    packages=["StatTools", "StatTools.analysis", "StatTools.generators", "StatTools.tests"],
    include_package_data=True,
    install_requires=[
        "numpy >= 1.19.2",
        "tqdm >= 4.50.1",
        "rich >= 10.9.0"
    ],
    description="A set of tools which allows to generate and process long-term dependent datasets",
)