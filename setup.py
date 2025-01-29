from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CustomBuildExt(build_ext):
    def finalize_options(self):
        if self.include_dirs is None:
            self.include_dirs = []

        import numpy
        self.include_dirs.append(numpy.get_include())
        super().finalize_options()


requirements = [line.strip() for line in open("requirements.txt").readlines()]

setup(
    name="StatTools",
    version="1.6.0",
    ext_modules=[Extension('C_StatTools', sources=[
                           'StatTools_C_API.cpp'], language='c++')],
    cmdclass={'build_ext': CustomBuildExt},
    author="Alexandr Kuzmenko",
    author_email="alexander.k.spb@gmail.com",
    packages=["StatTools", "StatTools.analysis",
              "StatTools.generators", "StatTools.tests"],
    include_package_data=True,
    install_requires=requirements,
    description="A set of tools which allows to generate and process long-term dependent datasets",
)
