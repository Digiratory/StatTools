from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CustomBuildExt(build_ext):
    def finalize_options(self):
        import numpy
        self.include_dirs.append(numpy.get_include())
        super().finalize_options()


requirements = ['memory-profiler==0.58.0', 'numpy==1.22.4', 'psutil==5.8.0', 'Pympler==0.9', 'scipy==1.13.1',
                'setuptools~=57.0.0', 'matplotlib~=3.5.1', 'seaborn~=0.11.2', 'tqdm~=4.62.3', 'Pillow~=8.4.0', 'pandas~=1.3.5', 'rich>=10.9.0']

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
