from setuptools import setup, find_packages
#from numpy.distutils.core import Extension, setup
import platform
#from setuptools import setup, Extension
#from numpy import get_include
#from numpy.f2py import f2py_extension

# fflags = ['-ffixed-line-length-0', '-march=native', '-O3', '-fPIC']
# if platform.system() == 'Darwin':  # macOS
#     fflags.append('-static-libgfortran')
#
# fortran_extension = Extension(
#     name='pygsfit.fit_Spectrum_Kl',
#     sources=[
#         'pygsfit_cp/fortran_src/Calc_GS_Spec_hom.for',
#         'pygsfit_cp/fortran_src/angular.for',
#         'pygsfit_cp/fortran_src/spidrsub.for',
#         'pygsfit_cp/fortran_src/FitFun.for',
#         'pygsfit_cp/fortran_src/ResolvedSpectrum.for',
#         'pygsfit_cp/fortran_src/klein.f',
#         'pygsfit_cp/fortran_src/fit_Spectrum_Kl.for'
#     ],
#     extra_f90_compile_args=fflags,
#     extra_f77_compile_args=fflags,
#     extra_link_args=['-shared'],
# )

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='pygsfit_cp',  #
    version='0.1.4',  #
    description='A python wrpper of the core fitting functionalities of gsFit.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Yuqian Wei',
    author_email='yw633@njit.edu',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'pygsfit_cp': [
            'IDL_utils/*.pro',
            'unix/arm64/*.so',
            'unix/x86/*.so',
            'linux/*.so',
            'win/*.dll',
            'demo/*.fits',
            'demo/*.py',
            'demo/*.ipynb',
            'docs/*.txt',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    #ext_modules=[fortran_extension],  #
    install_requires=[
        'numpy>=1.25.0',  #
        'astropy>=6.0.0',
        'dask>=2024.3.0',
        'h5py>=3.6.0',
        'matplotlib>=3.8.2',
        'scipy>=1.12.0',
        'sunpy>=5.1.1',
    ],
    setup_requires=[
        'numpy',  #
    ],
)