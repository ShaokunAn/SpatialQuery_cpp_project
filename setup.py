import pybind11
from setuptools import setup, Extension, find_packages

# Define the C++ extension
cpp_extension = Extension(
    name="spatial_module",
    sources=[
        "cpp_src/SingleFOVKDTree.cpp",
        "cpp_src/utils.cpp",
        "cpp_src/FPGrowth.cpp",
        "cpp_src/MultipleFOVKDTree.cpp",
    ],
    include_dirs=["cpp_src", "/opt/homebrew/Cellar/pcl/1.14.0_1/include/pcl-1.14",
                  "/opt/homebrew/Cellar/boost/1.84.0_1/include",
                  "/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3",
                  "/opt/homebrew/Cellar/flann/1.9.2_1/include"] + [pybind11.get_include()],
    libraries=["pcl_common", "pcl_kdtree", "pcl_search", "flann"],
    library_dirs=["/opt/homebrew/Cellar/pcl/1.14.0_1/lib",
                  "/opt/homebrew/Cellar/flann/1.9.2_1/lib"],
    language="c++",
    extra_compile_args=["-std=c++14"],
)

setup(
    name='SpatialQuery',
    version='0.1',
    packages=find_packages(),
    author='Shaokun An',
    author_email='shan12@bwh.harvard.edu',
    description='',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.8',
    install_requires=[
        'pybind11>=2.11.1',
        'setuptools>=68.0.0',
        'anndata>=0.8.0',
        'matplotlib>=3.7.3',
        'numpy>=1.24.4',
        'pandas>=2.0.3',
        'seaborn>=0.13.0',
        'scipy>=1.10.1'
    ],
    ext_modules=[cpp_extension],
)
