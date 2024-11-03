from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np


VERSION = "2.0.1"

# Define extensions individually
extensions = [
    Extension(
        "etlportfolio.optimized.risk_criteria",
        ["etlportfolio/optimized/risk_criteria.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(
    name="etlportfolio",
    version=VERSION,
    author="Jack Tobin",
    author_email="tobjack330@gmail.com",
    description="Expected tail loss portfolio optimisation in python",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jamesjtobin/etlportfolio",
    project_urls={
        "Bug Tracker": "https://github.com/jamesjtobin/etlportfolio/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows :: Windows 11",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "yfinance",
        "pandas",
        "numpy",
        "seaborn",
        "matplotlib",
        "scipy",
    ],
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': "3"}
    ),
)
