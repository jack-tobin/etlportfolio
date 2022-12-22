
from setuptools import setup

dependencies = [
    'pandas',
    'numpy',
    'scipy',
    'matplotlib',
    'seaborn',
    'yfinance'
]

setup(
    name='ETL Portfolio',
    version='0.0.1',
    description='ETL portfolio optimisation in Python',
    author='Jack Tobin, CFA',
    author_email='tobjack330@gmail.com',
    requires=dependencies
)
