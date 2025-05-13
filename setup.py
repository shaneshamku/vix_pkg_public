from setuptools import setup, find_packages

setup(
    name="vix_pkg",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
    'alpaca-trade-api==3.2.0',
    'numpy==2.2.3',
    'packaging==24.2',
    'pandas==2.2.3',
    'python-dateutil==2.9.0.post0',
    'python-dotenv==1.0.1',
    'requests==2.32.3',
    'scipy==1.15.2',
    'setuptools==76.0.0',
    'twine==6.1.0',
    'typing_extensions==4.12.2',
    'tzdata==2025.1',
    'urllib3==1.26.20',
    'yfinance==0.2.54',
    ],
    description='VIX ETF Trading strategy, trades executed with Alpaca',
    author='Cassel Robson',
    author_email='robs7000@mylaurier.ca',
    classifiers=[
        'Programming Language :: Python :: 3.8',
    ],
)

