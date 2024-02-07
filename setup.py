from setuptools import find_packages, setup

setup(
    name='timefiller',
    version='0.1.0',
    packages=find_packages(),
    install_requires=['numpy',
                      'pandas',
                      'optimask',
                      'statsmodels'],
    python_requires='>=3.8',
    author='Cyril Joly',
    description='A package for imputing missing data in time series',
    classifiers=['License :: OSI Approved :: MIT License'],
)
