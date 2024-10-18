from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='timefiller',
    version='0.1.2',
    packages=find_packages(),
    install_requires=['numpy',
                      'optimask',
                      'pandas',
                      'scikit-learn',
                      'statsmodels', 
                      'tqdm'],
    python_requires='>=3.8',
    author='Cyril Joly',
    description='A package for imputing missing data in time series',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=['License :: OSI Approved :: MIT License'],
    url="https://github.com/CyrilJl/TimeFiller"
)
