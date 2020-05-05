from setuptools import find_packages, setup
from pathlib import Path


def list_reqs(fname='requirements.txt'):
    with open(fname) as fd:
        return fd.read().splitlines()


with open('README.md') as f:
    readme = f.read()

ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / 'vehicles_model'
about = {}
with open(PACKAGE_DIR / 'VERSION') as f:
    _version = f.read().strip()
    about['__version__'] = _version


setup(
    name='vehicles-model',
    version=about['__version__'],
    description='Regression model to estimate sale prices of used cars',
    long_description=readme,
    author='Ali Naji',
    author_email='anaji7@gatech.edu',
    url='https://github.com/ali-naji',
    license='MIT',
    packages=find_packages(exclude=('tests', 'docs')),
    package_data={'vehicles_model': ['VERSION']},
    install_requires=list_reqs(),
    python_requires='>=3.6.0',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6']
)
