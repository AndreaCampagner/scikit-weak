from setuptools import setup, find_packages

setup(
   name='scikit-weak',
   version='0.1.3a2',
   author='Andrea Campagner',
   author_email='a.campagner@campus.unimib.it',
   packages=find_packages(include=['scikit_weak', 'scikit_weak.*']),
   url='https://pypi.org/project/scikit-weak/',
   license='LICENSE.txt',
   description='A package featuring utilities and algorithms for weakly supervised ML.',
   long_description=open('README.md').read(),
   install_requires=[
       "scikit-learn",
       "pandas",
       "numpy",
       "scipy"
   ],
)