from setuptools import setup, find_packages

setup(
   name='skweak',
   version='0.1.0',
   author='Andrea Campagner',
   author_email='a.campagner@campus.unimib.it',
   packages=find_packages(include=['skweak', 'skweak.*']),
   #url='http://pypi.python.org/pypi/PackageName/',
   #license='LICENSE.txt',
   description='A package featuring utilities and algorithms for weakly supervised ML.',
   long_description=open('README.md').read(),
   install_requires=[
       "sklearn",
       "pandas",
       "numpy",
       "scipy"
   ],
)