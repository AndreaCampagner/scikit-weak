from setuptools import setup, find_packages

setup(
   name='scikit-weak',
   version='0.2.2',
   author='Andrea Campagner',
   python_requires=">3.8.0",
   author_email='a.campagner@campus.unimib.it',
   packages=find_packages(include=['scikit_weak', 'scikit_weak.*']),
   url='https://pypi.org/project/scikit-weak/',
   license='LICENSE.txt',
   description='A package featuring utilities and algorithms for weakly supervised ML.',
   long_description=open('README.md').read(),
   long_description_content_type='text/markdown',
   install_requires=[
       "scikit-learn>=1.0.0",
       "numpy>=1.19.2",
       "scipy>=1.3.2",
       "tensorflow>=2.4.0",
       "keras>=2.4.0",
       "pytest"
   ],
)