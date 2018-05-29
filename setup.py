from setuptools import setup, find_packages

setup(name='pyBreakDown',
      version='0.0.1',
      description='breakDown python implementation',
      url='http://github.com/bondyra/pyBreakDown',
      author='Jakub Bondyra',
      author_email='jb10193@gmail.com',
      license='GPL-2',
      packages= find_packages(exclude=['tests']),
      install_requires=[
	'numpy==1.14.2',
	'scikit-learn==0.19.1',
	'scipy==1.0.0',
	'blist==1.3.6',
	'sphinx-bootstrap-theme==0.6.5',
	'matplotlib==2.1.2',
	'recordclass==0.5'])

