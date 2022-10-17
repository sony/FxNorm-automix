from setuptools import setup
# pip install --editable .

with open('README.md') as file:
    long_description = file.read()
# pip install -e .
setup(
   name='automix',
   version='0.0.1',
   description='Automix',
   author='Marco Martinez',
   author_email='Marco.Martinez@sony.com',
   url='http://43.4.24.48:10080/gitlab/atd2/martinez/automix',
   packages=['automix'],
   long_description=long_description,
   long_description_content_type='text/markdown',
   keywords='',
   license='',
   classifiers=[],
   install_requires=[],
)