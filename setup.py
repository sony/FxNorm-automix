from setuptools import setup

with open('README.md') as file:
    long_description = file.read()

setup(
   name='automix',
   version='0.0.1',
   description='Automix',
   author='Marco Martinez',
   author_email='Marco.Martinez@sony.com',
   url='https://github.com/sony/FxNorm-automix',
   packages=['automix'],
   long_description=long_description,
   long_description_content_type='text/markdown',
   keywords='',
   license='',
   classifiers=[],
   install_requires=[],
)