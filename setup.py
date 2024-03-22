from setuptools import setup
import setuptools

import sentence_store

with open('sentence_store/requirements.txt') as f:
    required = f.read().splitlines()
with open("README.md", "r") as f:
    long_description = f.read()

version = sentence_store.__version__
setup(name='sentence_store',
      version=version,
      description='Tool to extract and store sentence embeddings to a fast and scalable vector db',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/ptarau/sentence_store.git',
      author='Paul Tarau',
      author_email='paul.tarau@gmail.com',
      license='MIT',
      packages=setuptools.find_packages(),
      # packages=['sentence_store'],
      package_data={
          'sentence_store': [
              'requirements.txt'
          ]
      },
      include_package_data=True,
      install_requires=required,
      zip_safe=False
      )
