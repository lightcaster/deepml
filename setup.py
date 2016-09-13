from setuptools import setup

setup(name='deepml',
    version='0.1',
    description='Theano-based set of utilities for composing neural network models',
    author='Konstantin Selivanov',
    author_email='konstantin.selivanov@gmail.com',
    packages=['deepml'],
    zip_safe=False,
    scripts=[
        'deepml/scripts/dml_whiten.py',
    ],
    )


