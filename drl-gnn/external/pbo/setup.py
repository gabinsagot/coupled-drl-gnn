from setuptools import setup, find_packages

setup(
    name='pbo',
    version='0.0.1',
    packages=find_packages(include=['pbo', 'pbo.*']),
    entry_points={
        'console_scripts': ['pbo=pbo.src.main:main'],
    }
)
