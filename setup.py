from setuptools import setup, find_packages

entry_points = {
    'console_scripts': [
        'make_report=optical.cli:make_report',
        ]
    }

setup(
    name='solar_c',
    version='0.1',
    url='https://github.com/frederic-auchere/solar_c/',
    license='',
    author='fauchere',
    author_email='frederic.auchere@universite-paris-saclay.fr',
    description='Solar C utilities',
    packages=find_packages(),
    entry_points = entry_points
)
