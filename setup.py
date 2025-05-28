from setuptools import setup

entry_points = {
    'console_scripts': [
        'sfe=optical.cli:sfe',
        ]
    }

setup(
    name='solar_c',
    version='0.1',
    packages=[''],
    url='https://github.com/frederic-auchere/solar_c/',
    license='',
    author='fauchere',
    author_email='frederic.auchere@universite-paris-saclay.fr',
    description='Solar C utilities',
    entry_points = entry_points
)
