from setuptools import find_packages, setup

setup(
    name='evolution',
    packages=find_packages('evolution'),
    version='0.1.0',
    description='A straightforward implementation of the widely utilized Genetic Algorithm for optimization tasks.',
    author='Lucas Reis',
    install_requires=[  'numpy==1.26.4',
                        'pandas<2.0',
                        'matplotlib==3.9.2',
                        'tdqm'],
    setup_requires=[    'pytest-runner'],
    tests_require=[     'pytest==4.4.1'],
    test_suite='tests',
)