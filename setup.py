from setuptools import find_packages, setup

setup(
    name='evolvium',
    packages=find_packages('evolvium'),
    version='0.1.0',
    description='An efficient and direct implementation of bioinspired algorithms designed for solving optimization problems.',
    author='Lucas Reis',
    install_requires=[  'numpy==1.26.4',
                        'pandas<2.0',
                        'matplotlib==3.9.2',
                        'tdqm'],
    setup_requires=[    'pytest-runner'],
    tests_require=[     'pytest==4.4.1'],
    test_suite='tests',
)