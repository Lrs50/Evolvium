from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='evolvium',
    version='0.0.1.2',
    description='An efficient and direct implementation of bioinspired algorithms designed for solving optimization problems.',
    author='Lucas Reis',
    author_email='lucaspook12@gmail.com',
    license='GNU GENERAL PUBLIC LICENSE',
    long_description=long_description, 
    long_description_content_type="text/markdown",
    packages=find_packages(),  # Automatically include all packages
    install_requires=[  
        'numpy>=1.26.4',
        'matplotlib>=3.9.2',
        'tqdm>=0.0.1',
    ],
)