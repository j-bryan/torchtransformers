from setuptools import setup, find_packages

setup(
    name="torchtransformers",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=2.3.0"
    ],
    author="Jacob Bryan",
    author_email="jbryan314@gmail.com",
    description="A description of your package",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
