"""Setup file for packaging"""
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setuptools.setup(
    name="explainable-transformer",
    version="0.2.0",
    author="Rohan Awhad and Shreya Saxena",
    author_email="rohanawhad@gmail.com",
    description="Software for debugging transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RohanAwhad/explainable-transformers",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "loguru",
        "numpy==1.21.6",
        "shap==0.41.0",
        "transformers==4.20.1",
        "torch==1.12.0",
    ],
)
