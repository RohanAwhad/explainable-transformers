"""Setup file for packaging"""
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Explainable Transformers",
    version="0.1.0",
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
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "shap",
        "streamlit",
        "transformers",
        "torch",
        "matplotlib",
    ],
)
