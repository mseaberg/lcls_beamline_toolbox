from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["numpy>=1.17", "scipy>=1.4.1", "matplotlib>=3.2.1", "jupyter>=1.0.0", "scikit-image>=0.16.2"]

setup(
    name="lcls_beamline_toolbox",
    version="0.0.1",
    author="Matt Seaberg",
    author_email="seaberg@slac.stanford.edu",
    description="tools for LCLS beamline calculations",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/mseaberg/lcls_beamline_toolbox/",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
    ],
)
