from setuptools import setup, find_packages

setup(
    name="GrinderBot",
    version="0.0.1",
    packages=find_packages(exclude=["tests", "tests.*", "test_*", "docs"]),
)