from setuptools import setup, find_packages

setup(
    name="mevgs",
    version="0.0.1",
    author="Dan Oneață",
    author_email="dan.oneata@gmail.com",
    description="Mutual exclusivity bias in visually grounded speech models",
    packages=["mevgs"],
    install_requires=["black", "click", "streamlit", "ruff"],
)
