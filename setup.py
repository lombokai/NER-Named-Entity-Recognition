from setuptools import setup

with open("requirements.txt", "r", encoding="utf-8") as f:
    install_requires = f.read().splitlines()

setup(
    install_requires=install_requires
)
