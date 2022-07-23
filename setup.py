import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="simpthon",
    version="1.0.0",
    author="Hasanuddin",
    author_email="hasanuddin@physics.untan.ac.id",
    description="simulating a particle or particles with python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["simpthon"],
    install_requires=["numpy"]
)