from setuptools import find_packages, setup

setup(
    name="uptake_of_np",
    version="0.1.0",
    author="Sarah Iaquinta",
    author_email="sarah.r.iaquinta@gmail.com",
    packages=find_packages(),
    url="https://github.com/SarahIaquinta/PhDthesis",
    description="Repository relative to the PhD thesis of Sarah Iaquinta",  # TODO add description
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
