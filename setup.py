import setuptools
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setuptools.setup(
    name="hate_measure",
    version="0.0.0",
    description="Package for the D-Lab's 'Measuring Hate Speech' project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://hatespeech.berkeley.edu/",
    classifiers=[
        "Development Status :: 4 - Beta"
    ]
)
