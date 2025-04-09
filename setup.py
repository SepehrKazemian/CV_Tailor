#!/usr/bin/env python3
"""
Setup script for the resume_tailor package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="resume_tailor",
    version="1.0.0",
    author="Resume Tailor Team",
    author_email="example@example.com",
    description="A smart resume tailoring system that reads job descriptions and matches them with your resume",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/resume_tailor",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "resume-tailor=resume_tailor_cli:main",
        ],
    },
)
