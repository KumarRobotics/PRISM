from setuptools import find_packages, setup

# todo add deps, all that
setup(
    name="prism",
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[""],
    extra_requires={"examples": [""]},
)
