from pathlib import Path

import setuptools

with Path("README.md").open() as fh:
    long_description = fh.read()

with (Path() / "process_improve" / "__init__.py").open() as fh:
    version_number = next(
        line.split("=")[1].strip() for line in fh.read().split("\n") if line.startswith("__version__")
    ).replace('"', "")

setuptools.setup(
    name="process-improve",
    version=version_number,
    author="Kevin Dunn",
    author_email="kgdunn@gmail.com",
    description=(
        "Process Improvement using Data: Designed Experiments; Latent Variables (PCA, "
        "PLS, multivariate methods with missing data); Process Monitoring; Batch data analysis."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kgdunn/process_improve",
    packages=setuptools.find_packages(exclude=["tests"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas",
        "numpy",
        "statsmodels",
        "matplotlib",
        "bokeh",
        "scikit-learn",
        "patsy",
        "scikit-image",
        "scikit-learn",
        "plotly",
        "numba",
        "seaborn",
        "pydantic",
        "tqdm",
    ],
    # Include additional files into the package
    include_package_data=True,
)
