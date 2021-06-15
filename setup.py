import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="process-improve",
    version="0.9.16",
    author="Kevin Dunn",
    author_email="kgdunn@gmail.com",
    description="Process Improvement using Data: Designed Experiments; Latent Variables (PCA, PLS, multivariate methods with missing data); Process Monitoring; Batch data analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kgdunn/process_improve",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "pandas",
        "numpy",
        "statsmodels",
        "matplotlib",
        "bokeh",
        "sklearn",
        "patsy",
        "scikit-image",
    ],
    # Include additional files into the package
    include_package_data=True,
)
