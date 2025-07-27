# Documentation for process-improve

This directory contains Sphinx documentation for the process-improve package.

## Building the Documentation

### Requirements

Install the documentation requirements:

```bash
pip install -r requirements.txt
```

Or if you have the full development environment:

```bash
pip install sphinx sphinx-rtd-theme
```

### Building

To build the HTML documentation:

```bash
make html
```

To clean the build directory:

```bash
make clean
```

## Content Overview

- `multivariate.rst` - Comprehensive documentation for multivariate analysis methods (PCA, PLS, TPLS)
- `conf.py` - Sphinx configuration
- `index.rst` - Main documentation index

## Key Features

The documentation includes:

- **Comprehensive multivariate analysis coverage** for PCA, PLS, and TPLS methods
- **Working code examples** that can be copied and run
- **Mathematical background** and theory explanations
- **Practical guidance** for data preprocessing and model selection
- **Performance optimization** tips and best practices
- **Troubleshooting guide** for common issues
- **Complete API reference** with parameter descriptions

All code examples have been tested and verified to work correctly with the current version of the package.