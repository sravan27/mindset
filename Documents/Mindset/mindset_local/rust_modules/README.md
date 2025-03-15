# MINDSET Rust Acceleration Module

This directory contains Rust modules used to accelerate computationally intensive operations in the MINDSET application, primarily for feature engineering.

## Overview

The Rust modules are compiled into Python extensions using PyO3, allowing seamless integration with the Python codebase while providing significant performance improvements for specific operations.

## Modules

- **text_processing**: Fast text processing operations including tokenization, n-gram extraction, and feature calculation
- **vector_operations**: Optimized vector operations for embeddings and similarity calculations
- **metrics_calculation**: Fast calculation of metrics including Political Influence, Rhetoric Intensity, and Information Depth scores

## Setup

These modules are automatically built during the MINDSET environment setup process. If you need to rebuild them manually:

1. Ensure Rust is installed (https://www.rust-lang.org/tools/install)
2. Navigate to this directory
3. Run `maturin develop` to build and install the modules in development mode

## Usage

Once built, the modules can be imported in Python:

```python
import mindset_rust

# Example: Process text using the accelerated functions
processed_text = mindset_rust.process_text("Your text here")
```

## Development

To extend or modify these modules:

1. Edit the Rust code in the `src` directory
2. Run `maturin develop` to rebuild
3. Test your changes in Python

For detailed API documentation, see the function headers in the source files.