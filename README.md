# PyTorch Digit Classifier

Rebuilds the Project 8 handwritten-digit classifier using PyTorch to compare framework-based training with a NumPy implementation.

## Features

- PyTorch neural network: `784 → 32 → 10`
- MNIST data pipeline and mini-batch loading
- Automated differentiation and optimization
- Model evaluation and checkpoint saving/loading
- Tests, Ruff, logging, YAML configuration, GitHub Actions, and Docker

## Quick Start

```bash
pip install -e ".[dev]"
pytest
```

## Goal

Understand what PyTorch automates compared with manual neural-network implementation in NumPy, while keeping the comparison fair and reproducible.

**Part of my AI Engineering roadmap — Project 9.**