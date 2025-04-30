# Contributing to QBits

Thank you for your interest in contributing to QBits! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. We aim to foster an inclusive and welcoming community.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with the following information:
- A clear, descriptive title
- Steps to reproduce the bug
- Expected behavior
- Actual behavior
- Any relevant logs or screenshots

### Suggesting Enhancements

If you have ideas for enhancements, please create an issue with:
- A clear, descriptive title
- A detailed description of the proposed enhancement
- Any relevant examples, mockups, or references

### Pull Requests

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes
4. Run tests to ensure your changes don't break existing functionality
5. Submit a pull request with a clear description of the changes

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/qbits.git
   cd qbits
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run the main script to test your setup:
   ```bash
   python src/main.py
   ```

## Project Structure

- `src/core/`: Core quantum simulation modules
- `src/models/`: Neural network models
- `src/visualization/`: Visualization tools
- `src/utils/`: Utility modules
- `data/`: Training data
- `models/`: Saved neural network models
- `images/`: Visualization outputs

## Coding Standards

- Follow PEP 8 style guidelines
- Write docstrings for all functions, classes, and modules
- Include type hints where appropriate
- Write unit tests for new functionality

Thank you for contributing to QBits!
