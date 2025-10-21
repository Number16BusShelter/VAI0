# CONTINUE.md - Project Guide

## Project Overview

This appears to be a Python project with a focus on audio processing or voice-related functionality, given the presence of a file named `kokoro_test.wav`. The project uses modern Python packaging standards with a `pyproject.toml` file and has a test directory structure.

### Key Technologies Used

- Python (primary language)
- Poetry (dependency management, via pyproject.toml)
- Standard Python testing structure
- Makefile for automation

### High-Level Architecture

The project follows a standard Python project structure with:

- A `vaio/` directory likely containing the main implementation
- A `tests/` directory for unit and integration tests
- Configuration files in the root directory
- Audio test files for validation

## Getting Started

### Prerequisites

- Python 3.8+
- Poetry (for dependency management)
- Git (for version control)

### Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <project-name>
   ```
2. Install dependencies using Poetry:

   ```bash
   poetry install
   ```
3. Set up the development environment:

   ```bash
   poetry shell
   ```

### Basic Usage

The exact usage will depend on the specific implementation in the `vaio/` directory. Common patterns might include:

```python
from vaio import main_module

result = main_module.process_audio("input.wav")
print(result)
```

### Running Tests

Use the standard Python testing workflow:

```bash
python -m pytest tests/
```

## Project Structure

### Main Directories

- **`vaio/`**: Main implementation directory
  - Likely contains the core functionality of the project
- **`tests/`**: Test directory
  - Follows standard Python testing conventions
- **`docs/`**: Documentation directory
  - Should contain project documentation (currently empty or not shown)

### Key Files

- **`pyproject.toml`**: Project configuration and dependencies
- **`Makefile`**: Automation script for common tasks
- **`README.md`**: Project overview and documentation
- **`config.yml`**: Configuration file for the project
- **`requirements.txt`**: Alternative dependency specification (though Poetry is preferred)

### Important Configuration Files

- **`pyproject.toml`**: Contains project metadata, dependencies, and build configurations
- **`config.yml`**: Likely contains application-specific configuration settings
- **`Makefile`**: Contains automation recipes for common tasks

## Development Workflow

### Coding Standards

- Follow standard Python PEP8 style guide
- Use type hints where possible
- Write docstrings for public APIs

### Testing Approach

- Use Python's standard testing framework (pytest)
- Maintain a `tests/` directory parallel to the main code structure
- Write unit tests for individual functions and integration tests for larger components

### Build and Deployment Process

- Use Poetry for packaging and distribution
- The `pyproject.toml` file contains build configurations
- A Makefile likely contains deployment automation

### Contribution Guidelines

1. Fork the repository
2. Create a new branch for your feature/fix
3. Write tests for your changes
4. Run all tests to ensure nothing is broken
5. Submit a pull request with clear documentation of your changes

## Key Concepts

### Domain-Specific Terminology

- **Audio Processing**: The likely domain of this project, given the presence of a `.wav` file
- **Voice Analysis**: Potential application area based on the file name "kokoro_test.wav"

### Core Abstractions

- Audio file processing pipelines
- Signal analysis and transformation
- Voice feature extraction

### Design Patterns Used

- Module-based organization
- Test-driven development (TDD) approach
- Configuration management through YAML files

## Common Tasks

### Adding a New Feature

1. Create a new module in the `vaio/` directory
2. Write tests in the corresponding `tests/` subdirectory
3. Update the `pyproject.toml` file if new dependencies are needed
4. Document the feature in the README or docs directory

### Debugging

1. Use Python's built-in debugger (pdb) or an IDE debugger
2. Add print statements for intermediate values
3. Use the `--verbose` flag with tests to get more detailed output

### Performance Optimization

1. Use Python's `cProfile` module to identify bottlenecks
2. Consider using NumPy for numerical operations
3. Profile memory usage with tools like `memory_profiler`

## Troubleshooting

### Common Issues and Solutions

- **Missing dependencies**: Run `poetry install` to ensure all dependencies are installed
- **Test failures**: Check if the test environment is properly set up and all required files are present
- **Configuration errors**: Verify that the `config.yml` file is properly formatted and contains all required settings

### Debugging Tips

- Use `print()` statements to trace execution flow
- Use an IDE with debugging capabilities (VS Code, PyCharm)
- Use logging instead of print statements for production-level debugging
- Use the `--pdb` option with pytest to automatically drop into the debugger on test failures

## References

### Documentation

- [Python Official Documentation](https://docs.python.org/3/)
- [Poetry Documentation](https://python-poetry.org/docs/)
- [pytest Documentation](https://docs.pytest.org/en/stable/)

### Important Resources

- [PEP8 Style Guide](https://peps.python.org/pep-0008/)
- [Python Packaging User Guide](https://packaging.python.org/)
- [Audio Processing with Python](https://librosa.org/doc/latest/index.html) (if audio processing is involved)
