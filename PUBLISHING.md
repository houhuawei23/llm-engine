# Publishing Guide for llm-engine

This guide explains how to build and publish the `llm-engine` package to PyPI.

## Prerequisites

1. **PyPI Account**: Create an account at [PyPI](https://pypi.org/account/register/)
2. **TestPyPI Account** (recommended for testing): Create an account at [TestPyPI](https://test.pypi.org/account/register/)
3. **Install build tools**:
   ```bash
   pip install build twine
   ```

## Pre-release Checklist

Before publishing, ensure:

- [ ] Update version number in `pyproject.toml` and `llm_engine/__init__.py`
- [ ] Update `CHANGELOG.md` (if you have one)
- [ ] Run tests: `pytest`
- [ ] Check code quality: `ruff check .`
- [ ] Update `README.md` if needed
- [ ] Update URLs in `pyproject.toml` with your actual repository URLs

## Building the Package

1. **Clean previous builds**:
   ```bash
   rm -rf build/ dist/ *.egg-info
   ```

2. **Build source distribution and wheel**:
   ```bash
   python -m build
   ```

   This creates:
   - `dist/llm-engine-<version>.tar.gz` (source distribution)
   - `dist/llm-engine-<version>-py3-none-any.whl` (wheel)

3. **Verify the build**:
   ```bash
   # Check the contents of the wheel
   unzip -l dist/llm_engine-*.whl
   
   # Check the source distribution
   tar -tzf dist/llm-engine-*.tar.gz | head -20
   ```

## Testing on TestPyPI (Recommended)

Before publishing to production PyPI, test on TestPyPI:

1. **Upload to TestPyPI**:
   ```bash
   python -m twine upload --repository testpypi dist/*
   ```

2. **Test installation from TestPyPI**:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ llm-engine
   ```

3. **Verify it works**:
   ```python
   from llm_engine import LLMEngine, LLMConfig, LLMProvider
   # Test your imports and basic functionality
   ```

## Publishing to PyPI

1. **Upload to PyPI**:
   ```bash
   python -m twine upload dist/*
   ```

   You'll be prompted for:
   - Username: Your PyPI username
   - Password: Your PyPI password (or API token)

2. **Using API Token (Recommended)**:
   - Go to PyPI → Account Settings → API tokens
   - Create a new API token
   - Use `__token__` as username and the token as password

3. **Verify publication**:
   - Visit: https://pypi.org/project/llm-engine/
   - Test installation: `pip install llm-engine`

## Version Management

To release a new version:

1. **Update version** in `pyproject.toml`:
   ```toml
   version = "0.1.1"  # Increment as needed
   ```

2. **Update version** in `llm_engine/__init__.py`:
   ```python
   __version__ = "0.1.1"
   ```

3. **Follow semantic versioning**:
   - `MAJOR.MINOR.PATCH`
   - MAJOR: Breaking changes
   - MINOR: New features (backward compatible)
   - PATCH: Bug fixes

## Troubleshooting

### Common Issues

1. **"File already exists" error**:
   - Version number already exists on PyPI
   - Increment version number

2. **"Invalid distribution" error**:
   - Check `MANIFEST.in` includes all necessary files
   - Verify `pyproject.toml` is correct

3. **Import errors after installation**:
   - Check `__init__.py` exports are correct
   - Verify package structure matches `pyproject.toml`

### Useful Commands

```bash
# Check package metadata
python -m build --sdist --wheel
python -m twine check dist/*

# View package contents
python -m zipfile -l dist/llm_engine-*.whl

# Install in editable mode for testing
pip install -e .

# Uninstall
pip uninstall llm-engine
```

## Post-Release

After successful publication:

1. Create a git tag:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

2. Update documentation if needed
3. Announce the release (GitHub releases, etc.)

## Security Notes

- Never commit API tokens or passwords
- Use environment variables or `~/.pypirc` for credentials:
  ```ini
  [distutils]
  index-servers =
      pypi
      testpypi

  [pypi]
  username = __token__
  password = pypi-xxxxxxxxxxxxx

  [testpypi]
  username = __token__
  password = pypi-xxxxxxxxxxxxx
  ```
