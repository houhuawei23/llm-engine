# Quick Start Guide for Publishing llm-engine

## Step 1: Install Build Tools

```bash
pip install build twine
```

## Step 2: Update Version (if needed)

Edit `pyproject.toml` and `llm_engine/__init__.py`:
- Current version: `0.1.0`
- Update URLs in `pyproject.toml` with your actual repository URLs

## Step 3: Build the Package

```bash
cd /home/hhw/Desktop/00_Personal/my_scripts/llm-engine
python -m build
```

This creates:
- `dist/llm-engine-0.1.0.tar.gz`
- `dist/llm-engine-0.1.0-py3-none-any.whl`

## Step 4: Check the Build

```bash
python -m twine check dist/*
```

## Step 5: Test on TestPyPI (Recommended)

```bash
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ llm-engine
```

## Step 6: Publish to PyPI

```bash
python -m twine upload dist/*
```

You'll need:
- PyPI username
- PyPI password or API token

To create an API token:
1. Go to https://pypi.org/manage/account/
2. Create API token
3. Use `__token__` as username and token as password

## Verify Installation

```bash
pip install llm-engine
python -c "from llm_engine import LLMEngine; print('Success!')"
```

## Troubleshooting

- **"File already exists"**: Version already on PyPI, increment version
- **"Invalid distribution"**: Check MANIFEST.in and pyproject.toml
- **Import errors**: Verify __init__.py exports

For detailed instructions, see `PUBLISHING.md`.
