# LLM Engine

Unified LLM API provider and engine library for Python.

## Features

- Support for multiple LLM providers (OpenAI, DeepSeek, Ollama, Custom)
- Both synchronous and asynchronous API support
- Unified configuration via `providers.yml`
- Environment variable resolution
- Automatic retry logic
- Streaming support
- Token estimation and resource management

## Installation

Install from PyPI:

```bash
pip install llm-engine
```

For local development:

```bash
cd llm-engine
pip install -e .
```

Or install with development dependencies:

```bash
pip install -e ".[dev]"
```

## Configuration

Create a `providers.yml` file:

```yaml
providers:
  deepseek:
    base_url: "https://api.deepseek.com/v1"
    api_key: ${DEEPSEEK_API_KEY}
    default_model: "deepseek-chat"
    models:
      - name: "deepseek-chat"
        context_length: 128000
        functions:
          json_output: true
```

## Usage

### Async Usage

```python
from llm_engine import LLMConfig, LLMProvider, LLMEngine

config = LLMConfig(
    provider=LLMProvider.DEEPSEEK,
    model_name="deepseek-chat",
    api_key="your-api-key",
)

engine = LLMEngine(config)
response = await engine.generate("Hello, world!")
```

### Sync Usage

```python
from llm_engine import LLMConfig, LLMProvider
from llm_engine.providers.openai_compatible import OpenAICompatibleProvider

config = LLMConfig(
    provider=LLMProvider.DEEPSEEK,
    model_name="deepseek-chat",
    api_key="your-api-key",
)

provider = OpenAICompatibleProvider(config)
response = provider.call("Hello, world!")
```

## License

MIT
