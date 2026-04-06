# LLM Engine

[![PyPI version](https://badge.fury.io/py/llm-api-engine.svg)](https://badge.fury.io/py/llm-api-engine)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**统一的LLM API提供商和引擎库**

LLM Engine 是一个Python库，提供统一的接口来调用多种大语言模型API。它简化了不同LLM提供商的集成，支持同步和异步操作，内置重试机制、流式输出和Token估算等功能。

---

## 目录

- [功能特性](#功能特性)
- [支持的提供商](#支持的提供商)
- [安装](#安装)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [使用指南](#使用指南)
- [高级功能](#高级功能)
- [开发指南](#开发指南)
- [项目结构](#项目结构)
- [许可证](#许可证)

---

## 功能特性

- **多提供商支持** - 统一接口调用OpenAI、DeepSeek、Anthropic、Ollama、Kimi等多种LLM服务
- **同步/异步API** - 同时支持同步调用和异步调用，满足不同应用场景
- **统一配置管理** - 通过`providers.yml`集中管理所有提供商配置
- **环境变量支持** - 支持从环境变量读取API密钥等敏感信息
- **自动重试机制** - 内置指数退避重试策略，提高调用稳定性
- **流式输出** - 支持SSE流式响应，实现实时输出效果
- **Token估算** - 提供Token消耗预估和成本计算功能
- **类型安全** - 基于Pydantic的配置验证，提供完整的类型提示
- **🆕 中间件系统** - 可插拔的请求/响应处理管道
- **🆕 智能缓存** - 精确匹配 + 语义相似度缓存，降低API成本
- **🆕 可观测性** - 请求指标、成本跟踪、预算告警
- **🆕 性能优化** - 速率限制、并发控制、连接池

---

## 支持的提供商

| 提供商 | 说明 | 默认模型 |
|--------|------|----------|
| **OpenAI** | GPT-4, GPT-3.5-Turbo系列 | gpt-4 |
| **DeepSeek** | DeepSeek Chat / Reasoner | deepseek-chat |
| **Anthropic** | Claude系列模型 | claude-3-opus |
| **Ollama** | 本地部署的开源模型 | llama2 |
| **Kimi** | Moonshot Kimi系列 | moonshot-v1-8k |
| **Kimi Code** | 专为代码优化的Kimi模型 | kimi-k2-0711-preview |
| **自定义** | 任何OpenAI兼容的API端点 | - |

---

## 安装

### 从PyPI安装

```bash
pip install llm-api-engine
```

### 开发安装

```bash
git clone https://github.com/houhuawei23/llm-engine.git
cd llm-engine
pip install -e ".[dev]"
```

### 可选依赖

```bash
# 仅安装代码检查工具
pip install -e ".[lint]"

# 仅安装安全扫描工具
pip install -e ".[security]"
```

---

## 快速开始

### 1. 配置环境变量

```bash
# DeepSeek
export DEEPSEEK_API_KEY="your-deepseek-api-key"

# OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# Kimi
export KIMI_API_KEY="your-kimi-api-key"
```

### 2. 创建配置文件

创建 `providers.yml`:

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

### 3. 开始编码

**异步使用（推荐）:**

```python
import asyncio
from llm_engine import LLMConfig, LLMProvider, LLMEngine

async def main():
    # 创建配置
    config = LLMConfig(
        provider=LLMProvider.DEEPSEEK,
        model_name="deepseek-chat",
        api_key="your-api-key",  # 或从环境变量自动读取
    )

    # 创建引擎实例
    engine = LLMEngine(config)

    # 生成响应
    response = await engine.generate("你好，请介绍一下Python编程语言")
    print(response)

    # 流式输出
    async for chunk in engine.generate_stream("讲一个短故事"):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

**同步使用:**

```python
from llm_engine import LLMConfig, LLMProvider
from llm_engine.providers.openai_compatible import OpenAICompatibleProvider

config = LLMConfig(
    provider=LLMProvider.DEEPSEEK,
    model_name="deepseek-chat",
    api_key="your-api-key",
)

provider = OpenAICompatibleProvider(config)
response = provider.call("你好，世界！")
print(response)
```

---

## 配置说明

### 配置文件结构

`providers.yml` 支持以下配置项:

```yaml
providers:
  <provider_name>:
    base_url: "API基础URL"
    api_key: "${ENV_VAR_NAME}"  # 从环境变量读取
    default_model: "默认模型名称"
    timeout: 30                  # 请求超时时间（秒）
    max_retries: 3              # 最大重试次数
    models:
      - name: "模型名称"
        context_length: 128000  # 上下文长度
        max_output:
          default: 4000         # 默认最大输出token
          maximum: 8000         # 最大输出token限制
        speed_tokens_per_second: 30  # 生成速度估算
        token_per_character:          # 字符到token的转换比例
          english: 0.3
          chinese: 0.6
        pricing_per_million_tokens:   # 定价信息（人民币/元）
          input: 2
          input_cache_hit: 0.2
          output: 3
        functions:
          json_output: true       # 是否支持JSON输出
          function_calling: true  # 是否支持函数调用
```

### 环境变量映射

| 提供商 | 环境变量名 |
|--------|-----------|
| OpenAI | `OPENAI_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| Kimi | `KIMI_API_KEY` |
| Kimi Code | `KIMI_CODE_API_KEY` |
| Ollama | 无需API密钥 |

---

## 使用指南

### LLMConfig 配置类

```python
from llm_engine import LLMConfig, LLMProvider

config = LLMConfig(
    provider=LLMProvider.DEEPSEEK,      # 提供商枚举
    model_name="deepseek-chat",          # 模型名称
    api_key="your-api-key",              # API密钥（可选，优先从环境变量读取）
    base_url=None,                       # 自定义API端点（可选）
    timeout=30,                          # 超时时间（秒）
    temperature=0.7,                     # 温度参数
    max_tokens=4096,                     # 最大生成token数
    max_retries=3,                       # 最大重试次数
)
```

### LLMEngine 异步引擎

```python
from llm_engine import LLMEngine

engine = LLMEngine(config)

# 普通生成
response = await engine.generate(
    prompt="你的问题",
    temperature=0.8,
    max_tokens=2000,
)

# 带历史记录的对话
messages = [
    {"role": "system", "content": "你是一个有帮助的助手"},
    {"role": "user", "content": "你好"},
]
response = await engine.chat(messages)

# 流式生成
async for chunk in engine.generate_stream("讲个故事"):
    print(chunk, end="")

# 估算token消耗
estimated_tokens = engine.estimate_tokens("待估算的文本")
```

### 同步Provider直接调用

```python
from llm_engine.providers.openai_compatible import OpenAICompatibleProvider

provider = OpenAICompatibleProvider(config)

# 同步调用
response = provider.call("你好")

# 同步流式调用
for chunk in provider.call_stream("讲个故事"):
    print(chunk, end="")
```

### 从配置文件加载

```python
from llm_engine.config_loader import load_providers_config, create_llm_config_from_provider

# 加载配置文件
config_data = load_providers_config("providers.yml")

# 从配置创建LLMConfig
config = create_llm_config_from_provider("deepseek")

# 获取模型信息
model_info = get_model_info("deepseek", "deepseek-chat")
print(f"上下文长度: {model_info.context_length}")
```

---

## 高级功能

### 中间件系统

使用中间件扩展引擎功能：

```python
from llm_engine import LLMEngine
from llm_engine.middleware import LoggingMiddleware, TimingMiddleware

# 创建带中间件的引擎
engine = LLMEngine(
    config,
    middleware=[
        LoggingMiddleware(level="INFO", log_content=True),
        TimingMiddleware(),
    ]
)
```

内置中间件：
- `LoggingMiddleware` - 请求/响应日志记录
- `TimingMiddleware` - 请求耗时统计
- `RetryMiddleware` - 自动重试机制
- `ContentFilterMiddleware` - 内容过滤转换
- `HeaderInjectionMiddleware` - 自定义请求头

### 智能缓存

两阶段缓存系统（精确匹配 + 语义相似度）：

```python
from llm_engine import LLMEngine
from llm_engine.caching import CachingMiddleware, CacheConfig

config = CacheConfig(
    enable_semantic=True,      # 启用语义缓存
    semantic_threshold=0.9,    # 相似度阈值
    ttl=3600,                  # 缓存过期时间（秒）
)

engine = LLMEngine(
    llm_config,
    middleware=[CachingMiddleware(config)]
)
```

### 可观测性

请求指标和成本跟踪：

```python
from llm_engine import LLMEngine
from llm_engine.observability import ObservabilityMiddleware

mw = ObservabilityMiddleware(budget_usd=100.0, alert_threshold=0.8)
engine = LLMEngine(config, middleware=[mw])

# 获取统计信息
stats = mw.get_statistics()
print(f"总成本: ${stats['total_cost_usd']:.2f}")
print(f"平均延迟: {stats['avg_latency_ms']:.0f}ms")
```

### 性能优化

速率限制和并发控制：

```python
from llm_engine import LLMEngine
from llm_engine.performance import PerformanceMiddleware

mw = PerformanceMiddleware(
    rate_limiting=True,
    max_concurrent=20,
)
engine = LLMEngine(config, middleware=[mw])
```

### 组合使用

所有功能可以组合使用：

```python
from llm_engine import LLMEngine
from llm_engine.middleware import LoggingMiddleware
from llm_engine.caching import CachingMiddleware, CacheConfig
from llm_engine.observability import ObservabilityMiddleware
from llm_engine.performance import PerformanceMiddleware

engine = LLMEngine(
    config,
    middleware=[
        LoggingMiddleware(),
        CachingMiddleware(CacheConfig(enable_semantic=True)),
        ObservabilityMiddleware(budget_usd=100.0),
        PerformanceMiddleware(max_concurrent=10),
    ]
)
```

### 自定义提供商

支持任何OpenAI兼容的API端点:

```python
from llm_engine import LLMConfig, LLMProvider
from llm_engine.providers.openai_compatible import OpenAICompatibleProvider

class CustomProvider(OpenAICompatibleProvider):
    def _get_env_api_key(self) -> Optional[str]:
        return os.getenv("CUSTOM_API_KEY")

    def _get_default_base_url(self) -> str:
        return "https://api.custom-llm.com/v1"

config = LLMConfig(
    provider=LLMProvider.CUSTOM,
    model_name="custom-model",
    base_url="https://api.custom-llm.com/v1",
)
```

### 错误处理

```python
from llm_engine.exceptions import LLMProviderError, LLMConfigError

try:
    response = await engine.generate("测试")
except LLMConfigError as e:
    print(f"配置错误: {e}")
except LLMProviderError as e:
    print(f"提供商API错误: {e}")
except Exception as e:
    print(f"其他错误: {e}")
```

### Token成本估算

```python
from llm_engine.config_loader import get_model_info

model_info = get_model_info("deepseek", "deepseek-chat")

# 估算输入token数
input_tokens = engine.estimate_tokens("用户输入文本", language="chinese")

# 计算预估成本（人民币）
input_cost = input_tokens * model_info.pricing.input / 1_000_000
output_cost = 4000 * model_info.pricing.output / 1_000_000  # 预估输出
print(f"预估成本: ¥{input_cost + output_cost:.4f}")
```

---

## 开发指南

### 运行测试

```bash
# 运行所有测试
pytest

# 运行测试并生成覆盖率报告
pytest --cov=llm_engine --cov-report=html

# 运行特定测试文件
pytest tests/test_engine.py

# 运行异步测试
pytest -v tests/test_async.py
```

### 代码质量检查

```bash
# 运行Ruff代码检查
ruff check .

# 运行Ruff格式化
ruff format .

# 运行类型检查
mypy llm_engine

# 运行安全扫描
bandit -r llm_engine
```

### 预提交钩子

```bash
# 安装预提交钩子
pre-commit install

# 手动运行所有钩子
pre-commit run --all-files
```

---

## 项目结构

```
llm-engine/
├── llm_engine/              # 主包目录
│   ├── __init__.py          # 包入口，导出主要类
│   ├── config.py            # 配置类定义 (LLMConfig, LLMProvider)
│   ├── config_loader.py     # 配置文件加载器
│   ├── engine.py            # LLMEngine主类和各提供商实现
│   ├── exceptions.py        # 自定义异常类
│   ├── factory.py           # 提供商工厂函数
│   ├── caching/             # 缓存系统
│   │   ├── cache.py         # 核心缓存实现
│   │   ├── backends.py      # 缓存后端（内存/磁盘/Redis）
│   │   ├── semantic.py      # 语义相似度缓存
│   │   └── middleware.py    # 缓存中间件
│   ├── middleware/          # 中间件框架
│   │   ├── base.py          # 中间件基类
│   │   ├── chain.py         # 中间件链
│   │   └── builtin.py       # 内置中间件
│   ├── observability/       # 可观测性
│   │   ├── metrics.py       # 指标收集
│   │   ├── cost_tracking.py # 成本跟踪
│   │   └── middleware.py    # 观测中间件
│   ├── performance/         # 性能优化
│   │   ├── rate_limiting.py # 速率限制
│   │   ├── connection_pool.py # 连接池
│   │   └── middleware.py    # 性能中间件
│   └── providers/           # 提供商实现
│       ├── base.py          # 基础提供商抽象类
│       └── openai_compatible.py  # OpenAI兼容提供商
├── tests/                   # 测试目录
├── providers.yml            # 提供商配置文件示例
├── pyproject.toml           # 项目配置和依赖
├── README.md                # 本文件
└── LICENSE                  # MIT许可证
```

---

## 许可证

本项目采用 [MIT License](LICENSE) 开源许可证。

---

## 贡献

欢迎提交Issue和Pull Request！

- 提交Bug报告: [GitHub Issues](https://github.com/houhuawei23/llm-engine/issues)
- 查看源码: [GitHub Repository](https://github.com/houhuawei23/llm-engine)

---

## 相关链接

- [PyPI项目页面](https://pypi.org/project/llm-api-engine/)
- [LiteLLM文档](https://docs.litellm.ai/) - 底层使用的统一LLM接口库
- [OpenAI API文档](https://platform.openai.com/docs)
- [DeepSeek API文档](https://platform.deepseek.com/)
