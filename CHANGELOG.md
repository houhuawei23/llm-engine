# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-04-17

### Fixed

- **DeepSeekProvider JSON detection**: Fix incorrect `response_format` injection when prompts contain the word "JSON" in a negative context (e.g., "do NOT output as JSON"). Previously used a naive `"json" in content.lower()` check which matched any mention of JSON, causing DeepSeek Reasoner models to emit raw JSON with chain-of-thought text instead of Markdown. Now uses explicit pattern matching that distinguishes between positive JSON requests ("return json") and negative mentions ("不允许直接以整段 JSON"), satisfying both the DeepSeek API requirement and user intent.

### Contributors

- Fix designed and implemented with assistance from Claude Code (Anthropic).

## [0.2.0] - 2026-04-06

### Added

#### Middleware Framework
- **New module**: `llm_engine.middleware`
- `Middleware` abstract base class for request/response processing
- `MiddlewareChain` for sequential middleware execution
- `RequestContext` and `Response` dataclasses
- Built-in middleware:
  - `LoggingMiddleware` - configurable request/response logging
  - `TimingMiddleware` - latency measurement
  - `RetryMiddleware` - transient error handling
  - `ContentFilterMiddleware` - content transformation
  - `HeaderInjectionMiddleware` - custom header injection

#### Caching System
- **New module**: `llm_engine.caching`
- Two-tier caching (exact match + semantic similarity)
- Multiple cache backends:
  - `MemoryCacheBackend` - in-memory LRU cache
  - `DiskCacheBackend` - persistent file-based cache
  - `RedisCacheBackend` - Redis distributed cache
- `LLMCache` - unified caching interface
- `SemanticCache` - embedding-based similarity caching
- `SimpleEmbedder` - lightweight n-gram embedder (no external deps)
- `CachingMiddleware` - seamless middleware integration

#### Observability & Metrics
- **New module**: `llm_engine.observability`
- `RequestMetrics` - per-request metrics (latency, tokens, cost)
- `InMemoryMetricsCollector` - local metrics storage
- `PrometheusMetricsCollector` - Prometheus integration
- `CostTracker` - cost tracking with budget alerts
- `PricingProvider` - built-in pricing for major providers
- `ObservabilityMiddleware` - middleware integration

#### Performance Optimizations
- **New module**: `llm_engine.performance`
- `TokenBucketRateLimiter` - token bucket rate limiting
- `RateLimitManager` - provider-aware rate limit management
- Provider-specific rate limits (OpenAI, DeepSeek, Anthropic, Ollama)
- `ConcurrencyMiddleware` - async semaphore concurrency control
- `PerformanceMiddleware` - combined optimizations
- `ConnectionPool` - HTTP connection pooling

### Changed
- Updated `LLMEngine` to support middleware chain
- Enhanced `__init__.py` exports for new modules
- Improved test coverage (283 tests, 76% coverage)

## [0.1.5] - Previous

### Added
- Initial release
- Multi-provider support (OpenAI, DeepSeek, Anthropic, Ollama, Kimi)
- Async/sync dual API
- YAML configuration management
- Automatic retry mechanism
- Streaming output support
- Token estimation
- Pydantic-based configuration

[0.2.0]: https://github.com/houhuawei23/llm-engine/compare/v0.1.5...v0.2.0
