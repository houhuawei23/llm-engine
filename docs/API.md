我来为你系统梳理 LLM API 调用的主流格式和 Python 实现方式。

---

## 一、主流 API 格式分类

| 类别 | 代表 | 特点 | 适用场景 |
|:---|:---|:---|:---|
| **OpenAI 兼容格式** | OpenAI、Azure、DeepSeek、Qwen、LLaMA API 等 | 事实标准，生态最全 | 通用首选 |
| **Anthropic 格式** | Claude API | 长上下文优化，独特系统提示 | Claude 专用 |
| **Google 格式** | Gemini API | 原生多模态，安全过滤 | Gemini/Gemma |
| **本地/开源格式** | Ollama、vLLM、TGI、llama.cpp | 本地部署，OpenAI 兼容或自定义 | 私有化部署 |
| **云服务专用** | AWS Bedrock、Vertex AI、千帆/灵积 | 平台封装，多模型统一接口 | 企业云集成 |

---

## 二、详细格式与 Python 调用

### 1. OpenAI 兼容格式（最常用）

**请求格式**：
```json
{
  "model": "gpt-4",
  "messages": [
    {"role": "system", "content": "你是助手"},
    {"role": "user", "content": "你好"}
  ],
  "temperature": 0.7,
  "max_tokens": 512,
  "stream": false
}
```

**Python 调用**：
```python
from openai import OpenAI

# 标准 OpenAI
client = OpenAI(api_key="sk-...")

# 兼容其他服务（如 DeepSeek、本地 vLLM）
client = OpenAI(
    api_key="sk-...",
    base_url="https://api.deepseek.com/v1"  # 或 http://localhost:8000/v1
)

response = client.chat.completions.create(
    model="deepseek-chat",  # 或 "gpt-4", "Qwen/Qwen2-72B"
    messages=[
        {"role": "system", "content": "你是专业助手"},
        {"role": "user", "content": "解释量子计算"}
    ],
    temperature=0.7,
    max_tokens=1024,
    stream=False
)

print(response.choices[0].message.content)
```

**流式输出**：
```python
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "讲个故事"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

---

### 2. Anthropic Claude 格式

**特点**：`system` 单独字段，支持 `thinking` 模式

```python
import anthropic

client = anthropic.Anthropic(api_key="sk-ant-...")

message = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=4096,
    system="你是编程专家",  # 独立字段，非 messages 中
    messages=[
        {"role": "user", "content": "优化这段 Python 代码"}
    ],
    thinking={  # 扩展思考模式（可选）
        "type": "enabled",
        "budget_tokens": 2000
    }
)

print(message.content[0].text)
```

---

### 3. Google Gemini 格式

**特点**：原生多模态，内容分 `parts`

```python
import google.generativeai as genai

genai.configure(api_key="...")

model = genai.GenerativeModel('gemini-2.0-flash')

# 文本
response = model.generate_content("解释神经网络")

# 多模态（图片+文本）
import PIL.Image
img = PIL.Image.open('chart.png')
response = model.generate_content(["分析这张图表", img])

print(response.text)
```

**REST API 格式**：
```json
{
  "contents": [{
    "role": "user",
    "parts": [
      {"text": "描述这张图片"},
      {"inline_data": {"mime_type": "image/jpeg", "data": "base64..."}}
    ]
  }]
}
```

---

### 4. 本地部署格式（Ollama / vLLM）

**Ollama**（极简本地运行）：
```python
import ollama

response = ollama.chat(
    model='llama3.2',
    messages=[{'role': 'user', 'content': '你好'}]
)
print(response['message']['content'])

# 或 REST API
import requests
r = requests.post('http://localhost:11434/api/chat', json={
    "model": "llama3.2",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": False
})
```

**vLLM / TGI**（生产级本地服务）：
```python
# 与 OpenAI 完全兼容
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
response = client.chat.completions.create(model="meta-llama/Llama-2-7b", ...)
```

---

### 5. 统一封装：使用 LiteLLM

**一行代码切换任意后端**：
```python
from litellm import completion

# 自动路由到对应格式
response = completion(
    model="gpt-4",           # OpenAI
    # model="claude-3-opus", # Anthropic
    # model="gemini/gemini-pro", # Google
    # model="ollama/llama3.2",   # Ollama
    messages=[{"role": "user", "content": "你好"}],
    api_key="..."
)
```

---

## 三、格式对比表

| 特性 | OpenAI | Anthropic | Gemini | Ollama |
|:---|:---|:---|:---|:---|
| **消息字段** | `messages` | `messages` | `contents` | `messages` |
| **系统提示** | `messages[0].role="system"` | 独立 `system` 参数 | `system_instruction` | `messages` 中 |
| **多模态** | 需 GPT-4V 格式 | 支持 | 原生支持 | 依赖模型 |
| **流式** | `stream=True` | `stream=True` | 支持 | `stream=True` |
| **工具调用** | `tools` 参数 | `tools` 参数 | `tools` 声明 | 部分支持 |
| **Python SDK** | `openai` | `anthropic` | `google-generativeai` | `ollama` |

---

## 四、推荐选型

| 场景 | 推荐格式/库 |
|:---|:---|
| 快速原型/多后端切换 | **LiteLLM** |
| 生产环境标准化 | **OpenAI 兼容格式** + `openai` SDK |
| Claude 专属功能（长文本/思考） | **Anthropic SDK** |
| 私有化部署 | **vLLM** / **Ollama**（OpenAI 兼容） |
| 多模态（图文音视频） | **Gemini API** |
| 企业云（AWS/阿里/腾讯） | 各自 SDK 或 LiteLLM 适配 |

---

需要我针对特定场景（如多轮对话、工具调用、批量推理）给出完整代码示例吗？