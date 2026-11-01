# DeepSeek 双协议 LLM Gateway

一个训练用、可运行的 FastAPI 网关示例：对调用方统一提供 OpenAI Chat Completions 接口，按请求的 `model` 别名路由到不同协议的供应商适配器：

| 对外 `model` | 上游协议 | 默认适配器 | 上游路径 |
|---|---|---|---|
| `deepseek-v4-pro` | OpenAI Responses API | `openai_responses` | `/responses` |
| `deepseek-v4-flash` | Anthropic Messages API | `anthropic_messages` | `/anthropic/v1/messages` |

DeepSeek 官方支持两种 API 格式。Responses 的 base URL 为 `https://api.deepseek.com`，其 endpoint 是 `/responses`；Anthropic 格式 base URL 为 `https://api.deepseek.com/anthropic`，之后请求 `/v1/messages`。示例中 `deepseek-v4-flash` 是网关别名，映射到当前模型 ID `deepseek-flash`；官方说明旧模型 ID `deepseek-v4-flash` 虽仍接受但已退役。自动化验证通过 Mock HTTP 服务检查两套协议适配，不会消耗模型额度。

- [DeepSeek Responses API 指南](https://api-docs.deepseek.com/guides/responses_api)
- [DeepSeek Anthropic API 指南](https://api-docs.deepseek.com/guides/anthropic_api)
- [DeepSeek 模型与价格](https://api-docs.deepseek.com/quick_start/pricing)

架构职责与设计选择见 [设计文档](./docs/architecture.md)。

## 功能点

- **适配器模式**：统一入口按 `model` 别名选上游目标，适配器分别转换 Responses 和 Anthropic Messages 请求、鉴权及响应格式。
- **流式输出**：支持 `stream: true`，把两种上游 SSE 转成 OpenAI Chat Completions SSE。
- **结构化输出**：接受 `response_format.type=json_schema`，将约束传给或提示上游，并由 Gateway 本地解析与 Schema 校验；失败时可发起一次修复调用。
- **Prompt 版本管理**：SQLite 存模板版本，支持激活版本、显式版本和 Jinja 变量。
- **可观测性**：SQLite 记录总输入、未缓存输入、缓存读取、缓存创建、输出、推理 Token、总延迟和首 Token 延迟。
- **韧性**：统一 OpenAI 风格错误；指数退避，单请求上游尝试次数最多 3；按调用身份和模型别名分别限流，超限返回 429。

## 启动

需要 Python 3.10+ 和 uv：

```bash
cd /Users/hanxiangrui/Documents/agentcode/llmgateway
cp gateway.example.yaml gateway.yaml
uv sync --extra dev

export GATEWAY_API_KEY='replace-with-a-long-random-secret'
export DEEPSEEK_API_KEY='your-provider-api-key'
# 如果供应商的协议端点不是默认值，按实际地址设置：
export DEEPSEEK_RESPONSES_BASE_URL='https://your-responses-endpoint'
export DEEPSEEK_ANTHROPIC_BASE_URL='https://your-messages-endpoint'
GATEWAY_CONFIG=gateway.yaml uv run uvicorn app.main:app --reload --port 8000
```

不要提交真实密钥。通过 `GET /healthz` 检查进程，通过 `GET /readyz` 检查应用初始化状态。

## 调用示例

### Responses API 路由：V4 Pro

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $GATEWAY_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-pro",
    "messages": [{"role": "user", "content": "用一句话介绍你自己"}]
  }'
```

### Anthropic Messages API 路由：V4 Flash

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $GATEWAY_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "用一句话介绍你自己"}]
  }'
```

### 流式

在上述任一 body 中增加 `"stream": true`：

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $GATEWAY_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash",
    "stream": true,
    "messages": [{"role": "user", "content": "逐步解释 SSE"}]
  }'
```

网关统一输出 OpenAI Chat SSE。上游首个文本块到达前可以重试或 fallback；输出已开始后不跨模型续写。

### 结构化输出

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $GATEWAY_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-pro",
    "messages": [{"role": "user", "content": "给 Ada 设定年龄 36"}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "person",
        "strict": true,
        "schema": {
          "type": "object",
          "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
          "required": ["name", "age"],
          "additionalProperties": false
        }
      }
    }
  }'
```

网关会在本地再次校验，不依赖模型“保证”输出格式。Anthropic Messages adapter 将 Schema 作为系统约束文本，并依靠相同的本地校验与修复路径。

### Prompt 版本管理

创建一个激活版本：

```bash
curl http://localhost:8000/v1/prompts \
  -H "Authorization: Bearer $GATEWAY_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
    "id": "assistant",
    "name": "助手",
    "content": "你是一个 {{ language }} 助手，请简洁回答。",
    "role": "system",
    "activate": true
  }'
```

在推理请求中引用当前激活版本，或附带 `"version": 1` 固定版本：

```json
{
  "model": "deepseek-v4-pro",
  "messages": [{"role": "user", "content": "你好"}],
  "gateway_prompt": {
    "id": "assistant",
    "version": 1,
    "variables": {"language": "中文"}
  }
}
```

### 可观测数据

```bash
curl 'http://localhost:8000/admin/usage?limit=20' \
  -H "Authorization: Bearer $GATEWAY_API_KEY"
```

记录包含请求 ID、别名、实际供应商/模型、输入 Token 分类、输出与推理 Token、延迟、TTFT、重试、fallback 和错误。`/admin/routes` 可查看模型和熔断状态。

## 验证

```bash
cd /Users/hanxiangrui/Documents/agentcode/llmgateway
uv run --extra dev python scripts/verify_gateway.py
```

脚本执行自动化验证，使用 HTTP Mock 覆盖：

1. `deepseek-v4-pro` 确实调用 `/responses`，并正确转换请求、响应和认证。
2. `deepseek-v4-flash` 确实调用 `/anthropic/v1/messages`，并正确转换 Anthropic 请求、响应和认证。
3. 两协议的 SSE 均变成统一 Chat SSE。
4. JSON Schema 校验、Prompt 版本引用、Token 分类/延迟记录。
5. 指数退避重试次数上限、按模型独立限流和 429 错误格式。

Mock 验证证明 Gateway 协议和功能链路；真实供应商调用还需要有效密钥、已开通模型，以及配置正确的 base URL。

配置好 `gateway.yaml` 和真实 API Key 后，可额外对两个模型各发起一次最小请求（会消耗真实 API 额度）：

```bash
uv run --extra dev python scripts/verify_gateway.py --live
```

## 注意事项

- 限流和熔断状态保存在单进程内存；多副本部署需使用 Redis 等共享状态。
- SQLite 用于训练和轻量部署；高并发生产环境需替换为合适的数据库/事件存储。
- 流式检查点默认关闭，开启前需要制定正文加密、访问控制和保留期限。
- 配置价格是成本估算，不代表供应商账单。
