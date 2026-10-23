# API usage

The default model alias is `deepseek-v4.1`, with base URL `http://127.0.0.1:30011/v1`. OpenAI Chat Completions / Responses and Anthropic Messages are supported.

| Endpoint | Text | SSE | Image input | Automatic tool round trip |
|---|---|---|---|---|
| `/v1/chat/completions` | Supported | Supported | Supported | Supported |
| `/v1/responses` | Supported | Supported | Supported | Supported |
| `/v1/messages` | Supported | Supported | Supported | Supported |

## Chat Completions

```bash
curl http://127.0.0.1:30011/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1","messages":[{"role":"user","content":"What is 6 times 7?"}],"max_tokens":128,"temperature":0,"reasoning_effort":"none","stream":true}'
```

Standard OpenAI tool definitions use `{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{}}}}`. Tool selection is automatic; the client executes a selected tool and sends its result back. The server itself does not execute tools.

## Responses

```bash
curl http://127.0.0.1:30011/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1","input":"What is 6 times 7?","max_output_tokens":128,"temperature":0,"reasoning":{"effort":"none"}}'
```

Automatic function calls and client-supplied results are supported. Include conversation history in the request.

## Anthropic Messages

```bash
curl http://127.0.0.1:30011/v1/messages \
  -H 'Content-Type: application/json' -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"deepseek-v4.1","messages":[{"role":"user","content":"What is 6 times 7?"}],"max_tokens":128,"temperature":0,"thinking":{"type":"disabled"}}'
```

Use `thinking.type=disabled` to disable thinking, or enable thinking with a budget. Automatic tool selection is supported.

## Images

Image input uses each endpoint's image-content format and requires the `mmproj` model path in the configuration.

Authentication is optional on loopback. A non-loopback bind requires `api_key`; clients then supply the configured key. Do not commit real API keys in configuration examples.
