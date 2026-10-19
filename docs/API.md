# API behavior

The default model alias is `deepseek-v4.1`, with base URL `http://127.0.0.1:30011/v1`. The local service does not contact the OpenAI or Anthropic cloud. It implements the documented subset of their request shapes.

| Endpoint | Text | SSE | Image input | Automatic tool round trip |
|---|---|---|---|---|
| `/v1/chat/completions` | Checked | Checked | Checked | Checked |
| `/v1/responses` | Checked | Checked | Checked | Checked |
| `/v1/messages` | Checked | Checked | Checked | Checked |

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

Automatic function calls and client-supplied results are supported. Forced or named tool selection returns HTTP 400 with `unsupported_parameter` for `tool_choice`; it is not silently ignored. Stateful `previous_response_id` is unsupported. Send the needed history explicitly.

## Anthropic Messages

```bash
curl http://127.0.0.1:30011/v1/messages \
  -H 'Content-Type: application/json' -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"deepseek-v4.1","messages":[{"role":"user","content":"What is 6 times 7?"}],"max_tokens":128,"temperature":0,"thinking":{"type":"disabled"}}'
```

`thinking.type=disabled` is respected. Enabled thinking with a budget was also checked. Automatic tools work; forced selection is unsupported.

## Images and boundaries

Image data follows each endpoint's image-content format and requires the matching mmproj model. The regression used two fixtures across all three APIs and checked colors, shapes, left/right relations and text. Image preprocessing is capped at 1024 image tokens. Invalid image data is rejected.

The fixed adaptive starter is greedy-only. Use temperature 0; support for a generic API field does not mean every experimental runtime mode supports every sampling combination. The basic launcher has a different runtime profile.

Authentication is optional on loopback. A non-loopback bind requires `api_key`; clients then supply the configured key. Do not commit real API keys in configuration examples.

See the relevant [frontend](../serve/deepseek41_frontend.py), [engine protocol](../serve/deepseek41_engine.py) and [regression tests](../serve/test_deepseek41.py) for the implemented contracts.
