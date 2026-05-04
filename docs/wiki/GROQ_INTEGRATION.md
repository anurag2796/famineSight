# Groq Integration Guide

## Overview

FamineSight uses a **hybrid LLM architecture** that can leverage either a local Ollama instance (primary, production) or the Groq cloud API (optional, development/fallback). This document covers the Groq integration specifically.

## Architecture

```
src/llm/client.py          ← HybridClient (primary interface)
  ├── groq_client.py       ← Groq cloud API (OpenAI-compatible)
  ├── guardrails.py        ← Output validation
  └── prompts.py           ← Prompt templates
```

### Priority Order

| Priority | Backend | Used When |
|----------|---------|-----------|
| 1st | **Groq** | `GROQ_API_KEY` is set and Groq is reachable |
| 2nd | **Ollama** | Groq unavailable or key not set |
| Error | — | Both unavailable |

## Why Groq?

- **Speed** — Ultra-fast inference (>1000 tokens/sec on Groq hardware)
- **Cost** — Free tier available for development
- **OpenAI-compatible** — Same API shape as OpenAI, easy to integrate
- **No local RAM required** — Useful when Ollama model (~20 GB) isn't available

**Limitations:**
- Requires internet connectivity
- Free tier has rate limits
- Cloud processing — avoid sending sensitive humanitarian data in production

## Configuration

### Environment Variables

```bash
# .env
GROQ_API_KEY=your_groq_api_key_here   # Leave empty to disable Groq
GROQ_MODEL=llama3-8b-8192             # Or: llama3-70b-8192, mixtral-8x7b-32768
```

Get a free API key at: https://console.groq.com/

### Available Models

| Model | Context | Speed | Best For |
|-------|---------|-------|----------|
| `llama3-8b-8192` | 8192 tokens | Fastest | Development, quick tests |
| `llama3-70b-8192` | 8192 tokens | Fast | Higher quality responses |
| `mixtral-8x7b-32768` | 32768 tokens | Fast | Long documents |
| `llama-3.3-70b-versatile` | 128k tokens | Fast | Complex analysis |

## Implementation

### `src/llm/groq_client.py`

```python
# Groq API endpoint (OpenAI-compatible)
BASE_URL = "https://api.groq.com/openai/v1"

class GroqClient:
    def __init__(self, api_key: str, model: str = "llama3-8b-8192"):
        self.api_key = api_key
        self.model = model
        self.client = httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {api_key}"}
        )

    async def stream(self, prompt: str) -> AsyncGenerator[str, None]:
        # SSE streaming over OpenAI /chat/completions
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        }
        async with self.client.stream("POST", "/chat/completions", json=payload) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunk = json.loads(line[6:])
                    content = chunk["choices"][0]["delta"].get("content", "")
                    if content:
                        yield content
```

### `src/llm/client.py` — HybridClient

```python
class HybridClient:
    def __init__(self):
        self.groq = GroqClient(GROQ_API_KEY, GROQ_MODEL) if GROQ_API_KEY else None
        # OllamaClient communicates with local Ollama via OLLAMA_HOST

    async def stream(self, prompt: str) -> AsyncGenerator[str, None]:
        if self.groq:
            try:
                async for chunk in self.groq.stream(prompt):
                    yield chunk
                return
            except Exception as e:
                logger.warning(f"Groq failed, falling back to Ollama: {e}")
        # Fallback to Ollama
        async for chunk in self.ollama.stream(prompt):
            yield chunk

hybrid_client = HybridClient()
```

### Usage Example

```python
from src.llm.client import hybrid_client

async def generate_narrative(district: str, risk_score: float) -> str:
    prompt = f"Generate a humanitarian situation report for {district} with risk score {risk_score:.2f}."
    full_response = ""
    async for chunk in hybrid_client.stream(prompt):
        full_response += chunk
    return full_response
```

## Development vs Production

| Environment | `GROQ_API_KEY` | `OLLAMA_MODEL` | Behavior |
|-------------|---------------|----------------|---------|
| Development | Set | `mistral:7b` (small) | Groq primary, fast iteration |
| Hybrid | Set | `qwen3:32b` | Groq primary, Ollama fallback |
| Production | Empty | `qwen3:32b` | Ollama only, fully offline |

### Development Setup

```bash
# Use Groq + small local model for development
echo "GROQ_API_KEY=your_key" >> .env
echo "GROQ_MODEL=llama3-8b-8192" >> .env
echo "OLLAMA_MODEL=mistral:7b" >> .env
```

### Production Setup

```bash
# Offline mode — Ollama only
echo "GROQ_API_KEY=" >> .env          # Empty = disabled
echo "OLLAMA_MODEL=qwen3:32b" >> .env
```

## Security Considerations

- **API Key** — Store only in `.env` with `chmod 600 .env`, never commit to git
- **Data Privacy** — Groq is a cloud API; avoid sending personally identifiable or sensitive humanitarian data in production
- **Rate Limits** — Free tier: ~100 RPM. The HybridClient falls back to Ollama on `429` errors
- **Production** — Use Ollama only for fully offline, on-premise processing

## Testing

```bash
# Quick integration test
python -c "
import asyncio
from src.llm.client import hybrid_client

async def test():
    result = ''
    async for chunk in hybrid_client.stream('Say hello in one sentence.'):
        result += chunk
    print('Response:', result)

asyncio.run(test())
"
```

## Guardrails

All LLM output (from both Groq and Ollama) passes through `src/llm/guardrails.py` before being returned by the API:

- Rejects responses that combine low-risk language with famine terminology
- Flags probability estimate mismatches against model predictions
- Appends a mandatory verification disclaimer

## Troubleshooting

| Issue | Resolution |
|-------|-----------|
| Groq returns `401` | Check `GROQ_API_KEY` in `.env`; verify key at console.groq.com |
| Groq returns `429` | Rate limit exceeded; HybridClient falls back to Ollama |
| Both Groq and Ollama fail | `/narrative/generate` returns a 503 error |
| Slow responses | Switch to a smaller Groq model (`llama3-8b-8192`) |
| Guardrails reject output | Check guardrails.py; adjust prompt or thresholds |

## Cost Reference

| Model | Input | Output |
|-------|-------|--------|
| `llama3-8b-8192` | ~$0.05 / 1M tokens | ~$0.08 / 1M tokens |
| `llama3-70b-8192` | ~$0.59 / 1M tokens | ~$0.79 / 1M tokens |
| Free tier | ~100 RPM, 6000 RPD | — |

*Rates as of 2025. Check https://groq.com/pricing for current pricing.*