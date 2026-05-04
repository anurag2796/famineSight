# Groq Integration Guide

## Overview

This document outlines the approach for integrating Groq's free LLM API into the FamineSight system. Groq offers high-performance inference with competitive pricing and free access for development.

## Why Groq Integration?

### Benefits
- **Performance**: Ultra-fast inference (1000+ tokens/sec)
- **Cost-effective**: Free tier available for development
- **Quality**: High-quality language models
- **Scalability**: Cloud-based with automatic scaling

### Limitations
- **API Dependencies**: Requires internet connectivity
- **Rate Limits**: Free tier has usage limits
- **Data Privacy**: Cloud-based processing may raise concerns
- **Authentication**: Requires API key management

## Groq API Integration Approach

### 1. API Structure
Groq uses the OpenAI-compatible API structure:

```python
# Groq API endpoint
https://api.groq.com/openai/v1/chat/completions

# Required headers
{
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
}

# Request format
{
    "model": "llama3-8b-8192",  # or other Groq models
    "messages": [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello!"}
    ],
    "temperature": 0.7,
    "max_tokens": 1000
}
```

### 2. Integration Strategy

#### A. Hybrid Approach (Recommended)
- **Primary**: Use local Ollama for production (Jetson constraints)
- **Secondary**: Use Groq for development/testing
- **Fallback**: Local model when Groq unavailable

#### B. Environment-Based Switching
```python
# In src/llm/client.py
import os

class GroqClient:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.base_url = "https://api.groq.com/openai/v1"
        
    async def stream(self, prompt: str):
        # Groq implementation using OpenAI-compatible API
        pass
```

### 3. Implementation Steps

#### Step 1: Environment Setup
```bash
# Add to .env file
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama3-8b-8192
```

#### Step 2: Client Implementation
```python
# src/llm/groq_client.py
import httpx
import json
import asyncio
from typing import AsyncGenerator

class GroqClient:
    def __init__(self, api_key: str, model: str = "llama3-8b-8192"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1"
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"}
        )

    async def stream(self, prompt: str) -> AsyncGenerator[str, None]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        }
        
        async with self.client.stream("POST", "/chat/completions", json=payload) as response:
            async for line in response.aiter_lines():
                if line.strip() and line.startswith('data: '):
                    data = line[6:]  # Remove 'data: ' prefix
                    if data != '[DONE]':
                        try:
                            message = json.loads(data)
                            content = message.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
```

#### Step 3: Model Selection
Available Groq models:
- `llama3-8b-8192` - 8B parameter model (recommended for development)
- `llama3-70b-8192` - 70B parameter model (higher quality)
- `mixtral-8x7b-32768` - Mixture-of-Experts model
- `gemma-7b-it` - Google's efficient model

### 4. Configuration Management

#### Environment Variables
```bash
# .env file
LLM_PROVIDER=ollama        # or "groq"
OLLAMA_HOST=http://host.docker.internal:11434
OLLAMA_MODEL=mistral:7b
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama3-8b-8192
```

#### Client Switching Logic
```python
# src/llm/client.py
from src.config import LLM_PROVIDER, OLLAMA_HOST, OLLAMA_MODEL, GROQ_API_KEY, GROQ_MODEL

def get_llm_client():
    if LLM_PROVIDER == "groq" and GROQ_API_KEY:
        return GroqClient(GROQ_API_KEY, GROQ_MODEL)
    else:
        return OllamaClient(OLLAMA_HOST, OLLAMA_MODEL)
```

## Security Considerations

### 1. API Key Management
- Store keys in environment variables only
- Never commit keys to version control
- Use `.env` with proper permissions (600)
- Rotate keys regularly

### 2. Data Privacy
- **Local Processing**: Primary use should be local (Jetson constraints)
- **Cloud Processing**: Only for development/testing
- **Data Minimization**: Send only necessary context
- **Compliance**: Ensure compliance with humanitarian data regulations

### 3. Rate Limiting
- Monitor usage to avoid exceeding free tier limits
- Implement retry logic with backoff
- Cache responses when possible

## Testing Strategy

### 1. Development Testing
```python
# test_groq_integration.py
from src.llm.groq_client import GroqClient

async def test_groq():
    client = GroqClient("your_api_key", "llama3-8b-8192")
    async for chunk in client.stream("Hello, how are you?"):
        print(chunk, end="", flush=True)
    print("\n✅ Groq integration test passed")

if __name__ == "__main__":
    asyncio.run(test_groq())
```

### 2. Fallback Strategy
```python
# Enhanced client with fallback
class HybridClient:
    def __init__(self):
        self.ollama_client = OllamaClient()
        self.groq_client = GroqClient()
        self.use_groq = bool(os.getenv("GROQ_API_KEY"))
    
    async def generate(self, prompt: str):
        if self.use_groq:
            try:
                async for chunk in self.groq_client.stream(prompt):
                    yield chunk
            except Exception:
                # Fallback to local model
                async for chunk in self.ollama_client.stream(prompt):
                    yield chunk
        else:
            async for chunk in self.ollama_client.stream(prompt):
                yield chunk
```

## Production Deployment

### 1. Deployment Configuration
```yaml
# docker-compose.yml
services:
  backend:
    environment:
      - LLM_PROVIDER=ollama  # Production default
      - GROQ_API_KEY=        # Empty for production
    # ... other settings
```

### 2. Development Configuration
```yaml
# docker-compose.dev.yml  
services:
  backend:
    environment:
      - LLM_PROVIDER=groq    # Development override
      - GROQ_API_KEY=your_key
    # ... other settings
```

## Cost Considerations

### Free Tier Limits
- **llama3-8b-8192**: 100,000 tokens/month
- **llama3-70b-8192**: 50,000 tokens/month  
- **mixtral-8x7b-32768**: 100,000 tokens/month

### Production Cost
- **Local**: Zero cost (Jetson resources)
- **Cloud**: $0.0001-0.0005 per token (varies by model)
- **Hybrid**: Optimal balance of cost and performance

## Best Practices

### 1. Development Workflow
1. **Use Groq for development** - Faster iteration
2. **Use local Ollama for production** - Secure, offline processing
3. **Maintain same interface** - Same API for both clients

### 2. Error Handling
```python
try:
    response = await client.stream(prompt)
    # Process response
except httpx.RequestError as e:
    # Handle network issues
    logger.warning(f"Groq API error: {e}")
    # Fallback to local processing
except Exception as e:
    # Handle other errors
    logger.error(f"Unexpected error: {e}")
```

### 3. Monitoring
- Track API usage
- Monitor response times
- Log fallback events
- Performance comparison between models

## Migration Path

### Phase 1: Development
- Configure Groq integration
- Test with development data
- Validate output quality

### Phase 2: Hybrid
- Use Groq for development
- Use local for production
- Monitor performance differences

### Phase 3: Production
- Switch to local processing (Jetson constraints)
- Use Groq only for testing/evaluation
- Maintain both implementations for flexibility

## Conclusion

The Groq integration approach provides flexibility for development while maintaining the secure, local processing required for production deployment on Jetson platforms. The hybrid approach ensures that development teams can leverage Groq's performance benefits while production systems maintain the security and offline capabilities needed for humanitarian applications.