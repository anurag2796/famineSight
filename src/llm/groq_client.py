# src/llm/groq_client.py
import asyncio
import httpx
import json
import logging
from typing import AsyncGenerator, Dict, Any
from src.config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

class GroqClient:
    """
    Client for interacting with Groq LLM API.
    """

    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize Groq client.

        Args:
            api_key: Groq API key (defaults to config)
            model: Model name (defaults to config)
        """
        self.api_key = api_key or GROQ_API_KEY
        self.model = model or GROQ_MODEL
        self.base_url = "https://api.groq.com/openai/v1"

        if not self.api_key:
            raise ValueError("Groq API key is required")

        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            timeout=180.0  # 3 minute timeout
        )

    async def stream(self, prompt: str, context: list = None) -> AsyncGenerator[str, None]:
        """
        Stream response from Groq API.

        Args:
            prompt: Prompt to send to the model
            context: Optional conversation context

        Yields:
            Text chunks from the model response
        """
        try:
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]

            if context:
                # Add conversation history if provided
                messages = context + messages

            # Prepare payload
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "temperature": 0.7,
                "max_tokens": 2048
            }

            # Make API request
            async with self.client.stream("POST", "/chat/completions", json=payload) as response:
                if response.status_code != 200:
                    raise Exception(f"Groq API error: {response.status_code} - {response.text}")

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
                                # Skip malformed JSON
                                continue

        except Exception as e:
            logger.error(f"Error in Groq streaming: {e}")
            raise

    async def check_availability(self) -> bool:
        """
        Check if Groq API is available.

        Returns:
            True if available, False otherwise
        """
        try:
            # Simple ping to check API availability
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False
            }

            response = await self.client.post("/chat/completions", json=payload)
            return response.status_code == 200

        except Exception as e:
            logger.error(f"Error checking Groq availability: {e}")
            return False

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

# Global client instance
groq_client = None

def get_groq_client() -> GroqClient:
    """
    Get or create Groq client instance.

    Returns:
        GroqClient instance
    """
    global groq_client
    if groq_client is None:
        groq_client = GroqClient()
    return groq_client

if __name__ == "__main__":
    # Test the client
    logger.info("Testing Groq client...")
    print("Groq client initialized successfully")