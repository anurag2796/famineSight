# src/llm/client.py
import asyncio
import httpx
import json
import logging
from typing import AsyncGenerator, Dict, Any
from src.config import OLLAMA_HOST, OLLAMA_MODEL, GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

class OllamaClient:
    """
    Async client for interacting with Ollama LLM.
    """

    def __init__(self):
        self.client = httpx.AsyncClient(
            base_url=OLLAMA_HOST,
            timeout=180.0  # 3 minute timeout
        )
        self.model = OLLAMA_MODEL

    async def stream(self, prompt: str, context: list = None) -> AsyncGenerator[str, None]:
        """
        Stream response from Ollama.

        Args:
            prompt: Prompt to send to the model
            context: Optional context from previous conversation

        Yields:
            Text chunks from the model response
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": True
            }

            if context:
                payload["context"] = context

            async with self.client.stream("POST", "/api/generate", json=payload) as response:
                if response.status_code != 200:
                    raise Exception(f"Ollama error: {response.status_code}")

                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                        except json.JSONDecodeError:
                            # Skip malformed JSON
                            continue

        except Exception as e:
            logger.error(f"Error in Ollama streaming: {e}")
            raise

    async def check_availability(self) -> bool:
        """
        Check if Ollama is available and the model is loaded.

        Returns:
            True if available, False otherwise
        """
        try:
            # Check if Ollama is running
            response = await self.client.get("/api/tags")
            if response.status_code == 200:
                tags = response.json()
                # Check if our model is available
                model_available = any(
                    tag.get("name") == self.model or
                    tag.get("name", "").startswith(self.model.split(":")[0])
                    for tag in tags.get("models", [])
                )
                return model_available
            return False
        except Exception as e:
            logger.error(f"Error checking Ollama availability: {e}")
            return False

class HybridClient:
    """
    Hybrid client that can use either Ollama or Groq based on configuration.
    """

    def __init__(self):
        self.ollama_client = OllamaClient()
        self.groq_client = None
        self.use_groq = bool(GROQ_API_KEY and GROQ_API_KEY != "")

        if self.use_groq:
            try:
                from src.llm.groq_client import GroqClient
                self.groq_client = GroqClient()
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
                self.use_groq = False

    async def stream(self, prompt: str, context: list = None) -> AsyncGenerator[str, None]:
        """
        Stream response from either Groq or Ollama.

        Args:
            prompt: Prompt to send to the model
            context: Optional conversation context

        Yields:
            Text chunks from the model response
        """
        if self.use_groq and self.groq_client:
            try:
                # Try Groq first
                async for chunk in self.groq_client.stream(prompt, context):
                    yield chunk
            except Exception as e:
                logger.warning(f"Groq failed, falling back to Ollama: {e}")
                # Fall back to Ollama
                async for chunk in self.ollama_client.stream(prompt, context):
                    yield chunk
        else:
            # Use Ollama only
            async for chunk in self.ollama_client.stream(prompt, context):
                yield chunk

    async def check_availability(self) -> bool:
        """
        Check availability of either client.

        Returns:
            True if at least one client is available, False otherwise
        """
        try:
            if self.use_groq and self.groq_client:
                # For Groq, we can't easily check without making a request
                # Just assume it's available if configured
                return True
            else:
                return await self.ollama_client.check_availability()
        except Exception as e:
            logger.error(f"Error checking hybrid availability: {e}")
            return False

# Global client instance
hybrid_client = HybridClient()

# For backward compatibility, also expose the ollama client directly
ollama_client = hybrid_client.ollama_client

if __name__ == "__main__":
    # Test the client
    logger.info("Testing hybrid client...")
    print("Hybrid client initialized successfully")