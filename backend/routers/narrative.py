# backend/routers/narrative.py
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator, Dict, Any
import json
import logging
from src.llm.client import ollama_client
from src.llm.prompts import SYSTEM_PROMPT, build_prompt
from src.llm.guardrails import validate_narrative
from backend.schemas.input import NarrativeRequest
from backend.services.model_registry import registry

logger = logging.getLogger(__name__)

router = APIRouter()

async def generate_narrative_stream(
    prediction: Dict[str, Any],
    alerts: list,
    rules: Dict[str, Any]
) -> AsyncGenerator[str, None]:
    """
    Generate narrative stream from LLM.

    Args:
        prediction: Prediction results
        alerts: List of alerts
        rules: Association rules

    Yields:
        Stream of narrative text
    """
    try:
        # Build prompt
        prompt = build_prompt(prediction, alerts, rules)

        # Stream response from Ollama
        async for chunk in ollama_client.stream(prompt):
            yield chunk

    except Exception as e:
        logger.error(f"Error in narrative generation: {e}")
        yield f"Error: {str(e)}"

@router.post("/generate")
async def generate_narrative(request: NarrativeRequest):
    """
    Generate situation narrative.

    Args:
        request: Narrative generation request

    Returns:
        Streaming response with narrative
    """
    try:
        # Get the data from registry
        alerts = registry.anomaly_results.get('alerts', [])
        rules = registry.association_results

        # Generate narrative stream
        stream = generate_narrative_stream(
            request.prediction.model_dump(),
            alerts,
            rules
        )

        # Return streaming response
        return StreamingResponse(stream, media_type="text/plain")

    except Exception as e:
        logger.error(f"Error generating narrative: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate narrative: {str(e)}")