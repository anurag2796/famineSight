# backend/routers/anomaly.py
from fastapi import APIRouter, HTTPException
from typing import List
import logging
from backend.services.model_registry import registry

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/alerts", response_model=list)
async def get_alerts():
    """
    Get anomaly alerts.

    Returns:
        List of alerts
    """
    try:
        # Return alerts from registry
        alerts = registry.anomaly_results.get('alerts', [])
        return alerts

    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")