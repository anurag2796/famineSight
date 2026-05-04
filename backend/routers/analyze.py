# backend/routers/analyze.py
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import logging
from backend.schemas.output import ClusterProfile
from backend.services.model_registry import registry

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/rules", response_model=dict)
async def get_rules(algorithm: Optional[str] = Query(None)):
    """
    Get association rules from analysis.

    Args:
        algorithm: Algorithm to use (fpgrowth or apriori)

    Returns:
        Association rules
    """
    try:
        # Return rules from registry
        rules = registry.association_results

        # Filter by algorithm if specified
        if algorithm and algorithm in rules:
            return {algorithm: rules[algorithm]}
        else:
            return rules

    except Exception as e:
        logger.error(f"Error getting rules: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get rules: {str(e)}")

@router.get("/clusters", response_model=List[ClusterProfile])
async def get_clusters():
    """
    Get cluster profiles.

    Returns:
        List of cluster profiles
    """
    try:
        # Return cluster profiles from registry
        clusters = registry.cluster_results

        # Convert to proper response format
        profiles = []
        if 'district_profiles' in clusters:
            for profile in clusters['district_profiles']:
                profiles.append(ClusterProfile(
                    pcode=profile.get('pcode', ''),
                    district=profile.get('district', ''),
                    kmeans_cluster=profile.get('kmeans_cluster', 0),
                    cluster_name=profile.get('cluster_name', ''),
                    features={k: v for k, v in profile.items() if k not in ['pcode', 'district', 'kmeans_cluster', 'cluster_name']}
                ))

        return profiles

    except Exception as e:
        logger.error(f"Error getting clusters: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get clusters: {str(e)}")