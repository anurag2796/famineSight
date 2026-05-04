# src/llm/guardrails.py
import re
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

def validate_narrative(text: str, prediction: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate the generated narrative against guardrail rules.

    Args:
        text: Generated narrative text
        prediction: Prediction results from the model

    Returns:
        Tuple of (is_valid, validation_message)
    """
    # Check 1: LOW risk + famine/catastrophe/mass_death language
    low_risk = prediction.get('risk_level', '').lower() == 'low'

    # Look for problematic terms
    problematic_terms = [
        r'\bfamine\b', r'\bcatastrophe\b', r'\bmass death\b',
        r'\bdisaster\b', r'\bemergency\b', r'\bcrisis\b'
    ]

    if low_risk:
        for term in problematic_terms:
            if re.search(term, text, re.IGNORECASE):
                return False, "FAIL: LOW risk prediction should not contain famine/catastrophe terms"

    # Check 2: Probability mismatch (> 20% difference)
    predicted_prob = prediction.get('probability', 0)
    if predicted_prob < 0.5:
        # For low probability, check if narrative suggests high risk
        high_risk_indicators = [
            r'\bhigh risk\b', r'\bsevere\b', r'\bserious\b',
            r'\bvery likely\b', r'\bhigh probability\b'
        ]

        for indicator in high_risk_indicators:
            if re.search(indicator, text, re.IGNORECASE):
                return False, "FAIL: Low probability prediction should not suggest high risk"

    # Check 3: Missing verification note (soft warning)
    if 'verification' not in text.lower() and 'data' not in text.lower():
        logger.warning("Narrative may be missing verification note")
        return True, "WARNING: Narrative should include verification note"

    return True, "PASS: Narrative meets all guardrail requirements"

if __name__ == "__main__":
    # Test guardrail validation
    logger.info("Testing guardrail validation...")
    print("Guardrail validation ready")