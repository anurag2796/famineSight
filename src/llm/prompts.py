# src/llm/prompts.py
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# System prompt with strict rules
SYSTEM_PROMPT = """
You are an expert humanitarian data analyst with deep expertise in famine prediction, conflict analysis, and climate impact assessment. You must follow these strict rules:

1. NEVER make up data or invent scenarios that aren't supported by the provided data
2. NEVER escalate or amplify concerns beyond what the data clearly indicates
3. ALWAYS verify that your analysis is based on actual data patterns, not speculation
4. NEVER use terms like "catastrophe", "mass death", or "famine" unless explicitly supported by the data
5. ALWAYS provide clear citations to the data sources when making claims
6. ALWAYS state the confidence level of your analysis (low/medium/high)
7. ALWAYS include a verification note at the end of your response

Data Analysis Guidelines:
- Focus on patterns, not predictions
- Use only the provided features and metrics
- Be precise about what the data shows vs. what you infer
- When in doubt, state uncertainty clearly
- If data is missing or incomplete, state this explicitly

Your role is to provide objective, data-driven insights that can inform humanitarian response decisions. Be factual, precise, and avoid sensationalism.
"""

def build_prompt(prediction: Dict[str, Any], alerts: list, rules: Dict[str, Any]) -> str:
    """
    Build a prompt for the LLM based on the analysis results.

    Args:
        prediction: Prediction results from the model
        alerts: List of detected anomalies/alerts
        rules: Association rules from analysis

    Returns:
        Formatted prompt string
    """
    # Prepare the data for the prompt
    prompt_parts = [
        "ANALYSIS DATA:",
        f"Prediction: Crisis probability = {prediction.get('risk_level', 'unknown')}",
        f"Confidence: {prediction.get('confidence', 'unknown')}",
        f"SHAP factors: {prediction.get('shap_factors', 'unknown')}",
        ""
    ]

    # Add alerts if any
    if alerts:
        prompt_parts.append("ANOMALY ALERTS:")
        for alert in alerts[:3]:  # Show top 3 alerts
            prompt_parts.append(f"- {alert['date']}: {alert['severity']} alert for {alert['district']}")
        prompt_parts.append("")

    # Add association rules if any
    if rules:
        prompt_parts.append("ASSOCIATION RULES:")
        # Show top 3 rules — rules may be a dict of lists (JSON) or DataFrames
        if 'fpgrowth_rules' in rules:
            fr = rules['fpgrowth_rules']
            # If JSON list
            if isinstance(fr, list) and len(fr) > 0:
                for rule in fr[:3]:
                    antecedents = ', '.join(rule['antecedents']) if isinstance(rule.get('antecedents'), list) else rule.get('antecedents')
                    consequents = ', '.join(rule['consequents']) if isinstance(rule.get('consequents'), list) else rule.get('consequents')
                    prompt_parts.append(f"- {antecedents} → {consequents} (confidence: {rule.get('confidence',0):.2f}, lift: {rule.get('lift',0):.2f})")
            # If pandas DataFrame-like
            elif hasattr(fr, 'empty') and not fr.empty:
                for _, rule in fr.head(3).iterrows():
                    prompt_parts.append(f"- {rule['antecedents']} → {rule['consequents']} (confidence: {rule['confidence']:.2f}, lift: {rule['lift']:.2f})")
        prompt_parts.append("")

    # Add data verification note
    prompt_parts.append("VERIFICATION NOTE: All conclusions must be directly supported by the data provided. No speculation or extrapolation.")

    return "\n".join(prompt_parts)