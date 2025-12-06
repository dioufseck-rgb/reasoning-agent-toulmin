# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from __future__ import annotations

import os
import json
import math
from typing import List, Literal, Dict, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, Field
import requests

# -------------------------
# 0. Config & environment
# -------------------------

load_dotenv()

# IMPORTANT:
# Point LLAMA_MODEL to your QUANTIZED model tag in Ollama, e.g.:
#   set LLAMA_MODEL=llama3:8b-q4          (Windows)
#   export LLAMA_MODEL=llama3:8b-q4       (mac/Linux)
#
# If not set, it defaults to "llama3".
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "llama3")

# Thresholds for governance
ACTIONABLE_THRESHOLD = 0.85   # min confidence for auto-action
SUBCLAIM_MIN = 0.60           # min confidence for any sub-claim


# -------------------------
# 1. Generic DECISION_CONTEXT
#    Example: heavy equipment maintenance
# -------------------------

DECISION_CONTEXT: Dict[str, str] = {
    "CASE_DOMAIN": "heavy equipment maintenance",

    "CASE_SPECIFIC_INFORMATION": """
We are evaluating whether to perform scheduled preventive maintenance now
or delay it by approximately one month on a large hydraulic excavator used
in a mining operation (Unit ID: EX-47C).

Equipment details:
- Model: Komatsu PC7000 hydraulic excavator
- Age: 7 years
- Operating hours: 18,420 total hours
- Current usage: 16–18 hours/day on rock overburden removal
- Environment: high dust, abrasive materials, variable temperature (−5°C to 32°C)
- Recent changes: increased workload due to expansion of pit area

Maintenance history:
- Last full preventive maintenance: 11 months ago (recommended every 12 months or 2,000 hours)
- Last major component overhaul: 2.5 years ago
- No catastrophic failures in the past 3 years, but several near-miss events
  were logged involving overheating of hydraulic pumps.

Current condition observations (last 3 weeks):
- Hydraulic pump temperature trending slightly upward (+6°C above baseline)
- Slight increase in metal particles in lubrication oil (spectrographic wear analysis)
- Vibration monitoring shows early-stage bearing wear on swing motor
- Fuel consumption increased ~3% over last quarter
- No alarms or red flags triggered in onboard diagnostics

Operational considerations:
- Equipment is critical for maintaining production throughput
- A planned one-month maintenance window is available at the end of next month
  when ore processing plant is scheduled to be down for refit
- If maintenance is done now, production would be reduced by ~12% for 5 days
- If maintenance is delayed, no immediate production loss, but unplanned failure
  could result in a 10–15 day outage

Target decision:
- Should maintenance be performed now, or delayed until the next scheduled outage?
""",

    "DECISION_RULES_AND_POLICIES": """
Maintenance Policy MP-2022 §3.1 – Preventive maintenance intervals:
- Major preventive maintenance intervals:
  - Every 12 months OR
  - Every 2,000 operating hours (whichever occurs first)
- Tolerance window for scheduling:
  - ± 1 month is acceptable IF no critical wear indicators are present.

Maintenance Policy MP-2022 §4.2 – Wear and condition monitoring:
- If hydraulic temperature trend exceeds baseline by >8°C or
  metal particulate levels exceed threshold T1 → maintenance must be performed
  immediately.
- If trending is elevated but below critical threshold, engineering judgment
  may be used with risk analysis.

Safety Policy SP-2019 §5.4 – Major component failure risk:
- Unplanned failure of hydraulic components may cause:
  - Environmental hazard (oil spill)
  - Safety risk to personnel
  - Costly downtime
- Recommended guidance:
  - If probability of failure >5% within the next 30 days, schedule maintenance
    proactively.

Risk Appetite Statement RAS-2023:
- Acceptable production loss for planned maintenance: up to 10% for 1 week
- Unplanned outage risk tolerance: less than 3% probability of losing >7 days
  production in any given month

Production Policy PP-2021 §2.7:
- If equipment downtime reduces throughput >10% during peak demand periods,
  escalation to operations planning is required for approval.

Exception Policy EP-2024 §1.2:
- Exceptions to maintenance timing may be granted if:
  - Monitoring shows early wear but no critical indicators,
  - Required downtime exceeds 10% throughput,
  - A safe short-term mitigation plan is documented,
  AND
  - Maintenance is rescheduled within 6 weeks.
""",

    "GENERAL_KNOWLEDGE": """
General maintenance principles in heavy equipment:
- Operating heavy machinery in dusty, abrasive environments accelerates wear
  on hydraulic systems, bearings, and swing components.
- Small trend increases in temperature, vibration, and wear particles can be
  early indicators of developing faults.
- Preventive maintenance reduces the likelihood of catastrophic failure but
  incurs planned downtime that impacts production.
- Delaying maintenance often increases risk non-linearly, especially when
  multiple minor indicators exist simultaneously.

Hydraulic system knowledge:
- Elevated hydraulic temperature increases fluid oxidation and accelerates
  wear of pump components.
- Metal particulate in oil typically indicates wear of bearings, gears, or pump
  surfaces. Trending upward is a cautionary flag even if still below thresholds.

Operational trade-offs:
- Planned downtime during scheduled outages is economically preferable to
  unplanned failures.
- Unplanned failures often cause longer outages due to secondary damage,
  logistics delays, and safety/environmental incident management.

Risk considerations:
- Production impact must be balanced against failure risk cost.
- If failure probability is elevated but uncertain, proactive maintenance is
  often the lower-risk strategy—especially when equipment is mission-critical.
"""
}


def render_decision_context(ctx: Dict[str, str]) -> str:
    """
    Render DECISION_CONTEXT into a single text blob with clearly marked sections.
    """
    return f"""
CASE DOMAIN
-----------
{ctx.get("CASE_DOMAIN", "").strip()}

CASE SPECIFIC INFORMATION
-------------------------
{ctx.get("CASE_SPECIFIC_INFORMATION", "").strip()}

DECISION RULES AND POLICIES
---------------------------
{ctx.get("DECISION_RULES_AND_POLICIES", "").strip()}

GENERAL KNOWLEDGE
-----------------
{ctx.get("GENERAL_KNOWLEDGE", "").strip()}
""".strip()


# -------------------------
# 2. Core data models
# -------------------------

Qualifier = Literal[
    "very low", "low", "medium", "moderately high", "high", "very high"
]
ClaimType = Literal["information", "actionable"]


class Backing(BaseModel):
    source: Literal["policy", "model", "data"]
    document: Optional[str] = None
    section: Optional[str] = None
    snippet: Optional[str] = None
    ref: Optional[str] = None


class ToulminBlock(BaseModel):
    id: str
    claim: str
    claim_type: ClaimType
    data: List[str]
    warrant: str
    backing: List[Backing]
    qualifier_text: Qualifier
    confidence_value: float       # 0.0 - 1.0
    confidence_rationale: str
    rebuttals: List[str]


class ActionableToulmin(BaseModel):
    id: str = "F"
    claim: str
    claim_type: Literal["actionable"] = "actionable"
    data: List[str]               # IDs of supporting sub-claims, e.g. ["S1","S2","S3","S4"]
    warrant: str
    backing: List[Backing]
    qualifier_text: Qualifier
    confidence_value: float
    confidence_rationale: str
    rebuttals: List[str]


class FinalDecision(BaseModel):
    question: str
    sub_claims: Dict[str, ToulminBlock]
    final_claim: str
    final_confidence: float
    requires_human: bool


class SubQuestionNode(BaseModel):
    """
    Hierarchical sub-question node, e.g.:
    S1 -> S1.1, S1.2, ...
    """
    id: str
    text: str
    children: List["SubQuestionNode"] = Field(default_factory=list)
    type: Optional[str] = None  # e.g. "risk", "cost", "constraint", etc.


# Pydantic v2 uses model_rebuild, v1 uses update_forward_refs
try:
    SubQuestionNode.model_rebuild()
except AttributeError:
    SubQuestionNode.update_forward_refs()


# -------------------------
# 3. Qualifier mapping
# -------------------------

QUALIFIER_FROM_CONF = [
    (0.95, "very high"),
    (0.85, "high"),
    (0.70, "moderately high"),
    (0.50, "medium"),
    (0.35, "low"),
    (0.0,  "very low"),
]


def conf_to_qualifier(p: float) -> str:
    p = max(0.0, min(1.0, p))
    for threshold, label in QUALIFIER_FROM_CONF:
        if p >= threshold:
            return label
    return "very low"


# -------------------------
# 4. LLaMA (quantized) via Ollama
# -------------------------

def call_llama_chat(system_prompt: str, user_prompt: str) -> str:
    """
    Call a (possibly quantized) LLaMA model via Ollama /api/chat and return
    the assistant content as a string.

    The specific model (including quantization) is controlled by LLAMA_MODEL.
    """
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": LLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }
    resp = requests.post(url, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    content = data.get("message", {}).get("content", "")
    return content.strip()


def generate_hierarchical_sub_questions(
    question: str,
    decision_context_text: str,
    case_domain: str,
    max_depth: int = 2,
    max_top_level: int = 4,
) -> List[SubQuestionNode]:
    """
    Ask LLaMA to build a hierarchical decomposition of the decision question.
    Returns a list of root SubQuestionNode objects.
    """

    system_msg = (
        f"You are an AI reasoning architect in the domain of '{case_domain}'. "
        "Your job is to decompose a decision question into a hierarchical set of "
        "clear, non-overlapping sub-questions for transparent reasoning. "
        "You output only JSON, no explanation."
    )

    user_msg = f"""
You are designing a hierarchical reasoning plan for a decision in the domain of "{case_domain}".

Here is the full DECISION_CONTEXT:

{decision_context_text}

Main decision question:
{question}

Goal:
- Construct a hierarchical decomposition of the question into 1 to {max_top_level}
  top-level sub-questions.
- Each top-level sub-question may have 0 or more child sub-questions (nested) up to
  depth {max_depth}.
- Child sub-questions should refine or break down the parent question.
- All sub-questions are informational assessments (no final actions).

Output format (JSON only):

{{
  "roots": [
    {{
      "id": "S1",
      "text": "Top-level sub-question 1",
      "type": "risk | condition | cost | constraint | other",
      "children": [
        {{
          "id": "S1.1",
          "text": "Child sub-question 1.1",
          "type": "condition",
          "children": []
        }}
      ]
    }},
    {{
      "id": "S2",
      "text": "Top-level sub-question 2",
      "type": "cost",
      "children": []
    }}
  ]
}}

Rules:
- IDs must be unique. Use 'S1', 'S2', ... at top level, and 'S1.1', 'S1.2', etc. for children.
- 'type' should be one of: "risk", "condition", "cost", "constraint", or "other".
- Sub-question texts must be short, clear, and grounded in the DECISION_CONTEXT.
- Do NOT include the final decision as a sub-question.
- Respond with JSON ONLY. No markdown, no backticks, no comments.
"""

    content = call_llama_chat(system_msg, user_msg)

    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse hierarchical sub-questions JSON: {e}\nRaw content:\n{content}")

    if "roots" not in obj or not isinstance(obj["roots"], list):
        raise RuntimeError(f"Unexpected JSON structure for hierarchical sub-questions:\n{obj}")

    roots: List[SubQuestionNode] = []
    for item in obj["roots"]:
        try:
            node = SubQuestionNode(**item)
            roots.append(node)
        except ValidationError as e:
            raise RuntimeError(f"Invalid SubQuestionNode structure: {e}\nItem:\n{item}")

    if not roots:
        raise RuntimeError(f"No valid root sub-questions found in JSON:\n{obj}")

    return roots


def generate_toulmin_block(
    question: str,
    subq_id: str,
    subq_text: str,
    decision_context_text: str,
    case_domain: str,
) -> ToulminBlock:
    """
    Ask LLaMA to produce one ToulminBlock in JSON for a sub-question,
    grounded in the DECISION_CONTEXT and case_domain.

    NOTE: The model's numeric confidence_value is canonical.
    We ALWAYS recompute qualifier_text from that number in code to enforce consistency.
    """

    system_msg = (
        f"You are a transparent reasoning assistant for '{case_domain}' decisions. "
        "For each sub-question, you must produce a single Toulmin-style argument "
        "as strict JSON only. Do not include any text outside JSON."
    )

    user_msg = f"""
You are reasoning about a decision in the domain of "{case_domain}".

Here is the full DECISION_CONTEXT:

{decision_context_text}

Main decision question:
{question}

Sub-question (ID={subq_id}):
{subq_text}

You must answer with a single JSON object matching this schema:

{{
  "id": "S1",
  "claim": "string",
  "claim_type": "information",
  "data": ["string"],
  "warrant": "string",
  "backing": [
    {{
      "source": "policy" | "model" | "data",
      "document": "string or null",
      "section": "string or null",
      "snippet": "string or null",
      "ref": "string or null"
    }}
  ],
  "qualifier_text": "very low" | "low" | "medium" | "moderately high" | "high" | "very high",
  "confidence_value": 0.0 to 1.0,
  "confidence_rationale": "string",
  "rebuttals": ["string"]
}}

IMPORTANT:
- 'confidence_value' is your probability (0–1) that the CLAIM is correct.
- 'qualifier_text' is a verbal label for your confidence level, but the system will
  REPLACE it based on 'confidence_value', so consistency is more important than wording.
- Do NOT use 'qualifier_text' to describe the risk level itself. It describes how SURE
  you are that the CLAIM is true, not how big or small the risk is.

Rules:
- claim_type MUST be "information" for these sub-claims.
- Use the DECISION_CONTEXT above to ground your reasoning and backings.
- When citing rules or policies, refer to them by name and section if possible
  (e.g., "Maintenance Policy MP-2022 §3.1" or equivalent in other domains).
- Use 'policy' for policy/rule backings, 'data' for case facts/measurements,
  and 'model' for general domain knowledge or heuristic reasoning.
- Respond with JSON ONLY, no markdown, no comments, no backticks.
"""

    content = call_llama_chat(system_msg, user_msg)
    cleaned = content.strip()

    # Strip code fences if model wrapped JSON in ```json ... ```
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    # --- Robust JSON parsing with auto-closing of brackets/braces ---
    def try_parse_with_repair(s: str) -> dict:
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            repaired = s

            # Balance square brackets
            diff_brackets = repaired.count("[") - repaired.count("]")
            if diff_brackets > 0:
                repaired += "]" * diff_brackets

            # Balance curly braces
            diff_braces = repaired.count("{") - repaired.count("}")
            if diff_braces > 0:
                repaired += "}" * diff_braces

            return json.loads(repaired)

    try:
        obj = try_parse_with_repair(cleaned)

        # Enforce numeric-first, label-derived semantics
        if "confidence_value" not in obj:
            raise RuntimeError(f"No confidence_value in Toulmin JSON:\n{obj}")
        try:
            conf = float(obj["confidence_value"])
        except Exception as e:
            raise RuntimeError(f"Invalid confidence_value '{obj['confidence_value']}': {e}")

        # Override any model-provided qualifier_text
        obj["qualifier_text"] = conf_to_qualifier(conf)

        # Fill in missing optional lists if model omits them
        if "rebuttals" not in obj or obj["rebuttals"] is None:
            obj["rebuttals"] = []
        if "backing" not in obj or obj["backing"] is None:
            obj["backing"] = []

        block = ToulminBlock(**obj)
        return block

    except (json.JSONDecodeError, ValidationError, RuntimeError) as e:
        raise RuntimeError(f"Failed to parse Toulmin JSON: {e}\nRaw content:\n{content}")


# -------------------------
# 5. Tree utilities
# -------------------------

def iter_nodes(root: SubQuestionNode):
    """Depth-first iteration over a tree."""
    yield root
    for child in root.children:
        yield from iter_nodes(child)


def collect_leaves(roots: List[SubQuestionNode]) -> List[SubQuestionNode]:
    leaves: List[SubQuestionNode] = []
    for r in roots:
        for node in iter_nodes(r):
            if not node.children:
                leaves.append(node)
    return leaves


def collect_internal_nodes_bottom_up(roots: List[SubQuestionNode]) -> List[SubQuestionNode]:
    """
    Return internal nodes in bottom-up order (children before parents),
    so we can aggregate confidences.
    """
    ordered: List[SubQuestionNode] = []

    def visit(node: SubQuestionNode):
        for c in node.children:
            visit(c)
        if node.children:
            ordered.append(node)

    for r in roots:
        visit(r)
    return ordered


def aggregate_node_from_children(
    node: SubQuestionNode,
    child_blocks: Dict[str, ToulminBlock],
) -> ToulminBlock:
    """
    Build a ToulminBlock for an internal node by aggregating child confidences.
    """
    child_ids = [c.id for c in node.children]
    child_subclaims = {cid: child_blocks[cid] for cid in child_ids}
    conf = aggregate_confidence_weighted(child_subclaims)
    qualifier = conf_to_qualifier(conf)

    claim_text = f"Overall, the answer to sub-question '{node.text}' is supported by its child assessments."
    warrant = (
        "If all child sub-questions under this topic are reasonably well-supported, "
        "the parent question can be considered supported as well."
    )

    backing = [
        Backing(
            source="model",
            document=None,
            section=None,
            snippet=(
                "Parent conclusions can be derived from well-supported child assessments "
                "in a hierarchical reasoning structure."
            ),
            ref="hierarchical-reasoning",
        )
    ]

    return ToulminBlock(
        id=node.id,
        claim=claim_text,
        claim_type="information",
        data=child_ids,
        warrant=warrant,
        backing=backing,
        qualifier_text=qualifier,
        confidence_value=conf,
        confidence_rationale=(
            "Derived from child sub-claims' confidences using a weighted log-odds aggregation."
        ),
        rebuttals=[
            "If any child sub-claim is later revised significantly, this aggregate may change."
        ],
    )


# -------------------------
# 6. Confidence aggregation & governance
# -------------------------

def aggregate_confidence_weighted(
    sub_claims: Dict[str, ToulminBlock],
    weights: Dict[str, float] | None = None,
) -> float:
    """
    Aggregate sub-claim confidences using weighted log-odds.

    Intuition:
    - Convert each probability p to logit = log(p / (1 - p))
    - Weight more critical sub-claims higher (e.g., failure risk)
    - Sum weighted logits, then convert back to probability
    """
    if not sub_claims:
        return 0.0

    if weights is None:
        weights = {sid: 1.0 for sid in sub_claims.keys()}

    total_weight = sum(weights.values())
    if total_weight == 0:
        return 0.0

    logit_sum = 0.0
    for sid, block in sub_claims.items():
        w = weights.get(sid, 1.0)
        p = max(1e-6, min(1 - 1e-6, block.confidence_value))
        logit = math.log(p / (1 - p))
        logit_sum += w * logit

    avg_logit = logit_sum / total_weight
    p_final = 1 / (1 + math.exp(-avg_logit))
    return p_final


def compute_requires_human(final_conf: float, sub_claims: Dict[str, ToulminBlock]) -> bool:
    """
    Simple governance logic: if final confidence or any sub-claim is below threshold,
    require human review.
    """
    if final_conf < ACTIONABLE_THRESHOLD:
        return True
    for block in sub_claims.values():
        if block.confidence_value < SUBCLAIM_MIN:
            return True
    return False


def derive_top_level_weights(roots: List[SubQuestionNode]) -> Dict[str, float]:
    """
    Derive aggregation weights for top-level nodes based on their semantic 'type'
    and/or text (risk, cost, constraints, etc.).
    """
    weights: Dict[str, float] = {}

    for root in roots:
        base = 1.0
        t = (root.type or "").lower()
        txt = root.text.lower()

        # Type-driven weighting
        if t == "risk":
            base = 1.6
        elif t == "cost":
            base = 1.3
        elif t == "constraint":
            base = 1.2
        elif t == "condition":
            base = 1.0
        else:
            base = 1.0

        # Fallback keyword-based bump if type is missing or generic
        if not t or t == "other":
            if any(k in txt for k in ["risk", "failure", "safety"]):
                base = max(base, 1.5)
            if any(k in txt for k in ["cost", "production", "downtime", "throughput"]):
                base = max(base, 1.3)

        weights[root.id] = base

    return weights


# -------------------------
# 7. Demo pipeline (hierarchical)
# -------------------------

def run_demo_hierarchical():
    case_domain = DECISION_CONTEXT["CASE_DOMAIN"].strip()
    decision_context_text = render_decision_context(DECISION_CONTEXT)

    question = (
        "Given the current case and applicable rules, should we perform maintenance now "
        "or safely delay it by approximately one month?"
    )

    print(f"=== Domain: {case_domain} ===")
    print(f"Using LLaMA model: {LLAMA_MODEL}")
    print("=== Generating HIERARCHICAL sub-questions with LLaMA (quantized) ===")
    roots = generate_hierarchical_sub_questions(
        question=question,
        decision_context_text=decision_context_text,
        case_domain=case_domain,
        max_depth=2,
        max_top_level=4,
    )

    print("\nHierarchical sub-questions:")
    for r in roots:
        print(f"{r.id} [{r.type}]: {r.text}")
        for node in iter_nodes(r):
            if node is not r:
                indent = "  " * node.id.count(".")
                print(f"{indent}{node.id} [{node.type}]: {node.text}")

    # ---- Leaf Toulmin blocks from LLaMA ----
    leaves = collect_leaves(roots)
    leaf_blocks: Dict[str, ToulminBlock] = {}

    print("\n=== Generating Toulmin sub-claims for LEAF sub-questions ===")
    for node in leaves:
        print(f"\n--- {node.id}: {node.text}")
        block = generate_toulmin_block(
            question,
            node.id,
            node.text,
            decision_context_text,
            case_domain,
        )
        leaf_blocks[node.id] = block
        print("Claim:", block.claim)
        print("Confidence:", block.confidence_value, f"({block.qualifier_text})")

    # ---- Internal nodes via bottom-up aggregation ----
    all_blocks: Dict[str, ToulminBlock] = {}
    all_blocks.update(leaf_blocks)

    internal_nodes = collect_internal_nodes_bottom_up(roots)

    print("\n=== Aggregating internal nodes bottom-up ===")
    for node in internal_nodes:
        block = aggregate_node_from_children(node, all_blocks)
        all_blocks[node.id] = block
        print(
            f"{node.id} aggregated confidence:",
            round(block.confidence_value, 4),
            f"({block.qualifier_text})",
        )

    # For the final decision, use ONLY top-level nodes as 'data'
    top_level_ids = [r.id for r in roots]
    top_level_blocks = {tid: all_blocks[tid] for tid in top_level_ids}

    # Derive weights based on semantic 'type' / text
    top_level_weights = derive_top_level_weights(roots)

    print("\nTop-level weights (semantic):")
    for sid in top_level_ids:
        print(f"{sid} -> weight {top_level_weights.get(sid, 1.0)}")

    final_conf = aggregate_confidence_weighted(top_level_blocks, top_level_weights)
    final_qualifier = conf_to_qualifier(final_conf)
    requires_human = compute_requires_human(final_conf, all_blocks)

    # ---- Build final actionable Toulmin block ----
    final_claim_text = (
        "Perform preventive maintenance now rather than delaying by one month, "
        "based on the aggregated assessments of condition, risk, and production impact."
    )

    final_warrant = (
        "If the top-level assessments of equipment condition, failure risk, and production "
        "trade-offs are collectively supportive, then proactive maintenance is justified."
    )

    final_backing = [
        Backing(
            source="policy",
            document="Maintenance Policy MP-2022",
            section="§3.1 and §4.2",
            snippet="Defines preventive intervals and treatment of elevated but sub-critical indicators.",
            ref=None,
        ),
        Backing(
            source="policy",
            document="Risk Appetite Statement RAS-2023",
            section="§2.1",
            snippet="Limits tolerance for unplanned outages exceeding 7 days in a month.",
            ref=None,
        ),
    ]

    actionable = ActionableToulmin(
        claim=final_claim_text,
        data=top_level_ids,
        warrant=final_warrant,
        backing=final_backing,
        qualifier_text=final_qualifier,
        confidence_value=final_conf,
        confidence_rationale=(
            "Aggregated from top-level sub-claims' confidences using a weighted log-odds model, "
            "with weights derived from each branch's semantic type (risk, cost, constraints, etc.)."
        ),
        rebuttals=[
            "If any major branch (e.g., risk or production impact) is later re-evaluated with "
            "substantially lower confidence, this recommendation should be revisited.",
            "If production constraints make immediate maintenance infeasible, escalation to "
            "operations planning is required to adjust the schedule.",
        ],
    )

    final = FinalDecision(
        question=question,
        sub_claims=all_blocks,
        final_claim=actionable.claim,
        final_confidence=final_conf,
        requires_human=requires_human,
    )

    print("\n=== Final decision summary (hierarchical) ===")
    print("Final claim:", final.final_claim)
    print("Final confidence:", round(final.final_confidence, 4))
    print("Requires human approval:", final.requires_human)

    print("\n=== Final actionable Toulmin block (hierarchical) ===")
    print(actionable.model_dump_json(indent=2))

    print("\n=== Raw FinalDecision object (hierarchical) ===")
    print(final.model_dump_json(indent=2))


if __name__ == "__main__":
    # Check that Ollama is reachable and the model is available
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        r.raise_for_status()
        tags_json = r.json()
        models = tags_json.get("models") or tags_json.get("models", [])
        print("Ollama is reachable at", OLLAMA_HOST)
        try:
            available = [m.get("name") for m in models]
            print("Available models:", available)
        except Exception:
            pass
    except Exception as e:
        raise SystemExit(
            f"ERROR: Could not reach Ollama at {OLLAMA_HOST}. "
            f"Is Ollama running and is the '{LLAMA_MODEL}' model pulled?\nDetails: {e}"
        )

    run_demo_hierarchical()
