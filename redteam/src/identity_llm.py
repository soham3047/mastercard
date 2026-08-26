"""
identity_llm.py — Stage A5: KYC-style identity content + transaction
narratives, generated via an LLM call.

This is what makes the output "GenAI-powered fraud generation" rather
than pure parametric sampling — the dashboard (Person D) surfaces this
text next to each flagged account/ring so a judge/analyst sees a
plausible identity, not just a bare account id.

Falls back to an offline templated generator whenever ANTHROPIC_API_KEY
isn't set (or the SDK isn't installed, or a call fails), so the rest of
the pipeline — and any offline demo/judging run with no network — still
works without a key.

All generated identities are explicitly synthetic test fixtures for a
fraud-detection benchmark; no real people or real data are involved.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List

try:
    import anthropic  # pip install anthropic
    _HAS_ANTHROPIC_SDK = True
except ImportError:
    _HAS_ANTHROPIC_SDK = False


@dataclass
class Identity:
    account_id: str
    name: str
    occupation: str
    income_bracket: str
    address_plausibility: str
    narrative: str

    def as_dict(self) -> Dict[str, str]:
        return asdict(self)


_OCCUPATIONS = [
    "Retail shop owner", "IT contractor", "Freelance designer", "Delivery agent",
    "Salaried accountant", "Small trader", "College student", "Auto-rickshaw driver",
]
_INCOME_BRACKETS = [
    "Rs 15,000-25,000/mo", "Rs 25,000-50,000/mo", "Rs 50,000-1,00,000/mo", "Below Rs 15,000/mo",
]
_FIRST_NAMES = [
    "Aarav", "Priya", "Rohan", "Ishaan", "Ananya", "Vikram", "Neha", "Karan",
    "Meera", "Sanjay", "Divya", "Arjun", "Kavya", "Rahul", "Pooja", "Aditya",
]
_LAST_NAMES = [
    "Sharma", "Patel", "Reddy", "Nair", "Iyer", "Gupta", "Verma", "Joshi",
    "Mehta", "Rao", "Kumar", "Singh", "Desai", "Pillai",
]


def _offline_identity(account_id: str, node_type: str, rng: random.Random) -> Identity:
    """Deterministic, no-network fallback — used when no API key is present."""
    name = f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"
    occupation = rng.choice(_OCCUPATIONS)
    income = rng.choice(_INCOME_BRACKETS)
    if node_type == "decoy":
        plausibility = "Address matches occupation/income profile (registered, verifiable pincode)."
        narrative = f"{name} ({occupation}, {income}) — routine transaction activity."
    elif node_type == "hub":
        plausibility = "Address partially verifiable — pincode valid but registered residence age < 30 days."
        narrative = f"{name}'s account shows unusually high fan-out relative to its stated {occupation.lower()} income."
    else:  # "ring"
        plausibility = "Address partially verifiable — pincode valid but registered residence age < 30 days."
        narrative = f"{name} ({occupation}) is part of a sequence of transfers consistent with layering."
    return Identity(account_id, name, occupation, income, plausibility, narrative)


_SYSTEM_PROMPT = (
    "You generate SYNTHETIC KYC-style identity profiles and short transaction "
    "narratives for a fraud-detection RESEARCH TESTBED (Mastercard Innovation "
    "Challenge hackathon). All accounts are entirely fictional test fixtures used "
    "to benchmark a fraud-ring detector -- no real people, no real data. Respond "
    "ONLY with valid JSON matching the requested schema, no markdown fences, no "
    "preamble."
)


def _llm_identity_batch(accounts: List[Dict[str, str]], client: "anthropic.Anthropic") -> List[Identity]:
    """accounts: list of {"account_id": ..., "node_type": "ring|decoy|hub"}."""
    prompt = (
        "Generate one synthetic KYC-style identity + one-sentence transaction "
        "narrative for each account below. Return a JSON array, one object per "
        "account, each with exactly these keys: account_id, name, occupation, "
        "income_bracket, address_plausibility, narrative. For node_type 'ring' or "
        "'hub' accounts, address_plausibility should note a subtle red flag (e.g. "
        "recently registered address, income/occupation mismatch); for 'decoy' "
        "accounts it should read as unremarkable/verified.\n\n"
        f"Accounts:\n{json.dumps(accounts, indent=2)}"
    )
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in resp.content if getattr(block, "type", None) == "text")
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    parsed = json.loads(text)
    return [Identity(**item) for item in parsed]


def generate_identities(
    accounts: List[Dict[str, str]],
    seed: int = 42,
    batch_size: int = 25,
) -> Dict[str, Dict[str, str]]:
    """accounts: list of {"account_id": ..., "node_type": ...}.

    Returns {account_id: identity_dict}. Tries the Anthropic API first (if
    ANTHROPIC_API_KEY is set and the SDK is installed); falls back to the
    offline generator per-batch on any failure, so a partial API outage
    (e.g. flaky venue wifi during judging) never breaks the pipeline.
    """
    rng = random.Random(seed)
    result: Dict[str, Dict[str, str]] = {}

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key and _HAS_ANTHROPIC_SDK:
        client = anthropic.Anthropic(api_key=api_key)
        for i in range(0, len(accounts), batch_size):
            batch = accounts[i:i + batch_size]
            try:
                identities = _llm_identity_batch(batch, client)
                for ident in identities:
                    result[ident.account_id] = ident.as_dict()
            except Exception as exc:  # noqa: BLE001 — any failure falls back, never crashes the run
                print(f"[identity_llm] LLM batch failed ({exc}); using offline generator for this batch.")
                for acc in batch:
                    ident = _offline_identity(acc["account_id"], acc["node_type"], rng)
                    result[ident.account_id] = ident.as_dict()
    else:
        if not api_key:
            print("[identity_llm] ANTHROPIC_API_KEY not set -- using offline identity generator.")
        elif not _HAS_ANTHROPIC_SDK:
            print("[identity_llm] 'anthropic' package not installed -- using offline identity generator.")
        for acc in accounts:
            ident = _offline_identity(acc["account_id"], acc["node_type"], rng)
            result[ident.account_id] = ident.as_dict()

    return result
