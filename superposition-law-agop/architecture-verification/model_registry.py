from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class ModelSpec:
    family: str
    name: str
    repo_id: str
    notes: str = ""


MODEL_SPECS: Dict[str, ModelSpec] = {
    "pythia-70m": ModelSpec("pythia", "70m", "EleutherAI/pythia-70m"),
    "pythia-160m": ModelSpec("pythia", "160m", "EleutherAI/pythia-160m"),
    "pythia-410m": ModelSpec("pythia", "410m", "EleutherAI/pythia-410m"),
    "pythia-1b": ModelSpec("pythia", "1b", "EleutherAI/pythia-1b"),
    "pythia-1.4b": ModelSpec("pythia", "1.4b", "EleutherAI/pythia-1.4b"),
    "pythia-2.8b": ModelSpec("pythia", "2.8b", "EleutherAI/pythia-2.8b"),
    "pythia-6.9b": ModelSpec("pythia", "6.9b", "EleutherAI/pythia-6.9b"),
    "pythia-12b": ModelSpec("pythia", "12b", "EleutherAI/pythia-12b"),
    "llama-3.2-1b": ModelSpec(
        "llama",
        "3.2-1b",
        "meta-llama/Llama-3.2-1B",
        notes="Requires Hugging Face access approval from Meta.",
    ),
    "llama-3.2-3b": ModelSpec(
        "llama",
        "3.2-3b",
        "meta-llama/Llama-3.2-3B",
        notes="Requires Hugging Face access approval from Meta.",
    ),
    "llama-3.1-8b": ModelSpec(
        "llama",
        "3.1-8b",
        "meta-llama/Llama-3.1-8B",
        notes="Requires Hugging Face access approval from Meta.",
    ),
}


DEFAULT_MODEL_NAMES: List[str] = [
    "pythia-70m",
    "pythia-160m",
    "pythia-410m",
    "pythia-1b",
    "pythia-1.4b",
    "pythia-2.8b",
    "pythia-6.9b",
    "pythia-12b",
    "llama-3.2-1b",
    "llama-3.2-3b",
    "llama-3.1-8b",
]


def get_specs(model_names: Iterable[str]) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    for name in model_names:
        key = name.strip()
        if not key:
            continue
        if key not in MODEL_SPECS:
            available = ", ".join(sorted(MODEL_SPECS))
            raise KeyError(f"Unknown model '{key}'. Available models: {available}")
        specs.append(MODEL_SPECS[key])
    return specs
