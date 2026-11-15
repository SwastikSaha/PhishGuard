
from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


class ThreatIntelligenceReport(BaseModel):
    provider: str = "PhishHunt"

    status: Literal[
        "KNOWN_PHISHING",
        "NOT_FOUND",
        "UNAVAILABLE",
    ]

    verdict: str | None = None

    known_match: bool = False


class MachineLearningReport(BaseModel):
    prediction: Literal[
        "PHISHING",
        "BENIGN_CANDIDATE",
    ]

    phishing_score: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)

    base_model_scores: dict[str, float] = Field(
        default_factory=dict
    )


class SecurityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str

    final_prediction: Literal[
        "PHISHING",
        "BENIGN_CANDIDATE",
    ]

    classification_source: str

    risk_level: Literal[
        "HIGH",
        "UNKNOWN",
    ]

    threat_intelligence: ThreatIntelligenceReport

    machine_learning: MachineLearningReport | None = None

    evidence: list[str] = Field(default_factory=list)

    limitations: list[str] = Field(default_factory=list)

    recommendations: list[str] = Field(default_factory=list)

    explanation: str

    explanation_source: Literal[
        "LLM",
        "PYTHON_FALLBACK",
    ]
