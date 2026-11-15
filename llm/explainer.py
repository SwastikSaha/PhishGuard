
import json

from llm.llm_client import OllamaClient
from llm.prompts import SYSTEM_PROMPT, build_user_prompt

from llm.schemas import (
    SecurityReport,
    ThreatIntelligenceReport,
    MachineLearningReport,
)


class PhishGuardExplainer:

    def __init__(self, client=None):
        self.client = client or OllamaClient()

    # -------------------------------------------------
    # FALLBACK EXPLANATION
    # -------------------------------------------------

    @staticmethod
    def _build_fallback_explanation(
        prediction: str,
        intelligence_status: str,
        phishing_score: float | None,
        threshold: float | None,
    ) -> str:

        if prediction == "PHISHING":

            if intelligence_status == "KNOWN_PHISHING":
                return (
                    "PhishGuard classified this URL as PHISHING "
                    "because PhishHunt reported a known threat "
                    "match. Machine learning inference was "
                    "not required for this classification."
                )

            if intelligence_status == "UNAVAILABLE":
                intelligence_sentence = (
                    "PhishHunt was unavailable, so no threat-intelligence "
                    "conclusion could be obtained. "
                )
            else:
                intelligence_sentence = (
                    "PhishHunt did not report a known threat match. "
                )

            return (
                "PhishGuard classified this URL as PHISHING "
                "based on its machine learning prediction. "
                + intelligence_sentence
                + "The ML ensemble score exceeded the configured "
                  "classification threshold."
            )

        if intelligence_status == "UNAVAILABLE":
            return (
                "PhishGuard classified this URL as "
                "BENIGN_CANDIDATE based on its machine learning "
                "result. PhishHunt was unavailable, so no "
                "threat intelligence conclusion could be obtained."
            )

        return (
            "PhishGuard classified this URL as "
            "BENIGN_CANDIDATE. PhishHunt did not report a known "
            "threat match, and the machine learning phishing "
            f"score was {phishing_score:.4f}, below the "
            f"configured threshold of {threshold:.2f}. "
            "This classification does not establish that "
            "the URL is safe."
        )

    # -------------------------------------------------
    # EXPLANATION VALIDATION
    # -------------------------------------------------

    @staticmethod
    def _validate_generated_explanation(
        explanation: str,
        prediction: str,
        intelligence_status: str,
        phishing_score: float | None,
        threshold: float | None,
    ) -> bool:

        if not isinstance(explanation, str):
            return False

        text = explanation.lower().strip()

        if not text:
            return False

        # -------------------------------------------------
        # CHECK 1: THRESHOLD CONTRADICTIONS
        # -------------------------------------------------

        if phishing_score is not None and threshold is not None:

            if prediction == "PHISHING":

                contradictory_phrases = [
                    "below the threshold",
                    "below the configured threshold",
                    "below threshold",
                    "under the threshold",
                    "under the configured threshold",
                    "under threshold",
                    "less than the threshold",
                    "less than the configured threshold",
                    "lower than the threshold",
                    "lower than the configured threshold",
                    "does not exceed the threshold",
                    "did not exceed the threshold",
                ]

                if any(
                    phrase in text
                    for phrase in contradictory_phrases
                ):
                    return False

            elif prediction == "BENIGN_CANDIDATE":

                contradictory_phrases = [
                    "above the threshold",
                    "above the configured threshold",
                    "above threshold",
                    "over the threshold",
                    "over the configured threshold",
                    "greater than the threshold",
                    "higher than the threshold",
                    "exceeds the threshold",
                    "exceeded the threshold",
                ]

                if any(
                    phrase in text
                    for phrase in contradictory_phrases
                ):
                    return False

        # -------------------------------------------------
        # CHECK 2: THREAT INTELLIGENCE ATTRIBUTION
        # -------------------------------------------------

        if intelligence_status == "NOT_FOUND":

            incorrect_attributions = [
                "ml classification as not_found",
                "ml prediction as not_found",
                "machine learning classified as not_found",
                "machine learning prediction is not_found",
            ]

            if any(phrase in text for phrase in incorrect_attributions):
                return False

        if intelligence_status == "UNAVAILABLE":
            false_clean_claims = [
                "phishunt did not report a known threat match",
                "phishunt found no known threat",
                "no known threat match was found by phishunt",
            ]
            if any(phrase in text for phrase in false_clean_claims):
                return False

        return True

    # -------------------------------------------------
    # BUILD EVIDENCE
    # -------------------------------------------------

    @staticmethod
    def _build_evidence(
        intelligence: dict,
        ml: dict | None,
    ) -> list[str]:

        evidence = []

        status = intelligence.get(
            "status",
            "UNAVAILABLE",
        )

        if status == "KNOWN_PHISHING":

            known = intelligence.get("known", {})

            if known.get("in_active_feed"):
                evidence.append(
                    "PhishHunt reported a match in its active feed."
                )

            if known.get("previously_active"):
                evidence.append(
                    "PhishHunt reported a previously active threat match."
                )

            if known.get("in_new_registration_feed"):
                evidence.append(
                    "PhishHunt reported a match in its new-registration feed."
                )

        elif status == "NOT_FOUND":

            evidence.append(
                "PhishHunt did not report a known threat match."
            )

        elif status == "UNAVAILABLE":

            evidence.append(
                "PhishHunt could not provide a result."
            )

        verdict = intelligence.get("verdict")

        if verdict:
            evidence.append(
                f"PhishHunt heuristic verdict: {verdict}."
            )

        if ml is not None:

            score = ml["phishing_score"]
            threshold = ml["threshold"]

            evidence.append(
                f"ML ensemble phishing score: {score:.4f}."
            )

            evidence.append(
                f"ML classification threshold: {threshold:.2f}."
            )

            for model_name, model_score in (
                ml.get("base_model_scores", {}).items()
            ):

                evidence.append(
                    f"{model_name} phishing score: "
                    f"{model_score:.4f}."
                )

        return evidence

    # -------------------------------------------------
    # BUILD LIMITATIONS
    # -------------------------------------------------

    @staticmethod
    def _build_limitations(
        intelligence_status: str,
        ml: dict | None,
    ) -> list[str]:

        limitations = [
            "A classification result does not establish that "
            "a website is completely safe or malicious."
        ]

        if intelligence_status == "NOT_FOUND":

            limitations.append(
                "No known threat match does not mean the URL "
                "is verified safe."
            )

        elif intelligence_status == "UNAVAILABLE":

            limitations.append(
                "Threat intelligence could not be checked successfully."
            )

        if ml is not None:

            limitations.append(
                "The ML phishing score is a model output and is "
                "not necessarily a calibrated real-world probability."
            )

        return limitations

    # -------------------------------------------------
    # BUILD RECOMMENDATIONS
    # -------------------------------------------------

    @staticmethod
    def _build_recommendations(
        prediction: str,
    ) -> list[str]:

        if prediction == "PHISHING":

            return [
                "Do not enter credentials or payment information.",
                "Avoid downloading files from this URL.",
                "Use the official website or application instead.",
            ]

        return [
            "Verify the website's legitimacy before entering sensitive information.",
            "Avoid relying solely on an automated classification.",
        ]

    # -------------------------------------------------
    # MAIN EXPLANATION PIPELINE
    # -------------------------------------------------

    def explain(self, classification: dict) -> SecurityReport:

        prediction = classification["prediction"]
        url = classification["url"]

        intelligence = classification.get("phishunt") or {}

        intelligence_status = intelligence.get(
            "status",
            "UNAVAILABLE",
        )

        ml_data = classification.get("ml")

        # -------------------------------------------------
        # VALIDATE CLASSIFICATION INPUT
        # -------------------------------------------------

        if prediction not in (
            "PHISHING",
            "BENIGN_CANDIDATE",
        ):
            raise ValueError(
                f"Unsupported prediction: {prediction}"
            )

        if intelligence_status not in (
            "KNOWN_PHISHING",
            "NOT_FOUND",
            "UNAVAILABLE",
        ):
            raise ValueError(
                f"Unsupported threat intelligence status: "
                f"{intelligence_status}"
            )

        if prediction == "PHISHING" and (
            intelligence_status == "KNOWN_PHISHING"
        ):
            if ml_data is not None:
                raise ValueError(
                    "Known threat override should not contain ML results."
                )

        if prediction == "PHISHING" and ml_data is None:
            if intelligence_status != "KNOWN_PHISHING":
                raise ValueError(
                    "ML classification data is missing."
                )

        if prediction == "BENIGN_CANDIDATE" and ml_data is None:
            raise ValueError(
                "ML classification data is required."
            )

        # -------------------------------------------------
        # BUILD ML REPORT
        # -------------------------------------------------

        ml_report = None

        if ml_data is not None:

            ml_report = MachineLearningReport(
                prediction=prediction,
                phishing_score=ml_data["phishing_score"],
                threshold=ml_data["threshold"],
                base_model_scores=ml_data.get(
                    "base_model_scores",
                    {},
                ),
            )

        phishing_score = (
            ml_data["phishing_score"]
            if ml_data is not None
            else None
        )

        threshold = (
            ml_data["threshold"]
            if ml_data is not None
            else None
        )

        # -------------------------------------------------
        # BUILD RISK LEVEL
        # -------------------------------------------------

        risk_level = (
            "HIGH"
            if prediction == "PHISHING"
            else "UNKNOWN"
        )

        # -------------------------------------------------
        # BUILD THREAT INTELLIGENCE REPORT
        # -------------------------------------------------

        threat_report = ThreatIntelligenceReport(
            status=intelligence_status,
            verdict=intelligence.get("verdict"),
            known_match=(
                intelligence_status == "KNOWN_PHISHING"
            ),
        )

        # -------------------------------------------------
        # BUILD AUTHORITATIVE EVIDENCE
        # -------------------------------------------------

        evidence = self._build_evidence(
            intelligence,
            ml_data,
        )

        limitations = self._build_limitations(
            intelligence_status,
            ml_data,
        )

        recommendations = self._build_recommendations(
            prediction,
        )

        # -------------------------------------------------
        # BUILD LLM INPUT
        # -------------------------------------------------

        llm_evidence = {
            "url": url,

            "final_prediction": prediction,

            "classification_source": classification.get(
                "classification_source",
                "UNKNOWN",
            ),

            "threat_intelligence": {
                "provider": "PhishHunt",
                "status": intelligence_status,
                "verdict": intelligence.get("verdict"),
                "known_match": (
                    intelligence_status == "KNOWN_PHISHING"
                ),
            },

            "machine_learning": (
                {
                    "prediction": prediction,
                    "phishing_score": phishing_score,
                    "threshold": threshold,
                }
                if ml_data is not None
                else None
            ),

            "evidence": evidence,

            "limitations": limitations,
        }

        # -------------------------------------------------
        # PYTHON FALLBACK
        # -------------------------------------------------

        explanation = self._build_fallback_explanation(
            prediction,
            intelligence_status,
            phishing_score,
            threshold,
        )

        explanation_source = "PYTHON_FALLBACK"

        # -------------------------------------------------
        # GENERATE LLM EXPLANATION
        # -------------------------------------------------

        try:

            raw_response = self.client.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=build_user_prompt(llm_evidence),
            )

            parsed_response = json.loads(raw_response)

            if not isinstance(parsed_response, dict):
                raise ValueError(
                    "LLM response must be a JSON object."
                )

            generated_explanation = parsed_response.get(
                "explanation"
            )

            if (
                not isinstance(generated_explanation, str)
                or not generated_explanation.strip()
            ):
                raise ValueError(
                    "LLM explanation is missing or invalid."
                )

            generated_explanation = (
                generated_explanation.strip()
            )

            # -------------------------------------------------
            # VALIDATE GENERATED EXPLANATION
            # -------------------------------------------------

            is_valid = self._validate_generated_explanation(
                explanation=generated_explanation,
                prediction=prediction,
                intelligence_status=intelligence_status,
                phishing_score=phishing_score,
                threshold=threshold,
            )

            if not is_valid:

                raise ValueError(
                    "LLM explanation contradicts "
                    "classification evidence."
                )

            explanation = generated_explanation
            explanation_source = "LLM"

        except (RuntimeError, ValueError, TypeError) as error:

            print(
                "[PhishGuard LLM] "
                f"Using Python fallback: {error}"
            )

        # -------------------------------------------------
        # BUILD FINAL SECURITY REPORT
        # -------------------------------------------------

        return SecurityReport(
            url=url,

            final_prediction=prediction,

            classification_source=classification.get(
                "classification_source",
                "UNKNOWN",
            ),

            risk_level=risk_level,

            threat_intelligence=threat_report,

            machine_learning=ml_report,

            evidence=evidence,

            limitations=limitations,

            recommendations=recommendations,

            explanation=explanation,

            explanation_source=explanation_source,
        )

