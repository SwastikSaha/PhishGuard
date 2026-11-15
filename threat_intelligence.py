import requests
from urllib.parse import urlparse

PHISHUNT_API = "https://phishunt.io/api/v1/analyze"

TIMEOUT = (5, 15)


def check_phishunt(url: str) -> dict:
    """
    Query PhishHunt for every submitted URL.

    Status values:
        KNOWN_PHISHING - URL matched a known threat record.
        NOT_FOUND      - No known threat record was found.
        UNAVAILABLE    - API request or response failed.

    Important:
        PhishHunt's heuristic verdict is preserved separately.
        A heuristic verdict alone does not trigger KNOWN_PHISHING.
    """

    try:
        response = requests.get(
            PHISHUNT_API,
            params={"url": url},
            headers={
                "User-Agent": "PhishGuard/1.0"
            },
            timeout=TIMEOUT,
        )

        response.raise_for_status()

        data = response.json()

        if not isinstance(data, dict):
            raise ValueError("Invalid PhishHunt response.")

        known = data.get("known")

        if not isinstance(known, dict):
            raise ValueError("Missing or invalid known section.")

        verdict = str(data.get("verdict", "")).lower()

        active = bool(known.get("in_active_feed", False))
        previous = bool(known.get("previously_active", False))
        new_registration = bool(
            known.get("in_new_registration_feed", False)
        )

        # Only actual known-threat matches trigger the override.
        known_match = active or previous or new_registration

        live_analysis = data.get("live_analysis")

        risk_score = (
            live_analysis.get("url_risk_score")
            if isinstance(live_analysis, dict)
            else None
        )

        return {
            "status": (
                "KNOWN_PHISHING"
                if known_match
                else "NOT_FOUND"
            ),
            "source": "PhishHunt",
            "verdict": verdict,
            "known": {
                "in_active_feed": active,
                "previously_active": previous,
                "in_new_registration_feed": new_registration,
            },
            "record": known.get("record"),
            "live_analysis": live_analysis,
            "risk_score": risk_score,
        }

    except requests.RequestException as error:
        return {
            "status": "UNAVAILABLE",
            "source": "PhishHunt",
            "error": str(error),
        }

    except (ValueError, TypeError, KeyError) as error:
        return {
            "status": "UNAVAILABLE",
            "source": "PhishHunt",
            "error": str(error),
        }
