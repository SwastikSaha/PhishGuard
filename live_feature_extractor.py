
from feature_extractor import extract_url_features


def extract_live_features(url: str) -> dict:
    """
    Extract URL features using the same implementation
    used by the training pipeline.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL cannot be empty.")

    return extract_url_features(url.strip())
