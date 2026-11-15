
import json

from classifier import PhishGuardClassifier
from llm.explainer import PhishGuardExplainer


def main():

    print("=" * 55)
    print("                 PHISHGUARD")
    print("             LLM EXPLANATION TEST")
    print("=" * 55)

    url = input("\nEnter URL to analyse: ").strip()

    if not url:
        print("[Error] URL cannot be empty.")
        return

    # -----------------------------------------
    # STAGE 1: CLASSIFICATION
    # -----------------------------------------

    print("\n[1] Running threat intelligence and ML...")

    classifier = PhishGuardClassifier()

    classification = classifier.classify(url)

    print(
        f"[Classifier] Prediction: "
        f"{classification['prediction']}"
    )

    # -----------------------------------------
    # STAGE 2: LLM EXPLANATION
    # -----------------------------------------

    print("\n[2] Generating security explanation...")

    explainer = PhishGuardExplainer()

    report = explainer.explain(classification)

    # -----------------------------------------
    # STAGE 3: DISPLAY REPORT
    # -----------------------------------------

    print("\n" + "=" * 55)
    print("              SECURITY REPORT")
    print("=" * 55)

    print(report.model_dump_json(indent=4))

    print("\n" + "=" * 55)
    print(
        f"Explanation source: {report.explanation_source}"
    )
    print("=" * 55)


if __name__ == "__main__":
    main()
