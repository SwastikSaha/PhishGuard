import argparse

from classifier import PhishGuardClassifier


def main():
    parser = argparse.ArgumentParser(
        description="PhishGuard Live URL Classifier"
    )

    parser.add_argument(
        "url",
        help="URL to classify",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    args = parser.parse_args()

    try:
        classifier = PhishGuardClassifier()

        result = classifier.classify(
            args.url,
            threshold=args.threshold,
        )

        print("\n" + "=" * 50)
        print("                 PHISHGUARD")
        print("=" * 50)

        print(f"\nURL: {result['url']}")

        intelligence = result["phishunt"]

        print("\n[Threat Intelligence]")
        print(f"PhishHunt: {intelligence['status']}")

        if intelligence["status"] == "UNAVAILABLE":
            print(f"Reason: {intelligence.get('error')}")

        elif intelligence["status"] == "KNOWN_PHISHING":
            print(f"Verdict: {intelligence.get('verdict')}")

            for key, value in intelligence["known"].items():
                print(f"{key}: {value}")

        else:
            print("No known threat match reported.")

        ml = result["ml"]

        if ml is not None:
            print("\n[Machine Learning]")

            for model, score in ml["base_model_scores"].items():
                print(f"{model:<15}: {score:.4f}")

            print("-" * 50)
            print(f"Phishing score: {ml['phishing_score']:.4f}")
            print(f"Threshold:      {ml['threshold']:.2f}")

        else:
            print("\n[Machine Learning]")
            print("Skipped: PhishHunt identified a known threat.")

        print("\n" + "=" * 50)
        print(f"CLASSIFICATION: {result['prediction']}")
        print(f"SOURCE:          {result['classification_source']}")
        print("=" * 50)

        if result["prediction"] == "BENIGN_CANDIDATE":
            print(
                "\nWarning: No known threat was identified, "
                "but this does not guarantee the URL is safe."
            )

    except (ValueError, FileNotFoundError) as error:
        print(f"\n[Error] {error}")

    except Exception as error:
        print(f"\n[Unexpected error] {error}")


if __name__ == "__main__":
    main()
