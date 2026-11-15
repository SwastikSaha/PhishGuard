from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "ml" / "artifacts"
INTERVAL_SECONDS = 3600

ACQUISITION = [
    "phish_url_collector.py",
    "benign_url_collector.py",
    "feature_extractor.py",
]
TRAINING = ["train.py", "meta_learner.py", "evaluate.py"]

def run_script(name: str) -> None:
    path = ROOT / name
    print(f"\n[PhishGuard] Running {name}", flush=True)
    result = subprocess.run([sys.executable, str(path)], cwd=ROOT, check=False)
    if result.returncode:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")
    print(f"[PhishGuard] {name} completed", flush=True)

def run_cycle() -> None:
    started = time.monotonic()
    print("\n" + "=" * 68, flush=True)
    print(f"PHISHGUARD PIPELINE — {datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)
    print("=" * 68, flush=True)

    for script in ACQUISITION:
        run_script(script)

    params = ARTIFACTS / "best_params.json"
    if not params.exists():
        print("\n[PhishGuard] Initial hyperparameter optimization required.", flush=True)
        run_script("optimize.py")
    else:
        print("\n[PhishGuard] Reusing saved Optuna hyperparameters.", flush=True)

    for script in TRAINING:
        run_script(script)

    elapsed = time.monotonic() - started
    print(f"\n[PhishGuard] Pipeline completed in {elapsed / 60:.1f} minutes.", flush=True)

def main() -> None:
    parser = argparse.ArgumentParser(description="PhishGuard acquisition/training pipeline")
    parser.add_argument("--loop", action="store_true", help="Repeat on an hourly cadence")
    parser.add_argument("--interval", type=int, default=INTERVAL_SECONDS)
    args = parser.parse_args()

    pid_file = ROOT / "logs" / "pipeline.pid"

    def stop_signal(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)

    print("[PhishGuard] Pipeline runner started.", flush=True)
    try:
        while True:
            cycle_start = time.monotonic()
            try:
                run_cycle()
            except Exception as exc:
                print(f"\n[PhishGuard] PIPELINE ERROR: {exc}", flush=True)
            if not args.loop:
                break
            remaining = max(0, args.interval - (time.monotonic() - cycle_start))
            print(f"[PhishGuard] Next cycle in {remaining / 60:.1f} minutes.", flush=True)
            time.sleep(remaining)
    except KeyboardInterrupt:
        print("\n[PhishGuard] Shutdown requested.", flush=True)
    finally:
        pid_file.unlink(missing_ok=True)

if __name__ == "__main__":
    main()
