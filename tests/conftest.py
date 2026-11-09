import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `from app.api...` imports resolve regardless of cwd/rootdir
sys.path.insert(0, str(ROOT / "app"))  # so `from inference import ...` style imports resolve

# Tests use a stub Detector (see test_api.py) and must never touch the real DagsHub registry
# (network round-trip, needs credentials). Set before app.api.main is imported so its lifespan
# sees it at startup and skips Detector.load entirely.
os.environ["PROMPTGUARD_SKIP_MODEL_LOAD"] = "1"
