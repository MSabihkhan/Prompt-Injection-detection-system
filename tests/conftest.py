import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `from app.api...` imports resolve regardless of cwd/rootdir
sys.path.insert(0, str(ROOT / "app"))  # so `from inference import ...` style imports resolve
