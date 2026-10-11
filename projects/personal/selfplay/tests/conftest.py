import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for sub in ("src",):
    path = str(ROOT / sub)
    if path not in sys.path:
        sys.path.insert(0, path)
