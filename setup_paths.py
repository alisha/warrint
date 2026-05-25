"""Path setup for warrint scripts. Import this first in entry scripts."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "optar"))
