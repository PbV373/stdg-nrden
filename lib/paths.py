"""Project root resolution for stable data/config paths regardless of CWD."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def rel_to_root(*parts: str) -> str:
    return str(PROJECT_ROOT.joinpath(*parts))
