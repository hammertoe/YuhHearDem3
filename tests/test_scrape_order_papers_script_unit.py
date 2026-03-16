import sys
from pathlib import Path
import runpy


def test_scrape_order_papers_script_bootstraps_repo_root():
    repo_root = Path(__file__).resolve().parents[1]
    original_path = list(sys.path)
    try:
        sys.path = [p for p in sys.path if p not in {str(repo_root), ""}]
        script_path = repo_root / "scripts" / "scrape_order_papers.py"
        runpy.run_path(str(script_path))
        assert str(repo_root) in sys.path
    finally:
        sys.path = original_path
