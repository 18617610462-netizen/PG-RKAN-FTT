from __future__ import annotations

import re
from pathlib import Path


FORBIDDEN_PATH_PATTERNS = [
    re.compile(r"[A-Za-z]:[\\/]", re.IGNORECASE),
    re.compile("/" + "Users" + "/"),
]
FORBIDDEN_EXTENSIONS = {".xlsx", ".xls", ".pt", ".pth", ".pkl", ".joblib", ".opju"}
ALLOWED_FILES = {
    Path("examples/demo_input.csv"),
    Path("examples/demo_output.csv"),
    Path("data/sample_10_percent.csv"),
    Path("data/sample_10_percent.xlsx"),
    Path("data/sample_10_percent_summary.csv"),
    Path("released_model/PG-RKAN-FTT_best_model.pt"),
    Path("released_model/PG-RKAN-FTT_preprocessor.joblib"),
    Path("released_model/metrics/PG-RKAN-FTT_metrics_summary.csv"),
    Path("released_model/metrics/PG-RKAN-FTT_metrics_summary.xlsx"),
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    problems: list[str] = []
    for path in root.rglob("*"):
        if path.is_dir() or ".git" in path.parts:
            continue
        rel = path.relative_to(root)
        if path.suffix.lower() in FORBIDDEN_EXTENSIONS and rel not in ALLOWED_FILES:
            problems.append(f"Forbidden file extension: {rel}")
        if path.suffix.lower() in {".py", ".md", ".yaml", ".yml", ".txt"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in FORBIDDEN_PATH_PATTERNS:
                if pattern.search(text):
                    problems.append(f"Hard-coded local path pattern in: {rel}")
    if problems:
        raise SystemExit("\n".join(problems))
    print("Public repository check passed.")


if __name__ == "__main__":
    main()
