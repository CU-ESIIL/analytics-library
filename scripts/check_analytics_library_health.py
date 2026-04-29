#!/usr/bin/env python3
"""Health checks for the ESIIL Analytics Library."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MAX_FILE_BYTES = 5 * 1024 * 1024
IGNORED_PARTS = {".git", ".venv", "site", "node_modules", "__pycache__"}

REQUIRED_SECTIONS = [
    "What this analysis does",
    "When to use it",
    "Inputs",
    "R example",
    "Python example",
    "Minimum viable output",
    "Interpretation",
    "Limitations",
    "Tags",
]

DOC_PAGES = {
    "index.md",
    "how-to-use.md",
    "style-guide.md",
    "prompt-log.md",
    "tags.md",
    "innovation-summit-2025.md",
    "how-to-contribute.md",
}

SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"][^'\"]{8,}['\"]"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ghp_[A-Za-z0-9_]{30,}"),
]


def split_front_matter(text: str) -> tuple[dict, str]:
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            raw = text[4:end]
            body = text[end + 5 :]
            if yaml is None:
                return parse_simple_front_matter(raw), body
            try:
                return yaml.safe_load(raw) or {}, body
            except Exception:
                return parse_simple_front_matter(raw), body
    return {}, text


def parse_simple_front_matter(raw: str) -> dict:
    """Small fallback parser for title/authors/date/tags front matter."""
    data: dict[str, object] = {}
    current_key: str | None = None
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- ") and current_key:
            data.setdefault(current_key, [])
            if isinstance(data[current_key], list):
                data[current_key].append(stripped[2:].strip().strip('"'))
            continue
        if ":" in line and not line.startswith(" "):
            key, value = line.split(":", 1)
            current_key = key.strip()
            value = value.strip().strip('"')
            if value:
                data[current_key] = value
            else:
                data[current_key] = []
    return data


def is_analysis_page(path: Path) -> bool:
    rel = path.relative_to(DOCS).as_posix()
    if rel in DOC_PAGES or rel.startswith("topic/"):
        return False
    return path.name == "index.md" and len(path.relative_to(DOCS).parts) >= 2


def check_analysis(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    fm, body = split_front_matter(text)
    warnings: list[str] = []

    for section in REQUIRED_SECTIONS:
        if not re.search(rf"^##\s+{re.escape(section)}\s*$", body, re.MULTILINE):
            warnings.append(f"{path.relative_to(ROOT)}: missing section '{section}'")

    tags = fm.get("tags") or []
    if not tags:
        warnings.append(f"{path.relative_to(ROOT)}: missing front matter tags")
    if not re.search(r"^##\s+Tags\s*$[\s\S]*?(R example|Python example|regression|classification|forecasting|remote-sensing|time-series)", body, re.MULTILINE):
        warnings.append(f"{path.relative_to(ROOT)}: tags section is empty or too thin")

    has_r = bool(re.search(r"```\s*\{?r\}?|```\s*r\b", body, re.IGNORECASE))
    has_python = bool(re.search(r"```\s*\{?python\}?|```\s*python\b", body, re.IGNORECASE))
    if not has_r:
        warnings.append(f"{path.relative_to(ROOT)}: missing R code block")
    if not has_python:
        warnings.append(f"{path.relative_to(ROOT)}: missing Python code block")

    if not re.search(r"\b(function\s*\(|[A-Za-z0-9_.]+\s*<-\s*function\s*\()", body):
        warnings.append(f"{path.relative_to(ROOT)}: missing R function definition")
    if not re.search(r"^\s*def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(", body, re.MULTILINE):
        warnings.append(f"{path.relative_to(ROOT)}: missing Python function definition")

    plot_terms = r"(ggplot|plot\s*\(|plt\.|imshow|geom_|summary\s*\(|print\s*\()"
    if not re.search(plot_terms, body):
        warnings.append(f"{path.relative_to(ROOT)}: no visible plot/map/summary output detected")

    local_path_terms = ["/Users/", "C:\\", "<your_", "your_in_dir"]
    for term in local_path_terms:
        if term in body:
            warnings.append(f"{path.relative_to(ROOT)}: contains placeholder or hidden local path marker '{term}'")

    return warnings


def check_secrets_and_sizes(root: Path) -> tuple[list[str], list[str]]:
    severe: list[str] = []
    warnings: list[str] = []
    for path in root.rglob("*"):
        if any(part in IGNORED_PARTS for part in path.parts) or not path.is_file():
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        rel = path.relative_to(ROOT)
        if size > MAX_FILE_BYTES:
            severe.append(f"{rel}: file is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB")
        if size > 1024 * 1024:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                severe.append(f"{rel}: possible secret detected by pattern '{pattern.pattern}'")
        if path.suffix.lower() in {".csv", ".tif", ".tiff", ".zip", ".nc", ".parquet"} and size > 500_000:
            warnings.append(f"{rel}: data-like file is present; confirm it is intentionally small")
    return severe, warnings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="Treat structural warnings as failures.")
    args = parser.parse_args()

    analysis_pages = sorted(p for p in DOCS.rglob("*.md") if is_analysis_page(p))
    warnings: list[str] = []
    severe: list[str] = []

    if not analysis_pages:
        severe.append("No analysis pages found under docs/*/*/index.md")

    for page in analysis_pages:
        warnings.extend(check_analysis(page))

    size_severe, size_warnings = check_secrets_and_sizes(ROOT)
    severe.extend(size_severe)
    warnings.extend(size_warnings)

    print(f"Analytics pages checked: {len(analysis_pages)}")

    if warnings:
        print("\nWarnings:")
        for item in warnings:
            print(f"- {item}")
    else:
        print("No structural warnings found.")

    if severe:
        print("\nSevere issues:")
        for item in severe:
            print(f"- {item}")

    if severe or (args.strict and warnings):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
