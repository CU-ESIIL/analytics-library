#!/usr/bin/env python3
"""Minimal navigation and local link checker for the MkDocs site."""
from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import urlparse, unquote

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MKDOCS = ROOT / "mkdocs.yml"
LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)|!\[[^\]]*\]\(([^)]+)\)")


def iter_nav_items(items):
    if isinstance(items, list):
        for item in items:
            yield from iter_nav_items(item)
    elif isinstance(items, dict):
        for value in items.values():
            if isinstance(value, str):
                yield value
            else:
                yield from iter_nav_items(value)


def iter_nav_items_without_yaml(text: str):
    in_nav = False
    for line in text.splitlines():
        if line.startswith("nav:"):
            in_nav = True
            continue
        if in_nav and line and not line.startswith((" ", "-")):
            break
        if not in_nav:
            continue
        stripped = line.strip()
        if not stripped.startswith("- ") or ":" not in stripped:
            continue
        _, value = stripped.split(":", 1)
        value = value.strip()
        if value and not value.startswith("["):
            yield value


def is_external(target: str) -> bool:
    parsed = urlparse(target)
    return parsed.scheme in {"http", "https", "mailto"}


def resolve_doc_link(source: Path, target: str) -> Path | None:
    target = target.split("#", 1)[0].strip()
    if " " in target and (target.endswith('"') or target.endswith("'")):
        target = target.split(" ", 1)[0].strip()
    if not target or is_external(target):
        return None
    target = unquote(target)
    if target.startswith("/"):
        candidate = DOCS / target.lstrip("/")
    else:
        candidate = source.parent / target
    if candidate.is_dir():
        candidate = candidate / "index.md"
    if candidate.suffix == "":
        md_candidate = candidate.with_suffix(".md")
        if md_candidate.exists():
            return md_candidate
    return candidate


def main() -> int:
    failures: list[str] = []

    if not DOCS.joinpath("index.md").exists():
        failures.append("homepage docs/index.md is missing")

    if yaml is None:
        nav_targets = list(iter_nav_items_without_yaml(MKDOCS.read_text(encoding="utf-8")))
    else:
        cfg = yaml.safe_load(MKDOCS.read_text(encoding="utf-8")) or {}
        nav_targets = list(iter_nav_items(cfg.get("nav", [])))

    for nav_target in nav_targets:
        if is_external(nav_target):
            continue
        target_path = DOCS / nav_target
        if target_path.is_dir():
            target_path = target_path / "index.md"
        if not target_path.exists():
            failures.append(f"mkdocs nav target missing: {nav_target}")

    for md in DOCS.rglob("*.md"):
        text = md.read_text(encoding="utf-8")
        for match in LINK_RE.finditer(text):
            raw_target = match.group(1) or match.group(2)
            if raw_target.startswith("<") and raw_target.endswith(">"):
                raw_target = raw_target[1:-1]
            resolved = resolve_doc_link(md, raw_target)
            if resolved is not None and not resolved.exists():
                failures.append(f"{md.relative_to(ROOT)} links to missing local target: {raw_target}")

    if failures:
        print("Navigation/link failures:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Navigation checks passed: homepage, MkDocs nav, and local links resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
