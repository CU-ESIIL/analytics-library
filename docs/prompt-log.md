# Prompt Log

This log records major repository modernization prompts and structural updates so future maintainers can understand why the site changed.

## 2026-04-29: OASIS Analytics Library Modernization

Codex prompt: "Modernize Analytics Library to Match OASIS Data Library Standards."

Major updates requested:

- Clarify the Analytics Library purpose: reusable analytics workflows, not data hosting.
- Add contributor rules in `AGENTS.md`.
- Add this prompt log.
- Add an ESIIL/OASIS shared style guide.
- Add "How to use the Analytics Library" documentation.
- Standardize analytic entry sections.
- Require R and Python examples with a function and minimum viable output.
- Add a discoverability-focused tagging system.
- Add health checks for analytic structure, functions, plots, secrets, and large files.
- Add navigation checks and a site health CI workflow.
- Strengthen links from Data Library datasets to Analytics Library outputs.

Implementation notes:

- Existing analytic content should be preserved unless broken.
- Compact runnable examples should sit near the top of each analytic page.
- Longer notebooks or tutorials may remain as extended workflow details after the standard sections.
