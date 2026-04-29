# Analytics Library Agent Guide

This repository hosts reusable analytics workflows for environmental data science. It is the companion to the ESIIL Data Library: the Data Library helps users get data, and the Analytics Library helps users analyze data.

## Purpose

- Host analysis workflows, not datasets.
- Prefer examples that operate on small inline data or data retrieved from public, keyless sources.
- Link to Data Library entries when a workflow naturally starts from one of those datasets.
- Preserve existing useful content while improving reproducibility, structure, and discoverability.

## Required Analytic Structure

Every analytic entry should use this structure:

```markdown
# Analysis Name
## What this analysis does
## When to use it
## Inputs
## R example
## Python example
## Minimum viable output
## Interpretation
## Limitations
## Tags
```

## Code Expectations

Every analytic must include both R and Python implementations.

Examples must be:

- Copy-paste runnable.
- Free of API keys, tokens, passwords, and secrets.
- Free of hidden local file paths.
- Small enough to run quickly on a normal laptop.
- Built around functions, not one-off notebook state.

Functions should:

- Take simple inputs such as `data`, `aoi`, `time_range`, or plain vectors/tables.
- Run without manual setup beyond installing common public packages.
- Produce a visible result such as a plot, map, or summary.
- Return a useful object for reuse.

## Reproducibility Standards

- Favor clarity over cleverness.
- Use minimal dependencies and name them explicitly.
- Include small example data directly in the page when possible.
- Keep outputs deterministic by setting seeds where randomness is used.
- Avoid downloads that require accounts, API keys, cookies, or secrets.
- Avoid large generated files in the repository.

## Tagging

Use Google-like discoverability. Include relevant tags from these families:

- Method tags: `regression`, `clustering`, `classification`, `forecasting`, `spatial-statistics`, `machine-learning`.
- Data type tags: `raster`, `tabular`, `time-series`, `vector`, `remote-sensing`.
- Domain tags: `climate`, `ecology`, `fire`, `water`, `biodiversity`.
- Workflow tags: `R example`, `Python example`, `beginner`, `reproducible`.
- Synonym tags: include common alternate names users may search for.

## Repository Hygiene

- Do not turn this repository into a data host.
- Do not duplicate Data Library dataset documentation.
- Do not add AI, sustainability, or working-group lifecycle essays unless they directly support an analytic workflow.
- Do not remove existing content unless it is broken or superseded by a tested replacement.
- Before finishing, run:

```bash
mkdocs build
python scripts/check_analytics_library_health.py
python scripts/check_navigation.py
```
