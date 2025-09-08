# Analytics Library Style Guide

This document outlines conventions for adding new analyses to the library.

## File naming
- Place each analysis under an appropriate subdirectory in `docs/`.
- Use short, descriptive, lowercase file names separated by hyphens (e.g. `my-analysis.md`).

## Front matter
Start every file with YAML front matter providing metadata used by the site:

```yaml
---
title: Descriptive title
authors:
  - Your Name
date: YYYY-MM-DD
tags:
  - topic
  - method
---
```

## Required sections
Organize each entry with the following sections:

1. **Description** – Explain the analysis, why it is valuable, the type of data it accepts, and suggest sources from the data library.
2. **Usage Example** – Include a self-contained, copy‑and‑pasteable code snippet that loads a dataset from the data library, performs the analysis, and produces a plot demonstrating the results.
3. **Interpretation and Heuristics** – Provide context for reading the results and guidance on common pitfalls or rules of thumb.

## Writing tips
- Use `#` for the document title and `##` for section headings.
- Prefer concise sentences and active voice.
- Include links to relevant data library entries and external resources when helpful.
- Provide meaningful alt text for all images and plots.
- Keep code blocks executable as written; import all required libraries in the snippet.

Adhering to these guidelines ensures consistent, discoverable, and easy-to-understand analytics examples.
