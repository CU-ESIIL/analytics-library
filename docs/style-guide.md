# Analytics Library Style Guide

This guide aligns Analytics Library pages with the ESIIL/OASIS visual system while keeping the site focused on analysis.

## Color Tokens

Use these shared tokens in CSS and visual assets:

```css
--esiil-primary-blue: #234A65;
--esiil-accent-blue: #42BCDC;
--esiil-accent-green: #007135;
--esiil-body-text: #161A19;
--esiil-gray-relief: #E3E3E3;
```

## Visual Rules

- Use square tiles for browsable topics, analysis cards, and repeated items.
- Keep layouts flat, direct, and screen-print inspired.
- Do not use 3D effects, perspective, glassmorphism, or heavy shadows.
- Use consistent spacing between sections, code blocks, and cards.
- Use strong contrast and restrained color.
- Keep diagrams and plots legible in both light and dark mode.

## Buttons and Links

- Buttons should be rectangular.
- Button text should be bold.
- Use a blue-to-green gradient where a primary action needs emphasis.
- Use plain text links for supporting references and citations.

## Analytic Page Structure

Every analytic entry should include:

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

## Front Matter

Start each analytic with YAML front matter:

```yaml
---
title: Descriptive title
authors:
  - Your Name
date: YYYY-MM-DD
tags:
  - method-tag
  - data-type-tag
  - domain-tag
  - R example
  - Python example
---
```

## Code Style

- Include both R and Python.
- Define a function in each language.
- Use small example data directly in the page when possible.
- Produce a visible plot, map, or summary.
- Return a useful object from the function.
- Avoid secrets, API keys, hidden local paths, and manual setup.

## Tagging

Use tags that help people search the way they think:

- Method: `regression`, `clustering`, `classification`, `forecasting`, `spatial-statistics`.
- Data type: `raster`, `tabular`, `time-series`, `vector`, `remote-sensing`.
- Domain: `climate`, `ecology`, `fire`, `water`, `biodiversity`.
- Workflow: `R example`, `Python example`, `beginner`, `reproducible`.
- Synonyms: add alternate terms such as `prediction`, `segmentation`, `change-detection`, or `trend-analysis` when useful.

## Writing Style

- Lead with what the analysis does and when to use it.
- Keep prose concise and practical.
- Link to Data Library entries instead of repeating their dataset documentation.
- Put limitations close to interpretation so users understand uncertainty.
- Prefer simple, reusable functions over clever one-off scripts.
