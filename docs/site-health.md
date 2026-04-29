# Site Health

The site health system checks whether a person can build, open, and navigate the Analytics Library without hitting obvious problems.

## What It Checks

- `mkdocs build` completes.
- Analytics pages follow the required R/Python/function/output/tag structure.
- Local Markdown links and MkDocs navigation targets resolve.
- The built site homepage loads in Chromium through Playwright.
- Core navigation links open pages with visible body content.
- Key pages do not show obvious 404 text.

## What It Does Not Check

- It does not run data pipelines, notebooks, or long analyses.
- It does not validate scientific results.
- It does not test every browser, viewport, or edge case.
- It does not fail on minor content-quality warnings unless the site is clearly broken.

## Run Locally

Install Python and Node dependencies, then run:

```bash
mkdocs build
python scripts/check_analytics_library_health.py
python scripts/check_navigation.py
npm install
npx playwright install chromium
python -m http.server 8000 --directory site
```

In a second terminal:

```bash
npm test
```

## Interpreting Results

- MkDocs build failures usually mean the site cannot be published.
- Analytics health failures point to missing R/Python examples, functions, outputs, tags, secrets, or large files.
- Navigation failures point to broken local links or missing MkDocs nav targets.
- Playwright failures usually mean the built site cannot be loaded or navigated like a normal user would expect.

Warnings are there to guide cleanup. They should not become brittle rules unless they represent real breakage.
