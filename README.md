# ESIIL Analytics Library

The Analytics Library is the analysis companion to the [ESIIL Data Library](https://cu-esiil.github.io/data-library/).

- Data Library: how to access data.
- Analytics Library: what to do with data.

This repository hosts reusable, copy-paste runnable analytics workflows with R and Python examples, function-based code, tags, and minimum viable outputs.

## Local Checks

```bash
mkdocs build
python scripts/check_analytics_library_health.py
python scripts/check_navigation.py
npm install
npx playwright install chromium
python -m http.server 8000 --directory site
```

With the local server running, use a second terminal for:

```bash
npm test
```

In a local environment without `python` or `mkdocs` on PATH, use a virtual environment and run the equivalent commands through that environment.

See [Site Health](docs/site-health.md) for what the checks cover and how to interpret failures.
