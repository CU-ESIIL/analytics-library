# How to Use the Analytics Library

The ESIIL Analytics Library is a collection of reusable environmental data science workflows. Each page shows an analysis pattern you can copy, run, and adapt.

The core idea is simple:

**Data Library -> Analytics Library -> Outputs**

- Use the [ESIIL Data Library](https://cu-esiil.github.io/data-library/) to find or access data.
- Use this Analytics Library to choose an analysis method.
- Adapt the R or Python function to your dataset.
- Produce a visible output such as a plot, map, model summary, or table.

## What This Library Is

This repository is for analytics workflows. It does not host authoritative datasets and should not duplicate Data Library pages.

Good Analytics Library examples answer questions like:

- How do I summarize a time series?
- How do I fit a regression model?
- How do I cluster observations?
- How do I classify remote-sensing pixels?
- How do I turn environmental data into a plot, map, or decision-ready summary?

## How to Run Examples

Each analytic entry includes:

- A small R example.
- A small Python example.
- A function-based workflow.
- A minimum viable output.
- Tags for searching and browsing.

Most examples use small inline data so you can copy the code block directly into R, RStudio, Python, Jupyter, or a script. When an example links to a public dataset, it should not require API keys or secrets.

## Connecting to the Data Library

Many workflows include a "Use this analysis with these datasets" note. Start there when you want real environmental data.

For example:

- Data Library: find a climate, fire, water, raster, vector, or tabular dataset.
- Analytics Library: select a matching workflow such as forecasting, classification, regression, or cloud masking.
- Output: generate a plot, map, model object, or summary that supports interpretation.

## Adapting an Analytic to New Data

1. Match your data type to the workflow tags, such as `tabular`, `raster`, `time-series`, or `remote-sensing`.
2. Rename your columns or inputs to match the function arguments.
3. Run the example with the included small data first.
4. Replace the example data with your dataset.
5. Keep the returned object so you can inspect, save, or reuse the results.
6. Update interpretation and limitations for your study system.

Prefer small, tested changes. A useful adapted workflow is one that another person can rerun without knowing your local computer setup.
