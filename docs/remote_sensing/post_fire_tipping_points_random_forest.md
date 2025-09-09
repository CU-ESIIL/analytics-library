---
title: "Burning Boundaries: Random Forest Early Warnings for Post-Fire Collapse"
authors:
  - ESIIL Analytics Team
date: 2025-09-09
tags:
  - remote-sensing
  - sentinel-2
  - time-series
  - fire
  - tipping-points
  - machine-learning
  - innovation-summit-2025
---

# Burning Boundaries: Random Forest Early Warnings for Post-Fire Collapse

## Introduction

Wildfires can push ecosystems across **tipping points**—for example, from forests to shrublands—when vegetation fails to recover after disturbance. This analytics entry describes a reproducible workflow that:

* Aligns **fire events** (from the FIRED database) with **satellite vegetation time series** (Sentinel‑2 NDVI);
* Extracts **early‑warning signals (EWS)** from those time series; and
* Trains a **Random Forest (RF)** classifier to flag pixels that show **persistent post‑fire vegetation collapse**.

The approach integrates three research strands: fire event mapping (FIRED), critical‑transition theory (EWS), and remote‑sensing machine learning.

## Data Sources

**Fire Events: FIRED (Fire Events Data)**

* Source: Andela et al. (2019, *Nature Ecology & Evolution*); Balch et al. (2020, CU Scholar dataset).
* Content: Event‑level and daily polygons delineating individual wildfires globally.
* Usage here: Event polygons for CONUS + Alaska, including ignition date (`ig_date`) and fire perimeter geometry.
* Format: GeoPackage or Shapefile; read into a GeoDataFrame (EPSG:4326).

**Vegetation Time Series: Sentinel‑2 Level‑2A NDVI**

* Coverage: Mid‑2015 to present.
* Resolution: 10 m (red and near‑infrared bands).
* Access: SpatioTemporal Asset Catalog (STAC) APIs (e.g., Earth‑Search by Element84).
* Representation in code: 3‑D array `(time, height, width)` containing NDVI values.

> Optional: Other indices (e.g., Enhanced Vegetation Index, Normalized Burn Ratio) and other sensors (e.g., Landsat, PlanetScope) can be incorporated similarly.

## Why This Matters

* **Ecological resilience**: Provides evidence on whether ecosystems exhibit critical transitions after fire.
* **Management & restoration**: Identifies high‑risk areas where recovery is unlikely without intervention.
* **Theory into practice**: Tests whether statistical early‑warning indicators (e.g., variance, autocorrelation, trend) can predict persistent vegetation collapse.

## Workflow (Step‑by‑Step)

1. **Select fires** from FIRED (e.g., `ig_year >= 2018`, `tot_ar_km2 > 10`).
2. **Build time windows** around ignition date: `pre_days` (e.g., 120) before ignition, `post_days` (e.g., 180) after ignition, split into rolling windows (`window_days`, `step_days`).
3. **Fetch NDVI stacks** from Sentinel‑2 via STAC for each window and the fire’s bounding box, with a cloud cover filter (`cloud_lt`).
4. **Compute pre vs post means** of NDVI for each pixel.
5. **Rasterize the fire polygon** to the NDVI grid to identify burned pixels.
6. **Label pixels**: collapse = 1 if NDVI drop exceeds a threshold (relative or absolute); non‑collapse = 0 otherwise.

   * Relative drop:
     $\text{rel\_drop} = \frac{\text{pre\_mean} - \text{post\_mean}}{|\text{pre\_mean}| + \varepsilon}$
     If `rel_drop > \theta` (e.g., \(\theta = 0.30\)) inside the fire scar, label = 1.
   * Absolute drop: If post – pre < threshold (e.g., –0.20), label = 1.
7. **Extract early‑warning features** from each pixel’s NDVI time series:

   * *Slope*: overall trend;
   * *Variance*: volatility over time;
   * *Variance ratio*: short vs long window variance;
   * *Autocorrelation (AC1)*: lag‑1 memory;
   * *Last, Δ(last–first), minimum, maximum*: trajectory summary.
8. **Train a Random Forest classifier** (Breiman 2001) with grouped cross‑validation by fire (GroupKFold) to avoid leakage across pixels from the same event.
9. **Evaluate performance** using ROC‑AUC, average precision (PR‑AUC), precision, recall, and F1 for the positive (collapse) class.
10. **Interpret outputs**: feature importances, per‑fire summaries, and optional spatial maps of predicted collapse.

## Mathematical Framing

We treat NDVI as a proxy for green biomass and canopy condition. Persistent collapse is operationalized as a significant NDVI drop post‑fire that does not rebound within the defined window.

* **Label rule**: collapse if relative or absolute NDVI decline exceeds a threshold.
* **EWS features** capture time‑series properties theorized to increase near critical transitions (Scheffer et al. 2009; Dakos et al. 2012).
* **Model**: Random Forest learns nonlinear mappings from feature space → probability of collapse.

## Model & Validation

* **Classifier**: Random Forest with \~500 trees, class‑weighted, `min_samples_leaf = 2`.
* **Grouping**: GroupKFold by fire to prevent over‑optimistic results.
* **Metrics**: ROC‑AUC, PR‑AUC, positive‑class precision, recall, and F1.
* **Outputs**: bar plot of feature importances; summary tables of per‑fire performance and data coverage.

## Example Use Case

**Research question**: Within large fires from 2018–2023, where did vegetation most likely transition to a persistent low‑NDVI state?
Steps:

1. Filter FIRED events based on size and date.
2. Build dataset with `rel_drop_threshold=0.30`, `cloud_lt=20` (relax to 60 if imagery is sparse).
3. Train Random Forest with grouped cross‑validation.
4. Inspect performance and feature importance; map collapse risk within each burn.
5. Export summaries to guide restoration prioritization.

## Limitations & Best Practices

* **Label uncertainty**: NDVI collapse does not guarantee irreversible ecological state change; field data are crucial for validation.
* **Window design**: At least 2–3 windows pre‑ and post‑fire are needed to avoid mistaking short‑term scorch for collapse.
* **Cloud/seasonal effects**: Clouds and seasonal phenology can bias NDVI features; seasonal adjustment is recommended.
* **Domain shift**: Models trained on one ecoregion or period may not generalize elsewhere; always use grouped CV and out‑of‑region testing.

## References

* Andela, N., Morton, D. C., Giglio, L., Chen, Y., van der Werf, G. R., Kasibhatla, P. S., DeFries, R. S., Collatz, G. J., Hantson, S., Kloster, S., Bachelet, D., Forrest, M., Lasslop, G., Li, F., Mangeon, S., Melton, J. R., Yue, C., & Randerson, J. T. (2019). The Global Fire Atlas of individual fire size, duration, speed and direction. *Nature Ecology & Evolution*, 3, 1494–1502. [https://doi.org/10.1038/s41559-019-0386-1](https://doi.org/10.1038/s41559-019-0386-1)
* Balch, J. K., et al. (2020). FIRED: a global fire event database for the analysis of fire regimes, patterns, and drivers. CU Scholar Dataset.
* Scheffer, M., Bascompte, J., Brock, W. A., Brovkin, V., Carpenter, S. R., Dakos, V., Held, H., van Nes, E. H., Rietkerk, M., & Sugihara, G. (2009). Early‑warning signals for critical transitions. *Nature*, 461, 53–59. [https://doi.org/10.1038/nature08227](https://doi.org/10.1038/nature08227)
* Dakos, V., Carpenter, S. R., Brock, W. A., Ellison, A. M., Guttal, V., Ives, A. R., Kéfi, S., Livina, V., Seekell, D. A., van Nes, E. H., & Scheffer, M. (2012). Methods for detecting early warnings of critical transitions in time series: A review. *PLOS ONE*, 7(7), e41010. [https://doi.org/10.1371/journal.pone.0041010](https://doi.org/10.1371/journal.pone.0041010)
* Kéfi, S., Guttal, V., Brock, W. A., Carpenter, S. R., Ellison, A. M., Livina, V. N., Seekell, D. A., van Nes, E. H., & Scheffer, M. (2014). Early warning signals of ecological transitions: methods for spatial patterns. *Philosophical Transactions of the Royal Society B*, 370(1659), 20130283. [https://doi.org/10.1098/rstb.2013.0283](https://doi.org/10.1098/rstb.2013.0283)
* Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. [https://doi.org/10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)
* Hesketh, M., et al. (2021). Detecting post‑fire vegetation recovery using remote sensing and machine learning. *Remote Sensing of Environment*, 257, 112210. [https://doi.org/10.1016/j.rse.2020.112210](https://doi.org/10.1016/j.rse.2020.112210)
* Liu, X., et al. (2020). Remote sensing‑based early warning of vegetation degradation. *Ecological Indicators*, 115, 106764. [https://doi.org/10.1016/j.ecolind.2020.106764](https://doi.org/10.1016/j.ecolind.2020.106764)
