---
title: Detecting Breakpoints and Forecasting with Autoregressive Models
authors:
  - ESIIL Analytics Team
date: 2025-09-08
tags:
  - time-series
  - forecasting
  - breakpoints
---

# Detecting Breakpoints and Forecasting with Autoregressive Models
ESIIL Analytics Team

## Description
Autoregressive (AR) models capture temporal dependence in regularly sampled data by expressing each value as a function of its predecessors. By pairing AR models with change point detection algorithms, analysts can identify structural breaks—periods where the generating process shifts—and then forecast future behavior from the most recent stable regime. Typical inputs are one-dimensional time series such as atmospheric measurements, economic indicators, or sensor readings. Comparable series are available in the ESIIL data library, for example monthly atmospheric CO₂ concentrations.

## Usage Example
```python
import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt
from statsmodels.tsa.ar_model import AutoReg
import statsmodels.api as sm

# Load monthly CO2 data
data = sm.datasets.co2.load_pandas().data
co2 = data['co2'].resample('M').mean().fillna(method='ffill')

# Detect breakpoints using the PELT algorithm
signal = co2.values
detector = rpt.Pelt(model="l2").fit(signal)
breaks = detector.predict(pen=10)

# Fit AR model to most recent segment and forecast next 12 months
start = breaks[-2] if len(breaks) > 1 else 0
train = co2.iloc[start:]
model = AutoReg(train, lags=12).fit()
forecast = model.predict(start=len(train), end=len(train)+11)
forecast.index = pd.date_range(co2.index[-1] + pd.offsets.MonthBegin(1), periods=12, freq='M')

# Plot observed data, breakpoints, and forecast
fig, ax = plt.subplots()
co2.plot(ax=ax, label='observed')
for b in breaks[:-1]:
    ax.axvline(co2.index[b], color='red', linestyle='--', alpha=0.7)
forecast.plot(ax=ax, label='forecast')
ax.set_ylabel('CO₂ (ppm)')
ax.legend()
plt.show()
```
The plot shows detected breakpoints as dashed red lines and the twelve-month forecast in blue.

## Interpretation and Heuristics
- Breakpoints often align with interventions or regime changes. Investigate the context around detected points before drawing conclusions.
- The penalty parameter in the change point algorithm controls sensitivity; lower values detect more breaks but risk noise.
- Forecasts assume the process remains consistent after the last break. Refit models when new breaks emerge or residuals grow.
