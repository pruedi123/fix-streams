# Fixed Income Inflation Erosion

A Streamlit app that shows how inflation erodes the purchasing power of a fixed income stream over time, and provides tools to plan a reserve strategy that maintains real spending power.

## What It Does

- **Visualizes inflation erosion** — Compare nominal income vs. real (inflation-adjusted) income over any historical period from 1926 to 2025 using actual US CPI data.
- **Reserve fund strategy** — Set a real spending target below your income. The surplus builds a reserve fund that covers future shortfalls when inflation outpaces your income.
- **Solver** — Finds the maximum real spending target you can sustain for the entire period, for both your selected period and the worst historical inflation period of the same length.

## Key Controls

| Control | Description |
|---------|-------------|
| **Annual income** | Your fixed nominal income per year |
| **Start month / year** | When the analysis period begins |
| **Number of years** | How long the income stream lasts |
| **COLA rate** | Annual cost-of-living raise applied to income (0% = no raises) |
| **Real spending target** | Desired annual spending in today's dollars |
| **Reserve return spread** | Extra return the reserve fund earns above inflation |
| **Solve Max Sustainable Target** | Finds the highest spending that reserves can sustain through the full period |

## Running Locally

```bash
pip install streamlit pandas plotly
streamlit run app.py
```

## Data

`cpi_data_monthly.csv` contains US Consumer Price Index monthly returns from January 1926 through December 2025.

## Live App

Deployed on [Streamlit Cloud](https://fix-streams.streamlit.app).
