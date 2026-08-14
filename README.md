# Complaint Volume Forecasting (90-Day Horizon)

Reproducible workflow to forecast daily complaints for the next 90 days using STL decomposition and HistGradientBoosting models.

Repo: [UniCandice/Technical-Assessment-LiyuanLIU](https://github.com/UniCandice/Technical-Assessment-LiyuanLIU)

## Project layout

```text
Technical-Assessment-LiyuanLIU/
├── complaints_forecast_assessment.ipynb   # main analysis notebook
├── data/                                  # raw Excel inputs
├── src/forecasting.py                     # reusable modelling helpers
├── results/                               # tables, figures, forecasts
├── requirements.txt
└── README.md
```

## What the notebook does

1. Load and clean daily complaint records
2. EDA (quality checks, weekly/monthly patterns, correlations, STL + FFT)
3. Compare candidate models against naive baselines on a 90-day holdout:
   - `Naive_LastValue`
   - `SeasonalNaive_Weekly`
   - `STL_ComponentHGB_SelectedFeatures`
   - `HGB_RawTableFeatures`
4. Retrain the winning candidate and produce a 90-day deterministic forecast
5. Add residual-bootstrap fluctuation (median + P10-P90 band) for uncertainty communication

## Setup

```bash
# clone
git clone https://github.com/UniCandice/Technical-Assessment-LiyuanLIU.git
cd Technical-Assessment-LiyuanLIU

# create environment (recommended)
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

## Run

1. Put the Excel file in `data/` (or keep the provided copy):
   - `Principle_Data_Scientist_Tech_Assessment.xlsx`
2. Open the notebook from the **project root** (important so `src/` imports work):

```bash
jupyter notebook complaints_forecast_assessment.ipynb
```

3. Run all cells top to bottom.

Outputs are written to `results/`, including:

- `complaints_forecast_next_90_days.csv`
- `model_comparison.csv`
- `forecast_with_fluctuation_band.png`

## How to interpret the final forecast

- **Orange line:** deterministic planning forecast from the holdout winner
- **Blue median + P10-P90 band:** residual-bootstrap day-to-day uncertainty around that structure
- Use orange for capacity planning; use the band to communicate uncertainty

## Notes

- Run the notebook with working directory = project root.
- If the Excel file is locked/open in Excel, the notebook falls back to `Principle_Data_Scientist_Tech_Assessment_copy.xlsx`.
- `centered_7d_mean` is EDA-only (future leakage) and is not used in modelling.

## Limitations

- **Single holdout:** the winner is chosen on one 90-day window; the margin over last-value naive is modest (~1.4 MAE). Rolling-origin backtesting is not included.
- **Uncertainty band:** residual bootstrap is i.i.d. (ignores residual autocorrelation), uses only 30 paths, does not widen with horizon, and excludes model/trend uncertainty. `vol_scale` also mixes difference-std with residual-level-std, which inflates the scale.
- **Known future calendar:** bank holidays in the forecast window are knowable in advance but the component model persists the last observed flag (New Year's Day at the start of this forecast is unmodelled).
- **Recursive forecasts:** both candidates feed their own predictions back into rolling features, so volatility features decay over the horizon; the residual bootstrap is a patch for that artifact rather than a diagnosis of it.
