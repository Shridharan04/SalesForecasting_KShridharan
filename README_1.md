# Sales Forecasting & Demand Intelligence System

End-to-end sales forecasting and demand intelligence project built for the Technosignia Data Science & Analytics internship (Week 3–4). Uses the Superstore Sales dataset (2015–2018) to forecast future demand, detect anomalous sales patterns, and segment products by demand behavior.

**Live Dashboard:** [salesforecastingkshridharan-rosxvsi6bq6j4ypbe4eu3e.streamlit.app](https://salesforecastingkshridharan-rosxvsi6bq6j4ypbe4eu3e.streamlit.app)

## What's in this repo

| File | Description |
|---|---|
| `analysis.ipynb` | Full analysis notebook — EDA, time series decomposition, 3 forecasting models, segment forecasting, anomaly detection, product clustering |
| `app.py` | Streamlit dashboard (4 pages: Sales Overview, Forecast Explorer, Anomaly Report, Product Demand Segments) |
| `summary.docx` | 2-page executive report for non-technical stakeholders |
| `train.csv` | Superstore Sales dataset |
| `forecast_results.csv`, `anomaly_results.csv`, `cluster_results.csv` | Precomputed model outputs consumed by the dashboard |
| `requirements.txt` | Python dependencies for the dashboard |
| `charts/` | All generated chart images |

## Project Summary

- **Forecasting:** Compared SARIMA, Facebook Prophet, and XGBoost on 45 months of training data, evaluated against 3 held-out months. **SARIMA performed best** (MAPE ≈ 12.8%), likely because classical statistical models with strong seasonal/trend structure outperform more flexible ML models when training data is limited (~45 monthly points).
- **Anomaly Detection:** Isolation Forest and Z-score methods flagged 16 unusual weeks combined, with holiday-season spikes (Black Friday / December) as the most consistent pattern; the two methods agreed on only 1 week, reflecting their different definitions of "unusual" (global outlier vs. local trend deviation).
- **Product Segmentation:** K-Means clustering (k=4) grouped 17 sub-categories into four demand profiles — High Volume/Core Revenue, Low Volume/Steady Growth, Explosive Growth/High Value (Copiers), and Declining Demand/High Value (Machines) — each with a distinct recommended stocking strategy.

## Tech Stack

Python · Pandas · Statsmodels (SARIMA) · Prophet · XGBoost · Scikit-learn (Isolation Forest, K-Means, PCA) · Streamlit · Matplotlib

## Author

K Shridharan | [LinkedIn](https://linkedin.com/in/kshridharan) | [GitHub](https://github.com/Shridharan04)
