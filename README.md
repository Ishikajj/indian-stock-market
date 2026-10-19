# Indian Stock Market — Evidence-Based Investment Analysis

A research repository comparing passive index investing (Nifty 50 / Sensex) against
active mutual funds, gold, and government bonds in India — with Fama-French factor
analysis and portfolio optimization.

**Key question:** Should a 10-year retail investor choose an SIP into an index fund,
an active mutual fund, gold, G-Secs, or a factor portfolio?

---

## Repository structure

```
indian_stock_market/
├── notebooks/
│   └── india_market_analysis.ipynb   ← main analysis (start here)
├── src/
│   ├── data_loader.py   — yfinance + mftool downloaders, RBI repo rate series
│   ├── metrics.py       — CAGR, Sharpe, Sortino, drawdown, SIP simulator
│   ├── factors.py       — Fama-French/Carhart factor construction & OLS regressions
│   ├── tax_utils.py     — Indian LTCG/STCG/debt tax calculations (Budget 2024)
│   └── visualization.py — reusable matplotlib/seaborn plot functions
├── data/
│   ├── raw/             — place manually downloaded CSVs here
│   └── processed/       — charts saved here by the notebook
├── pyproject.toml
└── README.md
```

---

## Setup

```bash
# Install uv if needed
curl -Ls https://astral.sh/uv/install.sh | sh

# Install all dependencies
uv sync

# Launch JupyterLab
uv run jupyter lab notebooks/india_market_analysis.ipynb
```

---

## Data that downloads automatically

| Data | Source | Fetched by |
|------|--------|-----------|
| Nifty 50 (`^NSEI`) | Yahoo Finance | `download_prices()` |
| BSE Sensex (`^BSESN`) | Yahoo Finance | `download_prices()` |
| S&P 500 (`^GSPC`) | Yahoo Finance | `download_prices()` |
| Gold ETF — GOLDBEES.NS | Yahoo Finance | `download_prices()` |
| Nifty Next 50 ETF — JUNIORBEES.NS | Yahoo Finance | `download_prices()` |
| Momentum ETF — MOM100.NS | Yahoo Finance | `download_prices()` |
| Value ETF — NV20BEES.NS | Yahoo Finance | `download_prices()` |
| Quality ETF — QUAL30IETF.NS | Yahoo Finance | `download_prices()` |
| Liquid BeES — LIQUIDBEES.NS | Yahoo Finance | `download_prices()` |
| Mutual fund NAVs (AMFI codes) | AMFI via `mftool` | `download_mutual_fund_navs()` |
| India 10Y G-Sec yield | FRED (`IRLTLT01INM156N`) | `get_india_10y_gsec()` |
| RBI repo rate | Hardcoded from RBI press releases | `get_rbi_repo_rate_series()` |

---

## Data that needs manual download

### 1. Historical 10-year G-Sec yields (more accurate)

If FRED is unavailable or you want higher-frequency G-Sec data:

1. Go to **RBI DBIE** → https://dbie.rbi.org.in/DBIE/dbie.rbi?site=publications#!4
2. Select: **"Handbook of Statistics on Indian Economy"** → **"Financial Markets"**
3. Download: **"91-Day T-Bill / 10-Year Government Securities Yields"**
4. Save as: `data/raw/rbi_gsec_yields.csv` with columns `Date, Yield_10Y`

Then in the notebook, load it:
```python
gsec = pd.read_csv("../data/raw/rbi_gsec_yields.csv", parse_dates=["Date"], index_col="Date")
gsec_10y = gsec["Yield_10Y"] / 100
```

### 2. Fama-French factors for India (rigorous academic analysis)

For peer-reviewed factor research, download the India-specific factors from:

- **AQR Data Library** → https://www.aqr.com/Insights/Datasets
  (includes India in their global factor datasets)

- **Kenneth French Data Library** → https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
  (Asia-Pacific ex-Japan is the closest regional proxy)

Save as `data/raw/india_ff_factors.csv` with columns `Date, Mkt-RF, SMB, HML, MOM`

### 3. Sovereign Gold Bonds (SGB) — most tax-efficient gold investment

SGBs are exempt from LTCG on maturity (8 years) + pay 2.5% annual interest.
Buy from: **RBI Retail Direct** → https://rbiretaildirect.org.in

---

## Indian tax rules (FY 2024-25 — Budget 2024)

| Asset | Holding | Tax Rate | Exemption |
|-------|---------|----------|-----------|
| Equity MFs / ETFs / stocks | ≥ 12 months (LTCG) | **12.5%** | ₹1.25L/year |
| Equity MFs / ETFs / stocks | < 12 months (STCG) | **20%** | None |
| Gold ETF | ≥ 24 months (LTCG) | **12.5%** | None |
| Gold ETF | < 24 months | Slab rate | None |
| Debt MFs (post Apr-2023) | Any | Slab rate | None |
| Sovereign Gold Bonds (SGB) | On maturity (8Y) | **0%** | Full exemption |
| G-Sec coupon interest | Any | Slab rate | None |

---

## Factor ETFs available on NSE (2024)

| ETF | Tracks | TER |
|-----|--------|-----|
| MOM100.NS (Mirae Asset) | Nifty 200 Momentum 30 | ~0.25% |
| NV20BEES.NS (Nippon) | Nifty 50 Value 20 | ~0.27% |
| QUAL30IETF.NS (ICICI Pru) | Nifty 200 Quality 30 | ~0.29% |
| NIFTYBEES.NS (Nippon) | Nifty 50 | ~0.05% |
| JUNIORBEES.NS (Nippon) | Nifty Next 50 | ~0.19% |

For SIP into factor strategies, use the corresponding direct-plan index funds
(lower minimum investment, no bid-ask spread).

---

## Seminal references

- Sharpe, W.F. (1966). Mutual fund performance. *Journal of Business*, 39(S1), 119–138.
- Jensen, M.C. (1968). The performance of mutual funds 1945–1964. *Journal of Finance*, 23(2), 389–416.
- Fama, E.F. & French, K.R. (1993). Common risk factors in stock and bond returns. *Journal of Financial Economics*, 33(1), 3–56.
- Carhart, M.M. (1997). On persistence in mutual fund performance. *Journal of Finance*, 52(1), 57–82.
- Fama, E.F. & French, K.R. (2015). A five-factor asset pricing model. *Journal of Financial Economics*, 116(1), 1–22.
- Mohanty, P. (2002). Evidence of size effect on stock returns in India. *Vikalpa*, 27(3), 27–37.
- Wahal, S. & Yezdi, D. (2018). Fama-French factors in India. *Working Paper*.