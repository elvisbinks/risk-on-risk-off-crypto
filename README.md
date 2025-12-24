# Risk-On/Risk-Off Regime Detection

Unsupervised learning models (HMM, GMM, Autoencoder+KMeans) to detect market regimes in cryptocurrency data.

## Research Question
Can unsupervised learning identify Risk-On and Risk-Off regimes in crypto markets using returns, volatility, and correlation features?

## Setup

### Option 1: Using Conda (Recommended)
```bash
# Create environment
conda env create -f environment.yml
conda activate risk-on-risk-off-crypto
```

### Option 2: Using venv + pip
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run full pipeline (data download, feature engineering, model training, evaluation)
python main.py
```

**Expected output:**
- Regime predictions saved to `results/` (CSV files)
- Visualizations saved to `results/figures/` (PNG files)
- Console output showing regime statistics and model comparison

## Project Structure

```
risk-on-risk-off-crypto/
├── main.py                    # Main entry point
├── src/
│   ├── data/                  # Data loading (yfinance)
│   ├── features/              # Feature engineering
│   ├── models/                # HMM, GMM, Autoencoder
│   └── evaluation/            # Metrics and plots
├── scripts/                   # Individual pipeline steps
├── configs/                   # YAML configuration files
├── data/                      # Downloaded data (gitignored)
├── results/                   # Model outputs (gitignored)
├── environment.yml            # Conda dependencies
└── requirements.txt           # pip dependencies
```

## Requirements
- Python 3.11+
- pandas, numpy, scikit-learn, hmmlearn, torch, yfinance, matplotlib, seaborn
