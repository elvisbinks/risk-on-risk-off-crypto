# Risk-On/Risk-Off Regime Detection

Unsupervised learning models (HMM, GMM, Autoencoder+KMeans) to detect market regimes in cryptocurrency data.

## Research Question
Can unsupervised learning identify Risk-On and Risk-Off regimes in crypto markets using returns, volatility, and correlation features?

## Setup

## Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/risk-on-risk-off-crypto.git
cd risk-on-risk-off-crypto
```

### Option 1: Using Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate risk-on-risk-off-crypto
#**⚠️ IMPORTANT: PyTorch must be installed manually via pip (see step below)**
# Install PyTorch via pip 
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
```
## Usage

```bash
python main.py
```

### Option 2: Using venv + pip
```bash
python3 -m venv .venv
source .venv/bin/activate  
pip install -r requirements.txt
```
## Usage

```bash
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
