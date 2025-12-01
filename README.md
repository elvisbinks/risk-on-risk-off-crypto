# Risk-On / Risk-Off Regime Detection in Cryptocurrency Markets

This project investigates whether major cryptocurrencies exhibit risk-on / risk-off dynamics similar to traditional financial assets.  
Daily data are collected for Bitcoin (BTC), Ethereum (ETH), S&P 500, and VIX.

The analysis compares three unsupervised learning approaches:

- **Hidden Markov Model (HMM)** – models temporal regime-switching behavior  
- **Gaussian Mixture Model (GMM)** – clustering based on statistical features  
- **Autoencoder + K-Means** – non-linear feature extraction followed by clustering  

The objective is to identify market regimes and study how each model separates periods of market stress versus periods of optimism.

## Overview
This project provides a full pipeline for regime detection in cryptocurrency markets, including:

- Data collection (Yahoo Finance via `yfinance`)
- Feature engineering (returns, volatility, correlations)
- Three unsupervised models (HMM, GMM, Autoencoder+KMeans)
- Evaluation and visualization of regime stability and transitions

## Key Features

- **Modular Architecture**: Clean separation of data, features, models, and evaluation  
- **Comprehensive Testing**: 58 unit tests with 97% code coverage  
- **Type-Safe**: Extensive use of type hints  
- **Well-Documented**: Docstrings and clear folder structure  
- **Configurable**: YAML-based configuration for custom experiments  
- **Production-Ready**: Pre-commit hooks, linting (black, isort, flake8), CI pipeline  

## Results Preview

| Model | Regime 0 (Risk-On) | Regime 1 (Risk-Off) | Persistence |
|-------|-------------------|---------------------|-------------|
| HMM | +0.24% daily | -0.24% daily | 98.2% |
| GMM | +0.07% daily | +0.03% daily | 89.1% |
| Autoencoder | +0.74% daily | -0.29% daily | 92.4% |

In summary:

- **HMM** finds a classic pattern: positive average return in Risk-On and negative in Risk-Off, with very high persistence (≈98%), so regimes change rarely.
- **GMM** separates returns more weakly and produces less stable regimes (≈89% persistence), suggesting it captures noise as well as true shifts.
- **Autoencoder** gives the strongest separation in mean returns (+0.74% vs –0.29% per day) but regimes are slightly less persistent than HMM, indicating more frequent switches.
- Overall, HMM is the most stable and interpretable, while the Autoencoder gives the clearest return separation but is more complex.

## Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- Internet connection (for fetching data)

## Quick Start

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd risk-on-risk-off-crypto

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Full Pipeline

```bash
# 1. Fetch historical data
python -m scripts.fetch_data --config configs/default.yaml

# 2. Build features
python -m scripts.build_features --config configs/default.yaml

# 3. Train models
python -m scripts.run_hmm --config configs/hmm.yaml
python -m scripts.run_gmm --config configs/gmm.yaml
python -m scripts.run_autoencoder --config configs/autoencoder.yaml

# 4. Evaluate and visualize
python -m scripts.evaluate_models
```

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html
open htmlcov/index.html

# Run specific test module
pytest tests/test_models.py -v
```

### Code Quality Checks

```bash
# Format code
black src/ tests/ scripts/

# Sort imports
isort src/ tests/ scripts/

# Lint code
flake8 src/ tests/ scripts/

# Type checking
mypy src/
```

## Project Structure

```
risk-on-risk-off-crypto/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI pipeline
├── src/                        # Source code
│   ├── data/                   # Data loading utilities
│   │   ├── __init__.py
│   │   └── yfinance_loader.py
│   ├── features/               # Feature engineering
│   │   ├── __init__.py
│   │   └── build_features.py
│   ├── models/                 # ML models (HMM, GMM, Autoencoder)
│   │   ├── __init__.py
│   │   ├── hmm.py
│   │   ├── gmm.py
│   │   └── autoencoder.py
│   ├── evaluation/             # Metrics and visualization
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── plots.py
│   ├── utils/                  # Utilities (logging, etc.)
│   │   ├── __init__.py
│   │   └── logging_utils.py
│   └── __init__.py
├── scripts/                    # Runnable CLI scripts
│   ├── __init__.py
│   ├── fetch_data.py           # Download market data
│   ├── build_features.py       # Engineer features
│   ├── run_hmm.py              # Train HMM model
│   ├── run_gmm.py              # Train GMM model
│   ├── run_autoencoder.py      # Train Autoencoder model
│   └── evaluate_models.py      # Compare and visualize results
├── tests/                      # Unit and integration tests
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_evaluation.py
│   ├── test_smoke.py
│   └── test_performance.py
├── configs/                    # YAML configuration files
│   ├── default.yaml            # Data and feature config
│   ├── hmm.yaml                # HMM hyperparameters
│   ├── gmm.yaml                # GMM hyperparameters
│   └── autoencoder.yaml        # Autoencoder hyperparameters
├── data/                       # Data directory (gitignored)
│   ├── raw/                    # Downloaded OHLCV data
│   │   ├── BTC-USD.csv
│   │   ├── ETH-USD.csv
│   │   ├── GSPC.csv
│   │   └── VIX.csv
│   └── processed/              # Engineered features
│       └── features.csv
├── results/                    # Model outputs (gitignored)
│   ├── figures/                # Plots and visualizations
│   │   ├── hmm_timeline.png
│   │   ├── hmm_transition.png
│   │   ├── gmm_timeline.png
│   │   ├── gmm_transition.png
│   │   ├── autoencoder_timeline.png
│   │   ├── autoencoder_transition.png
│   │   └── conditional_returns.png
│   ├── hmm_regimes.csv         # HMM regime predictions
│   ├── gmm_regimes.csv         # GMM regime predictions
│   ├── autoencoder_regimes.csv # Autoencoder regime predictions
│   └── autoencoder_embeddings.csv  # Latent space embeddings
├── notebooks/                  # Jupyter notebooks
│   └── regime_analysis.ipynb   # Interactive analysis
├── htmlcov/                    # Coverage reports (gitignored)
├── ARCHITECTURE.md             # Design decisions and patterns
├── PROPOSAL.md                 # Project proposal
├── README.md                   # This file
├── .pre-commit-config.yaml     # Pre-commit hooks config
├── .gitignore                  # Git ignore rules
├── pyproject.toml              # Tool configurations (black, isort, etc.)
├── pytest.ini                  # Pytest configuration
├── requirements.txt            # Python dependencies
└── setup.py                    # Package setup
```

## Configuration

The behavior of the data pipeline and the models can be adjusted through the YAML files in the `configs/` directory.

### Data Configuration (`configs/default.yaml`)
```yaml
start_date: "2016-01-01"
end_date: null  # null = today
interval: "1d"
symbols:
  - BTC-USD
  - ETH-USD
  - ^GSPC
  - ^VIX
```

### Model Configuration

Each model has its own configuration file:

- `configs/hmm.yaml`: HMM parameters controls the number of states, covariance structure, and initialization.
- `configs/gmm.yaml`: GMM parameters defines the number of mixture components and initialization parameters.
- `configs/autoencoder.yaml`: Specifies the neural network architecture (encoding dimension, epochs, optimizer settings).

## Interpreting Results

### Regime Statistics
Printed by `evaluate_models.py`:
- **mean**: Average daily return in each regime
- **std**: Volatility (standard deviation)
- **count**: Number of days in regime

### Transition Matrices
- **Diagonal values**: represent regime persistence (staying in same state)
- **Off-diagonal**: represent regime switches
- A higher diagonal indicates more stable and well-defined regimes.

### Visualizations
Generated in `results/figures/`:
1. **Timeline plots**: Regime changes overlaid with BTC returns
2. **Conditional returns**: Mean returns comparison across models
3. **Transition heatmaps**: Regime stability visualization

## Testing

The project includes comprehensive test coverage:

- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test full workflows (data → features → models)
- **Performance tests**: Verify training and prediction speed
- **Edge cases**: Missing data, invalid inputs, empty DataFrames
- **Coverage**: 97% (410/424 lines covered)

Run tests with:
```bash
pytest                          # Run all tests
pytest --cov=src               # With coverage report
pytest -v                      # Verbose output
pytest -m "not slow"           # Skip slow tests
```

## Development

## Model Comparison (Summary)

The three models differ in how they capture market structure:

### Hidden Markov Model (HMM)
- Captures **temporal dependencies**  
- Naturally represents regime switches  
- Interpretable transition matrix  
- Very stable regimes  

### Gaussian Mixture Model (GMM)
- No temporal memory (each day independent)  
- Simple clustering based on statistical features  
- Less stable regimes but fast to train  

### Autoencoder + K-Means
- Learns **non-linear** representations  
- Can uncover more complex regime boundaries  
- Less interpretable  
- Strong separation between Risk-On and Risk-Off returns  

## Acknowledgments

I used the following open-source tools in this project:

- `yfinance` for downloading market data  
- `hmmlearn` for the Hidden Markov Model implementation  
- PyTorch for the autoencoder  
- matplotlib and seaborn for data visualization  

## Contact

For questions about this project, please contact me directly.

**Note**: This project was developed as part of an academic course. It is intended solely for educational and research purposes.
