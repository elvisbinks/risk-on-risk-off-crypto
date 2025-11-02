# Detecting "Risk-On" and "Risk-Off" Regimes in Cryptocurrency Markets

## Category  
Financial Data Science – Time Series & Unsupervised Learning  

---

### Problem Statement  
Traditional financial markets alternate between **Risk-On** phases (optimism) and **Risk-Off** phases (risk aversion, market stress).  
While these regimes are well known in equities or bonds, their presence in **cryptocurrency markets** is less understood.  

This project aims to **automatically detect** these hidden regimes in crypto markets using **historical data only**, without any predefined economic assumptions.  
The goal is to show whether cryptocurrencies follow similar patterns of risk behavior as traditional assets.

---

### Objectives  
1. Detect **latent market regimes** (Risk-On / Risk-Off) using statistical features such as returns, volatility, and correlations.  
2. Compare detected regimes to traditional indicators (S&P500, Gold, VIX) to assess synchronization with global risk sentiment.  
3. Visualize results and interpret their economic meaning.  
4. *(Optional)* Forecast regime transitions using Hidden Markov Models.

---

### Methodology  
- **Data:** Historical data from **CoinGecko** and **Yahoo Finance** (BTC, ETH, S&P500, Gold, VIX).  
- **Features:** Daily returns, rolling volatility, z-scored volume, BTC–S&P500 correlations.  
- **Models:** Unsupervised algorithms – **Gaussian Mixture Model (GMM)** and **Hidden Markov Model (HMM)**.  
- **Visualization:** BTC price colored by regimes, regime probabilities, and comparison with major events (e.g., COVID, FTX).  
- **Tech stack:** Python 3.10+, `pandas`, `numpy`, `scikit-learn`, `hmmlearn`, `plotly`, `pytest`, `typer`.

---

### Expected Outcome  
The project will demonstrate that cryptocurrencies exhibit **detectable Risk-On and Risk-Off phases**, similar to traditional markets, which can be identified **purely through statistical learning**.  
It combines data science, time series analysis, and financial interpretation to reveal how digital assets react to global market sentiment.

---

**Author:** Elvis Hoxha
**Course:** Data Science & Advanced Programming 
**Date:** November 2025  