# Architecture Documentation

## Design Principles

### 1. Separation of Concerns
- **Data Layer** (`src/data/`): Handles data fetching and I/O
- **Feature Layer** (`src/features/`): Transforms raw data into ML features
- **Model Layer** (`src/models/`): Implements regime detection algorithms
- **Evaluation Layer** (`src/evaluation/`): Metrics and visualization

### 2. Configuration-Driven Design
- All parameters externalized to YAML files
- Dataclasses for type-safe configuration
- Easy experimentation without code changes

### 3. Functional Core, Imperative Shell
- Core logic in pure functions (testable, composable)
- Side effects (I/O, plotting) isolated in scripts
- Clear data flow: raw → features → models → results

## Design Patterns

### Factory Pattern

The configuration classes act as factories:

- `HMMConfig` → builds and parametrizes the Hidden Markov Model  
- `GMMConfig` → builds and parametrizes the Gaussian Mixture Model  
- `AutoencoderConfig` → defines the neural network architecture and training settings

These classes centralize all hyperparameters and make it easy to create new model variants.

### Strategy Pattern

The project implements three interchangeable regime-detection strategies (HMM, GMM, Autoencoder).
All models expose a consistent functional interface:
	•	fit_* — trains the model on prepared features
	•	predict_* — generates regime labels

This design allows models to be:
	•	Compared directly
	•	Swapped easily without modifying the pipeline
	•	Extended with new algorithms in a plug-and-play manner

### Template Method Pattern

All models follow the same high-level workflow, ensuring consistent and fair evaluation:
	1.	Prepare features
Statistical indicators (returns, volatility, correlations) are computed and standardized.
	2.	Train the model (fit_*)
Each model learns a representation of market regimes according to its methodology.
	3.	Predict regimes (predict_*)
Produces the sequence of detected market states.

This template guarantees reproducibility and simplifies the integration of new models.

## Key Design Decisions

### Why Use Dataclasses for Configuration?

Dataclasses were chosen for model configuration because they provide:
	•	Type safety at runtime
	•	Automatic field generation (__init__, __repr__, etc.)
	•	Clear structure, easier to validate and document
	•	Better IDE support than dictionaries
	•	Compatibility with YAML-driven configuration

Dataclasses make the system easier to maintain and extend.

### Why Separate the prepare_features() Step?

The feature-preparation function is isolated because:
	•	It is reusable across all models
	•	It is fully testable without training a model
	•	It ensures consistent preprocessing for fair comparison
	•	It returns a scaler useful for deployment or inverse-transform operations

This separation respects the principle of single responsibility.

### Why Return Tuples Instead of Objects?

Model functions return tuples (e.g., (X, scaler) or (regimes, probabilities)) because:
	•	Tuples are simple and avoid unnecessary object-oriented overhead
	•	Values are explicit and easy to unpack
	•	They enable a functional programming style, reducing side effects
	•	They make the pipeline predictable and readable

This approach keeps the architecture lightweight and easy to test.

### Why StandardScaler?
- ML models sensitive to feature scales
- Returns, volatility, correlations have different ranges
- Standardization improves convergence
- Preserves scaler for production inference

## Data Flow

Raw Data (Yahoo Finance)
    ↓
CSV Files (data/raw/)
    ↓
Feature Engineering (build_features.py)
    ↓
Features DataFrame (data/processed/features.csv)
    ↓
Model Training (run_hmm.py, run_gmm.py, run_autoencoder.py)
    ↓
Regime Predictions (results/*_regimes.csv)
    ↓
Evaluation & Visualization (evaluate_models.py)
    ↓
Plots & Statistics (results/figures/)

## Testing Strategy

### Unit Tests
- Test individual functions in isolation
- Mock external dependencies (yfinance)
- Fast execution (<10s for all tests)

### Integration Tests
- Test full workflows end-to-end
- Use temporary directories for I/O
- Verify file creation and content

### Edge Case Tests
- Missing data (NaN handling)
- Invalid inputs (wrong types, empty DataFrames)
- Network failures (yfinance errors)

## Extensibility

### Adding a New Model
1. Create `src/models/your_model.py`
2. Implement `YourModelConfig` dataclass
3. Implement `fit_your_model()` and `predict_your_model()`
4. Add `configs/your_model.yaml`
5. Create `scripts/run_your_model.py`
6. Write tests in `tests/test_your_model.py`
7. Update `evaluate_models.py` to include new model

### Adding New Features
1. Add computation function to `build_features.py`
2. Update `FeatureConfig` if new parameters needed
3. Add tests for new feature
4. Re-run pipeline

## Performance Considerations

### Memory
- Streaming not needed (data fits in RAM)
- ~2000 rows × 12 features = ~200KB
- Models are lightweight (<1MB)

### Speed
- Data fetch: ~10s (network bound)
- Feature engineering: <1s
- HMM training: ~2s
- GMM training: ~1s
- Autoencoder training: ~10s (100 epochs)
- Total pipeline: <30s

### Scalability
- Current design handles 10+ years of daily data
- For intraday data, consider:
  - Chunked processing
  - Incremental learning
  - Database instead of CSV

## Dependencies

### Core
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `yfinance`: Data source

### ML
- `hmmlearn`: HMM implementation
- `scikit-learn`: GMM, preprocessing, metrics
- `torch`: Autoencoder

### Visualization
- `matplotlib`: Static plots
- `seaborn`: Statistical visualizations

### Development
- `pytest`: Testing framework
- `black`: Code formatting
- `flake8`: Linting
- `mypy`: Type checking

## Implemented Features

- ✅ **CI/CD**: GitHub Actions for automated testing and coverage
- ✅ **Pre-commit Hooks**: Automated code quality checks (black, isort, flake8, mypy)
- ✅ **Comprehensive Testing**: 58 unit tests with 97% coverage
- ✅ **Logging**: Centralized logging utilities for scripts

## Future Improvements

1. **Online Learning**: Update models with new data without retraining
2. **Feature Selection**: Identify most informative features
3. **Ensemble Methods**: Combine multiple models
4. **Regime Prediction**: Forecast next regime (not just classify current)
5. **Trading Strategy**: Backtest regime-based portfolio
6. **Web Dashboard**: Interactive visualization with Plotly Dash
7. **Docker**: Containerize for reproducibility
