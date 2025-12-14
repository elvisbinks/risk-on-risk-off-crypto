def test_import_data_loader():
    from src.data.yfinance_loader import download_and_save, fetch_ohlcv

    assert callable(fetch_ohlcv)
    assert callable(download_and_save)


def test_import_features():
    from src.features.build_features import FeatureConfig, build_features

    assert callable(build_features)
    assert FeatureConfig is not None


def test_import_models():
    from src.models.autoencoder import fit_autoencoder, predict_autoencoder
    from src.models.gmm import fit_gmm, predict_gmm
    from src.models.hmm import decode_hmm, fit_hmm

    assert callable(fit_hmm)
    assert callable(decode_hmm)
    assert callable(fit_gmm)
    assert callable(predict_gmm)
    assert callable(fit_autoencoder)
    assert callable(predict_autoencoder)


def test_import_evaluation():
    from src.evaluation.metrics import compute_regime_stats, compute_transition_matrix
    from src.evaluation.plots import plot_conditional_stats, plot_regime_timeline

    assert callable(compute_regime_stats)
    assert callable(compute_transition_matrix)
    assert callable(plot_regime_timeline)
    assert callable(plot_conditional_stats)
