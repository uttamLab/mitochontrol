"""Unit tests for mitochontrol.models."""

import numpy as np

from mitochontrol.models import (
    fit_empirical_histogram,
    online_em_gmm,
)


class TestFitEmpiricalHistogram:
    def test_returns_bin_info(self):
        data = np.random.default_rng(0).normal(5, 1, 200)
        pdf_empirical, bin_centers, bin_edges = fit_empirical_histogram(data)
        assert pdf_empirical is not None
        assert bin_centers is not None
        assert bin_edges is not None
        assert len(bin_centers) == len(pdf_empirical)
        assert len(bin_edges) == len(pdf_empirical) + 1


class TestOnlineEmGmm:
    def test_single_component(self):
        data = np.random.default_rng(1).normal(3, 0.5, 100)
        model, labels, bic = online_em_gmm(
            data, max_components=1,
        )
        assert model is not None
        assert labels is not None
        assert np.isfinite(bic)
        assert model["k"] == 1

    def test_bimodal_finds_two(self):
        rng = np.random.default_rng(2)
        data = np.concatenate([
            rng.normal(2, 0.3, 150),
            rng.normal(8, 0.3, 150),
        ])
        model, labels, _ = online_em_gmm(
            data, max_components=3,
        )
        assert model is not None
        assert model["k"] >= 2

    def test_empty_after_filter(self):
        data = np.array([np.nan, np.inf, -np.inf])
        model, labels, bic = online_em_gmm(data)
        assert model is None
        assert labels is None
        assert bic == np.inf
