"""Integration tests for the main pipeline entry points."""

import pandas as pd
import pytest

from mitochontrol.pipelines import (
    get_thresholds,
    single_cluster_mitochontrol,
)


class TestGetThresholds:
    def test_basic_run(self, clustered_adata, tmp_path):
        results = get_thresholds(
            adatas={"sample": clustered_adata},
            outdir=tmp_path,
            threshold_probs=(0.8,),
            show=False,
            save=True,
        )
        assert "sample" in results
        r = results["sample"]
        assert "adata" in r
        assert "threshold_stats" in r

        stats_path = tmp_path / "mitochontrol" / "threshold_stats.csv"
        assert stats_path.exists()

    def test_requires_leiden(self, small_adata, tmp_path):
        with pytest.raises(ValueError, match="leiden"):
            get_thresholds(
                adatas={"s": small_adata},
                outdir=tmp_path,
            )

    def test_save_requires_outdir(self, clustered_adata):
        with pytest.raises(ValueError, match="outdir"):
            get_thresholds(
                adatas={"s": clustered_adata},
                outdir=None,
                save=True,
            )


class TestSingleClusterMitochontrol:
    def test_basic_run(self, clustered_adata, tmp_path):
        stats = single_cluster_mitochontrol(
            clustered_adata,
            sample_id="test",
            outdir=tmp_path,
            show=False,
            save=True,
            color_by=None,
        )
        assert isinstance(stats, pd.DataFrame)
        assert len(stats) > 0

    def test_save_requires_outdir(self, clustered_adata):
        with pytest.raises(ValueError, match="outdir"):
            single_cluster_mitochontrol(
                clustered_adata,
                sample_id="test",
                outdir=None,
                save=True,
            )
