"""Integration tests for the main pipeline entry points."""

import pandas as pd
import pytest

from mitochontrol.pipelines import (
    mtctrl_with_clustering,
    mtctrl_without_clustering,
)


class TestMtctrlWithClustering:
    def test_basic_run(self, clustered_adata, tmp_path):
        results = mtctrl_with_clustering(
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
        assert "post_prob_compromise" in r["threshold_stats"].columns

        stats_path = tmp_path / "mitochontrol" / "threshold_stats.csv"
        assert stats_path.exists()

    def test_requires_leiden(self, small_adata, tmp_path):
        with pytest.raises(ValueError, match="leiden"):
            mtctrl_with_clustering(
                adatas={"s": small_adata},
                outdir=tmp_path,
            )

    def test_save_requires_outdir(self, clustered_adata):
        with pytest.raises(ValueError, match="outdir"):
            mtctrl_with_clustering(
                adatas={"s": clustered_adata},
                outdir=None,
                save=True,
            )


class TestMtctrlWithoutClustering:
    def test_basic_run(self, clustered_adata, tmp_path):
        stats = mtctrl_without_clustering(
            clustered_adata,
            sample_id="test",
            outdir=tmp_path,
            show=False,
            save=True,
            color_by=None,
        )
        assert isinstance(stats, pd.DataFrame)
        assert len(stats) > 0
        assert "post_prob_compromise" in stats.columns

    def test_save_requires_outdir(self, clustered_adata):
        with pytest.raises(ValueError, match="outdir"):
            mtctrl_without_clustering(
                clustered_adata,
                sample_id="test",
                outdir=None,
                save=True,
            )
