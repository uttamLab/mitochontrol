"""Unit tests for mitochontrol.clustering."""

import pandas as pd
import pytest

from mitochontrol.clustering import (
    assign_celltypes,
    cluster_data,
    construct_neighbors,
    differential_expression,
    read_marker_genes,
)


class TestConstructNeighbors:
    def test_builds_graph(self, small_adata):
        import scanpy as sc

        sc.pp.normalize_total(small_adata, target_sum=1e4)
        sc.pp.log1p(small_adata)
        small_adata.layers["lognorm"] = small_adata.X.copy()

        work = construct_neighbors(
            small_adata,
            analyzed_layer="lognorm",
            n_neighbors=10,
        )
        assert "neighbors" in work.uns

    def test_missing_layer_raises(self, small_adata):
        with pytest.raises(ValueError, match="not found"):
            construct_neighbors(
                small_adata,
                analyzed_layer="nonexistent",
            )


class TestClusterData:
    def test_adds_leiden(self, clustered_adata):
        import scanpy as sc

        work = clustered_adata.copy()
        sc.pp.neighbors(work, n_neighbors=10, use_rep="X_umap")
        cluster_data(work, resolution=0.5)
        assert "leiden" in work.obs.columns

    def test_missing_neighbors_raises(self, small_adata):
        with pytest.raises(ValueError):
            cluster_data(small_adata, resolution=0.5)


class TestDifferentialExpression:
    def test_returns_dataframe(self, clustered_adata):
        deg = differential_expression(
            clustered_adata,
            analyzed_layer="lognorm",
            groupby="leiden",
            n_top=5,
        )
        assert isinstance(deg, pd.DataFrame)
        assert "gene" in deg.columns
        assert len(deg) > 0


class TestReadMarkerGenes:
    def test_reads_csv(self, tmp_path):
        csv_path = tmp_path / "markers.csv"
        csv_path.write_text(
            "Tcell,CD3D,CD3E\nBcell,CD19,MS4A1\n"
        )
        markers = read_marker_genes(csv_path)
        assert "Tcell" in markers
        assert markers["Tcell"] == ["CD3D", "CD3E"]

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            read_marker_genes("/no/such/file.csv")


class TestAssignCelltypes:
    def test_assigns_labels(self, clustered_adata):
        deg = differential_expression(
            clustered_adata,
            analyzed_layer="lognorm",
            groupby="leiden",
            n_top=10,
        )
        markers = {"TypeA": ["Gene0", "Gene1"], "TypeB": ["Gene10"]}
        labels_df = assign_celltypes(
            clustered_adata, deg, markers,
        )
        assert "celltype" in clustered_adata.obs.columns
        assert isinstance(labels_df, pd.DataFrame)
