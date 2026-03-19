"""Unit tests for mitochontrol.core utilities."""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

from mitochontrol.core import (
    _copy_matrix,
    _is_count_like,
    _matrix_values,
    compute_mt_fraction,
    get_cluster_dict,
    get_mt_dict,
)


# -- _is_count_like --------------------------------------------------

class TestIsCountLike:
    def test_integer_array(self):
        X = np.array([[1, 2], [3, 4]], dtype=float)
        assert _is_count_like(X) is True

    def test_float_array(self):
        X = np.array([[1.5, 2.3], [3.7, 4.1]])
        assert _is_count_like(X) is False

    def test_negative_values(self):
        X = np.array([[-1, 2], [3, 4]], dtype=float)
        assert _is_count_like(X) is False

    def test_sparse_counts(self):
        X = sparse.csr_matrix(
            np.array([[0, 5], [3, 0]], dtype=float)
        )
        assert _is_count_like(X) is True

    def test_empty_array(self):
        X = np.array([]).reshape(0, 0)
        assert _is_count_like(X) is True


# -- _copy_matrix / _matrix_values -----------------------------------

class TestMatrixHelpers:
    def test_copy_dense(self):
        X = np.array([[1.0, 2.0]])
        Y = _copy_matrix(X)
        assert np.array_equal(X, Y)
        Y[0, 0] = 999
        assert X[0, 0] == 1.0

    def test_copy_sparse(self):
        X = sparse.csr_matrix(np.eye(3))
        Y = _copy_matrix(X)
        assert sparse.issparse(Y)
        assert (X != Y).nnz == 0

    def test_matrix_values_sampling(self):
        X = np.arange(100).astype(float)
        vals = _matrix_values(X, max_items=10)
        assert vals.size == 10


# -- compute_mt_fraction ---------------------------------------------

class TestComputeMtFraction:
    def test_adds_mt_frac_column(self, small_adata):
        assert "mt_frac" not in small_adata.obs.columns
        compute_mt_fraction(small_adata)
        assert "mt_frac" in small_adata.obs.columns
        assert (small_adata.obs["mt_frac"] >= 0).all()

    def test_no_mt_genes(self):
        adata = AnnData(
            X=np.ones((5, 3)),
            var=pd.DataFrame(index=["A", "B", "C"]),
        )
        compute_mt_fraction(adata)
        assert (adata.obs["mt_frac"] == 0.0).all()


# -- get_cluster_dict / get_mt_dict ----------------------------------

class TestClusterDict:
    def test_splits_by_leiden(self, clustered_adata):
        cd = get_cluster_dict(clustered_adata)
        assert set(cd.keys()) == {"0", "1"}
        total = sum(a.n_obs for a in cd.values())
        assert total == clustered_adata.n_obs

    def test_missing_column_raises(self, small_adata):
        with pytest.raises(KeyError):
            get_cluster_dict(small_adata, "nonexistent")

    def test_mt_dict(self, clustered_adata):
        cd = get_cluster_dict(clustered_adata)
        md = get_mt_dict(cd)
        assert set(md.keys()) == {"0", "1"}
        for v in md.values():
            assert isinstance(v, pd.Series)
