"""Clustering utilities for single-cell AnnData objects.

This module provides a compact clustering pipeline centered on
``clustering()``. The pipeline:

1. Verifies that raw counts are available.
2. Ensures a normalized, log-transformed analysis layer exists.
3. Excludes mitochondrial and ribosomal genes for neighbor construction and
   resolution selection.
4. Selects an optimal Leiden resolution.
5. Runs Leiden clustering and UMAP.
6. Computes the top differentially expressed genes per cluster.
7. Optionally assigns cell types from a marker-gene file.

Outputs are written under ``{outdir}/clustered`` and a summary dictionary is
returned.
"""

from __future__ import annotations

import csv
import logging
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from igraph import Graph
from scipy import sparse
from sklearn.metrics import silhouette_score

from .core import PathLike, _to_path, _ensure_dir, _copy_matrix, _is_count_like

DEFAULT_RAW_LAYER = "raw_counts"
DEFAULT_ANALYZED_LAYER = "lognorm"
DEFAULT_LABEL_COLUMN = "celltype"

logger = logging.getLogger(__name__)


def _resolve_raw_layer(
    adata: AnnData,
    raw_layer: Optional[str] = None,
) -> str:
    """Find or create the layer that stores raw counts."""
    if raw_layer is not None:
        if raw_layer not in adata.layers:
            raise ValueError(
                f"Raw counts layer '{raw_layer}' "
                "was not found in adata.layers."
            )
        if not _is_count_like(adata.layers[raw_layer]):
            raise ValueError(
                f"Layer '{raw_layer}' does not appear to contain raw counts."
            )
        return raw_layer

    preferred = [DEFAULT_RAW_LAYER, "counts", "raw", "counts_raw"]
    for candidate in preferred:
        if (candidate in adata.layers
                and _is_count_like(adata.layers[candidate])):
            return candidate

    for candidate in adata.layers.keys():
        if _is_count_like(adata.layers[candidate]):
            return str(candidate)

    if _is_count_like(adata.X):
        adata.layers[DEFAULT_RAW_LAYER] = _copy_matrix(adata.X)
        return DEFAULT_RAW_LAYER

    raise ValueError(
        "Raw counts are required for clustering, but no raw-count matrix was "
        "found in `adata.layers` or `adata.X`. Provide raw counts via "
        "`raw_layer` or store integer-like counts in the AnnData object."
    )


def _resolve_analyzed_layer(
    adata: AnnData,
    raw_layer: str,
    analyzed_layer: Optional[str] = None,
) -> str:
    """Find or create the normalized, log-transformed analysis layer."""
    target_layer = analyzed_layer or DEFAULT_ANALYZED_LAYER

    if target_layer in adata.layers:
        return target_layer

    if analyzed_layer is None and not _is_count_like(adata.X):
        adata.layers[target_layer] = _copy_matrix(adata.X)
        return target_layer

    warnings.warn(
        "Warning: adata did not contain preprocessed counts. Running "
        "normalization and log-transformation now",
        stacklevel=2,
    )
    proc = AnnData(
        X=_copy_matrix(adata.layers[raw_layer]),
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )
    sc.pp.normalize_total(proc, target_sum=1e4)
    sc.pp.log1p(proc)
    adata.layers[target_layer] = _copy_matrix(proc.X)
    return target_layer


def _feature_mask(var_names: pd.Index) -> np.ndarray:
    """Return a mask of genes to keep for clustering."""
    upper = var_names.astype(str).str.upper()
    mt = upper.str.startswith("MT-")
    ribo = (
        upper.str.startswith("RPS")
        | upper.str.startswith("RPL")
        | upper.str.startswith("MRPS")
        | upper.str.startswith("MRPL")
    )
    return np.asarray(~(mt | ribo), dtype=bool)


def construct_neighbors(
    adata: AnnData,
    analyzed_layer: str = DEFAULT_ANALYZED_LAYER,
    *,
    n_neighbors: int = 30,
    n_pcs: int = 30,
    n_top_genes: int = 3500,
) -> AnnData:
    """Build a neighbor graph, excluding mt/ribo genes.

    Creates a PCA-based kNN graph on the analyzed-count layer
    after filtering mitochondrial and ribosomal genes,
    selecting highly variable genes, and scaling.

    Args:
        adata: AnnData with an analyzed-count layer.
        analyzed_layer: Layer containing normalized,
            log-transformed counts.
        n_neighbors: Number of neighbors for the kNN graph.
        n_pcs: Number of principal components.
        n_top_genes: Number of highly variable genes.

    Returns:
        A working-copy AnnData containing the neighbor graph
        in ``uns["neighbors"]`` and ``obsp``, ready for
        clustering and UMAP.

    Raises:
        ValueError: If *analyzed_layer* is missing, fewer than
            2 cells are present, or no genes remain after
            filtering.
    """
    if analyzed_layer not in adata.layers:
        raise ValueError(
            f"Analyzed layer '{analyzed_layer}' was not found in adata.layers."
        )
    if adata.n_obs < 2:
        raise ValueError("At least two cells are required for clustering.")

    keep_mask = _feature_mask(adata.var_names)
    if not np.any(keep_mask):
        raise ValueError(
            "No genes remain after excluding "
            "mitochondrial and ribosomal genes."
        )

    work = adata[:, keep_mask].copy()
    work.X = _copy_matrix(adata.layers[analyzed_layer][:, keep_mask])

    n_top = min(int(n_top_genes), work.n_vars)
    if n_top >= 1:
        sc.pp.highly_variable_genes(
            work,
            n_top_genes=n_top,
            flavor="seurat",
            subset=False,
        )
        hvg_ok = (
            "highly_variable" in work.var
            and work.var["highly_variable"].sum() >= 2
        )
        if hvg_ok:
            work = work[:, work.var["highly_variable"]].copy()

    sc.pp.scale(work, max_value=10)

    max_neighbors = max(1, work.n_obs - 1)
    n_neighbors = min(int(n_neighbors), max_neighbors)
    min_dim = min(work.n_obs, work.n_vars)

    if min_dim >= 3:
        n_comps = min(int(n_pcs), min_dim - 1)
        sc.tl.pca(work, n_comps=n_comps, svd_solver="arpack")
        sc.pp.neighbors(
            work,
            n_neighbors=n_neighbors,
            n_pcs=n_comps,
            metric="cosine",
        )
    else:
        sc.pp.neighbors(
            work, n_neighbors=n_neighbors,
            use_rep="X", metric="cosine",
        )

    return work


def _membership_to_numeric(labels: pd.Series) -> np.ndarray:
    """Convert cluster labels to numeric membership values."""
    if hasattr(labels, "cat"):
        return labels.cat.codes.to_numpy()
    _, membership = np.unique(labels.to_numpy(), return_inverse=True)
    return membership


def _minmax_scale(values: np.ndarray) -> np.ndarray:
    """Min-max scale numeric values with NaN-safe handling."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values)
    vmin = np.nanmin(values[finite])
    vmax = np.nanmax(values[finite])
    if vmax == vmin:
        return np.zeros_like(values)
    out = np.zeros_like(values)
    out[finite] = (values[finite] - vmin) / (vmax - vmin)
    return out


def optimal_res(
    adata: AnnData,
    *,
    label: str = "adata",
    min_res: float = 0.1,
    max_res: float = 1.4,
    step: float = 0.05,
    plot_path: Optional[PathLike] = None,
    show: bool = False,
) -> float:
    """Select a Leiden resolution via modularity and silhouette.

    Sweeps a range of resolutions, scoring each by a weighted
    combination of modularity (0.7) and silhouette (0.3).
    Saves a diagnostic plot when *plot_path* is provided.

    Args:
        adata: AnnData with a precomputed neighbor graph
            (from ``construct_neighbors``).
        label: Label for the plot title.
        min_res: Start of the resolution sweep.
        max_res: End of the resolution sweep.
        step: Step size between resolutions.
        plot_path: File path for the resolution-selection
            plot.  ``None`` skips saving.
        show: Display the plot interactively.

    Returns:
        The selected Leiden resolution (float).

    Raises:
        ValueError: If no neighbor graph is found in *adata*.
    """
    if "neighbors" not in adata.uns:
        raise ValueError(
            "Run construct_neighbors() before "
            "selecting resolution."
        )

    resolutions = np.round(np.arange(min_res, max_res + step / 2, step), 3)
    base = adata.copy()
    conn = base.obsp["connectivities"]
    if not sparse.issparse(conn):
        conn = sparse.csr_matrix(conn)
    coo = conn.tocoo()
    graph = Graph(
        n=conn.shape[0],
        edges=list(zip(coo.row.tolist(), coo.col.tolist())),
        directed=False,
    )
    graph.es["weight"] = coo.data.tolist()
    if "X_pca" in adata.obsm:
        X_sil = adata.obsm["X_pca"]
    else:
        X_sil = np.asarray(adata.X)

    modularity_scores = []
    silhouette_scores = []
    cluster_counts = []
    min_cluster_sizes = []

    for resolution in resolutions:
        trial = base.copy()
        sc.tl.leiden(
            trial,
            resolution=float(resolution),
            key_added="leiden",
            random_state=42,
            use_weights=True,
        )
        labels = trial.obs["leiden"].astype("category")
        counts = labels.value_counts()
        cluster_counts.append(int(labels.nunique()))
        min_cluster_sizes.append(int(counts.min()))

        if labels.nunique() < 2:
            modularity_scores.append(np.nan)
            silhouette_scores.append(np.nan)
            continue

        membership = _membership_to_numeric(labels)
        mod = graph.modularity(
            list(membership), weights="weight",
        )
        modularity_scores.append(mod)

        try:
            silhouette_scores.append(
                silhouette_score(
                    X_sil,
                    labels.astype(str),
                    sample_size=min(2000, trial.n_obs),
                    random_state=42,
                )
            )
        except ValueError:
            silhouette_scores.append(np.nan)

    modularity_scaled = _minmax_scale(np.asarray(modularity_scores))
    silhouette_scaled = _minmax_scale(np.asarray(silhouette_scores))
    combined = 0.7 * modularity_scaled + 0.3 * silhouette_scaled

    min_size_cutoff = max(10, int(0.005 * adata.n_obs))
    valid = [
        i
        for i, (n_clusters, min_size) in enumerate(
            zip(cluster_counts, min_cluster_sizes)
        )
        if n_clusters >= 2 and min_size >= min_size_cutoff
    ]
    if valid:
        best_idx = max(valid, key=lambda i: combined[i])
    else:
        eligible = [
            i for i, n_clusters
            in enumerate(cluster_counts)
            if n_clusters >= 2
        ]
        best_idx = max(
            eligible or range(len(resolutions)),
            key=lambda i: combined[i],
        )

    selected = float(resolutions[best_idx])

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(
        resolutions, modularity_scaled,
        marker="o", label="Modularity",
    )
    axes[0].plot(
        resolutions, silhouette_scaled,
        marker="s", label="Silhouette",
    )
    axes[0].plot(
        resolutions, combined,
        linestyle="--", color="black",
        label="Combined",
    )
    axes[0].axvline(selected, color="red", linestyle="--")
    axes[0].set_ylabel("Normalized score")
    axes[0].legend()

    axes[1].plot(resolutions, cluster_counts, marker="^", label="Clusters")
    axes[1].plot(
        resolutions, min_cluster_sizes,
        marker="v", label="Min cluster size",
    )
    axes[1].axvline(selected, color="red", linestyle="--")
    axes[1].axhline(
        min_size_cutoff,
        color="grey",
        linestyle=":",
        label="Min size cutoff",
    )
    axes[1].set_xlabel("Leiden resolution")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    fig.suptitle(f"{label}\nSelected resolution: {selected:.2f}")
    fig.tight_layout()

    if plot_path is not None:
        plot_path = _to_path(plot_path)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    return round(selected, 2)


def cluster_data(
    adata: AnnData,
    resolution: float,
    *,
    key_added: str = "leiden",
    random_state: int = 42,
) -> None:
    """Run Leiden clustering in place.

    Args:
        adata: AnnData with a precomputed neighbor graph.
        resolution: Leiden resolution parameter.
        key_added: Column name added to ``adata.obs``.
        random_state: Random seed for reproducibility.

    Raises:
        ValueError: If no neighbor graph is found in *adata*.
    """
    if "neighbors" not in adata.uns:
        raise ValueError("Run construct_neighbors() before cluster_data().")
    sc.tl.leiden(
        adata,
        resolution=float(resolution),
        key_added=key_added,
        random_state=random_state,
        use_weights=True,
    )


def differential_expression(
    adata: AnnData,
    analyzed_layer: str = DEFAULT_ANALYZED_LAYER,
    *,
    groupby: str = "leiden",
    n_top: int = 50,
    output_path: Optional[PathLike] = None,
) -> pd.DataFrame:
    """Compute top differentially expressed genes per cluster.

    Uses the Wilcoxon rank-sum test on the analyzed-count
    layer and returns a tidy ``DataFrame`` with the top
    *n_top* genes per group.

    Args:
        adata: AnnData with cluster labels.
        analyzed_layer: Layer with normalized, log-transformed
            counts used for the test.
        groupby: Column in ``adata.obs`` defining groups.
        n_top: Number of top genes per group.
        output_path: Optional CSV path.  Written when provided.

    Returns:
        ``DataFrame`` with columns *groupby*, ``rank``,
        ``gene``, and Wilcoxon statistics.

    Raises:
        ValueError: If *analyzed_layer* or *groupby* is missing.
    """
    if analyzed_layer not in adata.layers:
        raise ValueError(
            f"Analyzed layer '{analyzed_layer}' was not found in adata.layers."
        )
    if groupby not in adata.obs:
        raise ValueError(f"Column '{groupby}' was not found in adata.obs.")

    work = adata.copy()
    work.X = _copy_matrix(adata.layers[analyzed_layer])
    sc.tl.rank_genes_groups(
        work, groupby=groupby,
        method="wilcoxon", use_raw=False,
    )

    cats = (
        work.obs[groupby].astype(str).astype("category")
    )
    groups = list(cats.cat.categories)
    frames = []
    for group in groups:
        df = sc.get.rank_genes_groups_df(
            work, group=group,
        ).head(int(n_top)).copy()
        df.insert(0, groupby, str(group))
        df.insert(1, "rank", np.arange(1, len(df) + 1))
        frames.append(df.rename(columns={"names": "gene"}))

    deg_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if output_path is not None:
        output_path = _to_path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        deg_df.to_csv(output_path, index=False)
    return deg_df


def read_marker_genes(filepath: PathLike) -> dict[str, list[str]]:
    """Read marker genes from a headerless CSV file.

    Each row has the format ``celltype, gene1, gene2, ...``.

    Args:
        filepath: Path to the CSV file.

    Returns:
        Dictionary mapping cell-type labels to lists of
        marker-gene names.

    Raises:
        FileNotFoundError: If *filepath* does not exist.
    """
    filepath = _to_path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Marker gene file not found: {filepath}")

    markers: dict[str, list[str]] = {}
    with filepath.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            label = row[0].strip()
            genes = [gene.strip() for gene in row[1:] if gene.strip()]
            if label:
                markers[label] = genes
    return markers


def assign_celltypes(
    adata: AnnData,
    deg_df: pd.DataFrame,
    marker_genes: Union[PathLike, Mapping[str, list[str]]],
    *,
    label_column: str = DEFAULT_LABEL_COLUMN,
    output_path: Optional[PathLike] = None,
) -> pd.DataFrame:
    """Assign cell types from cluster DEGs and marker genes.

    For each cluster, the cell type whose marker-gene set has
    the largest overlap with the cluster's top DEGs is chosen.

    Args:
        adata: AnnData with a ``"leiden"`` column in ``obs``.
            A new *label_column* column is added in place.
        deg_df: ``DataFrame`` of DEGs (output of
            ``differential_expression``).
        marker_genes: Path to a marker-gene CSV or a
            ``{celltype: [genes]}`` mapping.
        label_column: Name of the column written to
            ``adata.obs``.
        output_path: Optional CSV path for the label table.

    Returns:
        ``DataFrame`` with columns ``leiden``,
        *label_column*, ``overlap_count``, and
        ``overlap_genes``.
    """
    markers = (
        read_marker_genes(marker_genes)
        if isinstance(marker_genes, (str, Path))
        else {k: list(v) for k, v in marker_genes.items()}
    )
    marker_sets = {
        label: {gene.upper() for gene in genes}
        for label, genes in markers.items()
    }

    rows = []
    mapping: dict[str, str] = {}
    for cluster, cluster_df in deg_df.groupby("leiden", sort=False):
        genes = [str(gene) for gene in cluster_df["gene"] if pd.notna(gene)]
        gene_set = {gene.upper() for gene in genes}
        best_label = "Unknown"
        best_overlap: list[str] = []

        for celltype, marker_set in marker_sets.items():
            overlap = sorted(gene_set.intersection(marker_set))
            if len(overlap) > len(best_overlap):
                best_label = celltype
                best_overlap = overlap

        mapping[str(cluster)] = best_label
        rows.append(
            {
                "leiden": str(cluster),
                label_column: best_label,
                "overlap_count": len(best_overlap),
                "overlap_genes": ";".join(best_overlap),
            }
        )

    labels_df = pd.DataFrame(rows)
    adata.obs[label_column] = (
        adata.obs["leiden"].astype(str)
        .map(mapping)
        .fillna("Unknown")
        .astype("object")
    )

    if output_path is not None:
        output_path = _to_path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        labels_df.to_csv(output_path, index=False)

    return labels_df


def _cluster_output_paths(outdir: PathLike, label: str) -> dict[str, Path]:
    """Build standard clustering output paths."""
    base = _ensure_dir(_to_path(outdir) / "clustered")
    return {
        "adata": _ensure_dir(base / "adata") / f"{label}.h5ad",
        "umap": _ensure_dir(base / "umap") / f"{label}.pdf",
        "res": _ensure_dir(base / "res_selection") / f"{label}.pdf",
        "deg": _ensure_dir(base / "DEG") / f"{label}.csv",
        "labels": _ensure_dir(base / "celltype_labels") / f"{label}.csv",
    }


def clustering(
    adata: AnnData,
    label: str,
    outdir: PathLike,
    *,
    raw_layer: Optional[str] = None,
    analyzed_layer: Optional[str] = None,
    marker_genes: Optional[Union[PathLike, Mapping[str, list[str]]]] = None,
    n_neighbors: int = 30,
    n_pcs: int = 30,
    n_top_genes: int = 3500,
    min_res: float = 0.1,
    max_res: float = 1.4,
    resolution_step: float = 0.05,
    random_state: int = 42,
    show: bool = False,
) -> dict[str, Any]:
    """Run the full clustering pipeline on one ``AnnData`` object.

    Args:
        adata: Input AnnData object.
        label: Label used in output filenames.
        outdir: Root output directory.  Results are written
            under ``{outdir}/clustered/``.
        raw_layer: Layer containing raw counts.  Auto-detected
            from ``adata.layers`` or ``adata.X`` when ``None``.
        analyzed_layer: Layer with normalized, log-transformed
            counts.  When ``None``, the function either adopts
            ``adata.X`` (if already preprocessed) or runs
            normalization and log-transformation from raw counts.
        marker_genes: Marker-gene CSV path or ``{celltype: [genes]}``
            mapping for optional cell-type assignment.
        n_neighbors: Number of neighbors for the kNN graph.
        n_pcs: Number of principal components for PCA.
        n_top_genes: Number of highly variable genes to select.
        min_res: Minimum Leiden resolution to search.
        max_res: Maximum Leiden resolution to search.
        resolution_step: Step size for the resolution sweep.
        random_state: Seed for Leiden clustering and UMAP.
        show: Display the resolution plot and UMAP.

    Returns:
        Dictionary with keys ``"adata"`` (clustered AnnData),
        ``"resolution"``, ``"n_clusters"``, ``"raw_layer"``,
        and ``"analyzed_layer"``.

    Raises:
        ValueError: If no raw counts can be found.
    """
    adata = adata.copy()
    paths = _cluster_output_paths(outdir, label)

    raw_layer_name = _resolve_raw_layer(adata, raw_layer=raw_layer)
    analyzed_layer_name = _resolve_analyzed_layer(
        adata,
        raw_layer=raw_layer_name,
        analyzed_layer=analyzed_layer,
    )
    adata.X = _copy_matrix(adata.layers[analyzed_layer_name])

    work = construct_neighbors(
        adata,
        analyzed_layer=analyzed_layer_name,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        n_top_genes=n_top_genes,
    )
    selected_resolution = optimal_res(
        work,
        label=label,
        min_res=min_res,
        max_res=max_res,
        step=resolution_step,
        plot_path=paths["res"],
        show=show,
    )

    cluster_data(
        work, selected_resolution,
        key_added="leiden",
        random_state=random_state,
    )
    sc.tl.umap(work, random_state=random_state)

    adata.obs["leiden"] = work.obs["leiden"].astype(str).to_numpy()
    adata.obsm["X_umap"] = work.obsm["X_umap"].copy()

    deg_df = differential_expression(
        adata,
        analyzed_layer=analyzed_layer_name,
        groupby="leiden",
        n_top=50,
        output_path=paths["deg"],
    )

    label_df: Optional[pd.DataFrame] = None
    if marker_genes is not None:
        label_df = assign_celltypes(
            adata,
            deg_df,
            marker_genes,
            label_column=DEFAULT_LABEL_COLUMN,
            output_path=paths["labels"],
        )

    fig = sc.pl.umap(adata, color="leiden", show=False, return_fig=True)
    fig.savefig(paths["umap"], bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    adata.write(paths["adata"])

    n_clusters = int(adata.obs["leiden"].astype(str).nunique())
    logger.info(
        "Clustering complete for %s | resolution=%.2f | n_clusters=%d | "
        "raw_layer=%s | analyzed_layer=%s",
        label,
        selected_resolution,
        n_clusters,
        raw_layer_name,
        analyzed_layer_name,
    )

    return {
        "adata": adata,
        "resolution": selected_resolution,
        "n_clusters": n_clusters,
        "raw_layer": raw_layer_name,
        "analyzed_layer": analyzed_layer_name,
        "deg": deg_df,
        "celltype_labels": label_df,
        "paths": paths,
    }
