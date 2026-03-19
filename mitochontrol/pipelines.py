"""Pipeline functions for end-to-end thresholding workflows."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import pandas as pd
import scanpy as sc
from anndata import AnnData

from . import core
from . import enrichment
from . import models
from . import thresholding
from . import visualization
from .core import PathLike, _to_path, _ensure_dir, _copy_matrix, _is_count_like

logger = logging.getLogger(__name__)


def _extract_adata(
    entry: Union[AnnData, Mapping[str, Any]],
) -> tuple[AnnData, Optional[str]]:
    """Extract AnnData and raw-layer name from a pipeline entry."""
    if isinstance(entry, AnnData):
        return entry.copy(), None
    if isinstance(entry, Mapping) and isinstance(entry.get("adata"), AnnData):
        return entry["adata"].copy(), entry.get("raw_layer")
    raise TypeError(
        "Each value in `adatas` must be an AnnData object or a mapping "
        "containing an 'adata' key with an AnnData value."
    )


def _resolve_raw_layer(
    adata: AnnData,
    preferred: Optional[str] = None,
) -> str:
    """Find the layer containing raw counts."""
    if preferred is not None:
        if preferred not in adata.layers:
            raise ValueError(
                f"Raw counts layer '{preferred}' "
                "was not found in adata.layers."
            )
        if not _is_count_like(adata.layers[preferred]):
            raise ValueError(
                f"Layer '{preferred}' does not appear to contain raw counts."
            )
        return preferred

    candidates = ["raw_counts", "counts", "raw", "counts_raw"]
    for candidate in candidates:
        if (candidate in adata.layers
                and _is_count_like(adata.layers[candidate])):
            return candidate
    for candidate in adata.layers.keys():
        if _is_count_like(adata.layers[candidate]):
            return str(candidate)
    if _is_count_like(adata.X):
        adata.layers["raw_counts"] = _copy_matrix(adata.X)
        return "raw_counts"
    raise ValueError(
        "Raw counts are required for MitoChontrol "
        "thresholding, but no raw-count matrix was "
        "found in `adata.layers` or `adata.X`."
    )


def _ensure_mt_fraction(adata: AnnData, raw_layer: str) -> None:
    """Compute ``mt_frac`` from raw counts if it is missing."""
    if "mt_frac" in adata.obs.columns:
        return
    temp = adata.copy()
    temp.X = _copy_matrix(adata.layers[raw_layer])
    core.compute_mt_fraction(temp)
    adata.obs["mt_frac"] = temp.obs["mt_frac"].astype(float).to_numpy()


def _ensure_umap(adata: AnnData) -> None:
    """Ensure the AnnData object has a UMAP embedding."""
    if "X_umap" in adata.obsm:
        return
    if "neighbors" in adata.uns:
        sc.tl.umap(adata, random_state=42)
        return
    raise ValueError(
        "The input AnnData does not contain `obsm['X_umap']` or a neighbor "
        "graph. Run `clustering()` first so sample-level UMAPs can be saved."
    )


def _format_prob(prob: float) -> str:
    """Format threshold probability for filenames and column names."""
    return f"{float(prob):g}"


def _threshold_column_name(prob: float) -> str:
    """Return the standard sample-level threshold column name."""
    return f"mitochontrol_threshold_out_{_format_prob(prob)}"


def _cluster_overlay_path(
    outdir: PathLike, label: str,
) -> Path:
    """Return path for a cluster-overlay PDF."""
    base = _ensure_dir(
        _to_path(outdir) / "mitochontrol" / "cluster_overlays"
    )
    return base / f"{label}.pdf"


def _adata_path(
    outdir: PathLike, label: str,
) -> Path:
    """Return path for a thresholded AnnData file."""
    base = _ensure_dir(
        _to_path(outdir) / "mitochontrol" / "adata"
    )
    return base / f"{label}.h5ad"


def _threshold_plot_path(
    outdir: PathLike,
    label: str,
    cluster_id: Optional[str],
    prob: float,
) -> Path:
    """Return path for a naive-Bayes threshold PDF."""
    base = _ensure_dir(
        _to_path(outdir) / "mitochontrol" / "threshold"
    )
    suffix = (
        f"_cluster{cluster_id}"
        if cluster_id is not None else ""
    )
    return base / f"{label}{suffix}_{_format_prob(prob)}.pdf"


def _enrichment_plot_path(
    outdir: PathLike,
    label: str,
    cluster_id: Optional[str],
    prob: float,
) -> Path:
    """Return path for an enrichment-analysis PDF."""
    base = _ensure_dir(
        _to_path(outdir) / "mitochontrol" / "enrichment"
    )
    suffix = (
        f"_cluster{cluster_id}"
        if cluster_id is not None else ""
    )
    return base / f"{label}{suffix}_{_format_prob(prob)}.pdf"


def _filtered_umap_path(
    outdir: PathLike,
    label: str,
    cluster_id: Optional[str],
    prob: float,
) -> Path:
    """Return path for a filtered-UMAP PDF."""
    base = _ensure_dir(
        _to_path(outdir) / "mitochontrol" / "filtered_umap"
    )
    suffix = (
        f"_cluster{cluster_id}"
        if cluster_id is not None else ""
    )
    return base / f"{label}{suffix}_{_format_prob(prob)}.pdf"


def _save_filtered_umap(
    adata: AnnData,
    column_name: str,
    save_path: Path,
    show: bool,
    cluster_obs_names: Optional[pd.Index] = None,
    cluster_id: Optional[str] = None,
    prob: Optional[float] = None,
    sample_id: Optional[str] = None,
) -> None:
    """Save a UMAP of the sample colored by threshold status."""
    visualization.plot_threshold_umap(
        adata,
        threshold_column=column_name,
        cluster_obs_names=cluster_obs_names,
        cluster_id=cluster_id,
        prob=prob,
        sample_id=sample_id,
        save=True,
        save_path=save_path,
        show=show,
    )


def _run_cluster_thresholds(
    adata_cluster: AnnData,
    *,
    label: str,
    cluster_id: Optional[str],
    threshold_probs: Sequence[float],
    outdir: Optional[PathLike],
    save: bool,
    show: bool,
) -> tuple[dict[float, float], list[dict[str, Any]]]:
    """Fit the cluster GMM and apply naive Bayes thresholds."""
    mt_data = adata_cluster.obs["mt_frac"].to_numpy(dtype=float)
    gmm_model, _, _ = models.online_em_gmm(
        mt_data,
        init_method="quantile",
        max_components=5,
    )

    thresholds: dict[float, float] = {}
    stats_rows: list[dict[str, Any]] = []

    for prob in threshold_probs:
        prob = float(prob)
        column_name = _threshold_column_name(prob)
        save_path = None
        if save and outdir is not None:
            save_path = _threshold_plot_path(outdir, label, cluster_id, prob)
        result = thresholding.naive_bayes_threshold(
            adata_cluster=adata_cluster,
            x_values=mt_data,
            model=gmm_model,
            threshold_prob=prob,
            visualize=True,
            save=save,
            save_path=save_path,
            show=show,
            column_name=column_name,
            suptitle=(
                f"{label}: cluster {cluster_id}"
                if cluster_id is not None else label
            ),
            celltype_color="lightgrey",
        )
        adata_cluster.obs[column_name] = (
            adata_cluster.obs[column_name].astype(bool)
        )
        thresholds[prob] = float(result["threshold"])
        cells_lost = int(adata_cluster.obs[column_name].sum())
        cells_retained = int(adata_cluster.n_obs - cells_lost)
        stats_rows.append(
            {
                "adata_label": label,
                "cluster_id": "" if cluster_id is None else str(cluster_id),
                "threshold_probability": prob,
                "threshold_value": float(result["threshold"]),
                "cells_lost": cells_lost,
                "cells_retained": cells_retained,
            }
        )

    return thresholds, stats_rows


def get_thresholds(
    adatas: Mapping[str, Union[AnnData, Mapping[str, Any]]],
    outdir: Optional[PathLike] = None,
    threshold_probs: Optional[Sequence[float]] = None,
    show: bool = False,
    save: bool = True,
) -> dict[str, dict[str, Any]]:
    """Run MitoChontrol thresholding on clustered AnnData objects.

    Args:
        adatas: Mapping from sample label to either an
            ``AnnData`` or a dict with an ``"adata"`` key
            (e.g., the return value of ``clustering()``).
        outdir: Root directory for all MitoChontrol outputs.
        threshold_probs: Posterior-probability cutoffs for
            naive Bayes thresholding.  Defaults to ``(0.8,)``.
        show: Whether to display plots interactively.
        save: Whether to write outputs to disk.

    Returns:
        Mapping from sample label to a result dict containing
        the final ``adata``, per-cluster thresholds, and a
        ``threshold_stats`` ``DataFrame``.

    Raises:
        ValueError: If ``save=True`` and ``outdir`` is ``None``,
            or if an AnnData is missing a ``"leiden"`` column.
        TypeError: If a value in *adatas* is not an ``AnnData``
            or a mapping containing one.
    """
    if threshold_probs is None:
        threshold_probs = (0.8,)
    if save and outdir is None:
        raise ValueError("`outdir` is required when `save=True`.")

    results: dict[str, dict[str, Any]] = {}
    all_stats: list[dict[str, Any]] = []

    for label, entry in adatas.items():
        adata, preferred_raw_layer = _extract_adata(entry)
        raw_layer = _resolve_raw_layer(adata, preferred=preferred_raw_layer)

        if "leiden" not in adata.obs.columns:
            raise ValueError(
                f"AnnData '{label}' must contain "
                "a 'leiden' column in adata.obs."
            )

        _ensure_mt_fraction(adata, raw_layer)
        _ensure_umap(adata)

        for prob in threshold_probs:
            adata.obs[_threshold_column_name(float(prob))] = False

        overlay_path = None
        if save and outdir is not None:
            overlay_path = _cluster_overlay_path(outdir, label)
        visualization.plot_cluster_overlays(
            adata=adata,
            show=show,
            save=save,
            save_path=overlay_path,
            sample_id=label,
        )

        cluster_dict = core.get_cluster_dict(adata, cluster_label="leiden")
        per_cluster_thresholds: dict[str, dict[float, float]] = {}
        sample_stats: list[dict[str, Any]] = []

        for cluster_id, adata_cluster in cluster_dict.items():
            cluster_id_str = str(cluster_id)
            cluster_thresholds, stats_rows = _run_cluster_thresholds(
                adata_cluster,
                label=label,
                cluster_id=cluster_id_str,
                threshold_probs=threshold_probs,
                outdir=outdir,
                save=save,
                show=show,
            )
            per_cluster_thresholds[cluster_id_str] = cluster_thresholds

            for prob in threshold_probs:
                prob = float(prob)
                column_name = _threshold_column_name(prob)
                adata.obs.loc[adata_cluster.obs_names, column_name] = (
                    adata_cluster.obs[column_name].astype(bool).to_numpy()
                )

                enrichment_path = None
                if save and outdir is not None:
                    enrichment_path = _enrichment_plot_path(
                        outdir,
                        label,
                        cluster_id_str,
                        prob,
                    )
                enrichment.comparative_enrichment(
                    adata_cluster=adata_cluster,
                    outdir=outdir,
                    sample_id=label,
                    cluster_id=cluster_id_str,
                    threshold_label=column_name,
                    exclude_mt_ribo=True,
                    min_lfc=2,
                    padj=0.05,
                    top_n=6,
                    show=show,
                    save=save,
                    save_path=enrichment_path,
                )

                if save and outdir is not None:
                    _save_filtered_umap(
                        adata,
                        column_name=column_name,
                        save_path=_filtered_umap_path(
                            outdir,
                            label,
                            cluster_id_str,
                            prob,
                        ),
                        show=show,
                        cluster_obs_names=adata_cluster.obs_names,
                        cluster_id=cluster_id_str,
                        prob=prob,
                        sample_id=label,
                    )

            sample_stats.extend(stats_rows)
            all_stats.extend(stats_rows)

        for prob in threshold_probs:
            column_name = _threshold_column_name(float(prob))
            adata.obs[column_name] = adata.obs[column_name].astype(bool)

        if save and outdir is not None:
            adata.write(_adata_path(outdir, label))

        results[label] = {
            "adata": adata,
            "thresholds": per_cluster_thresholds,
            "mitochontrol_thresholds": per_cluster_thresholds,
            "threshold_stats": pd.DataFrame(sample_stats),
        }

    if save and outdir is not None:
        stats_dir = _ensure_dir(_to_path(outdir) / "mitochontrol")
        stats_path = stats_dir / "threshold_stats.csv"
        pd.DataFrame(all_stats).to_csv(
            stats_path, index=False,
        )

    return results


def single_cluster_mitochontrol(
    adata: AnnData,
    sample_id: str,
    outdir: Optional[PathLike] = None,
    show: bool = False,
    color_by: Optional[str] = "mt_frac",
    save: bool = True,
    threshold_probs: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Apply naive-Bayes thresholding without per-cluster splitting.

    Treats the entire ``adata`` as a single cluster and applies
    the same probability-based thresholding as ``get_thresholds()``.
    Useful when no Leiden clustering has been performed.

    Args:
        adata: AnnData object.  Modified in place (threshold
            columns are added to ``obs``).
        sample_id: Sample label for filenames and stats.
        outdir: Root output directory.  Required when
            ``save=True``.
        show: Display plots interactively.
        color_by: Column for the mt-by-UMI scatter plot.
            Set to ``None`` to skip.
        save: Write outputs to disk.
        threshold_probs: Posterior-probability cutoffs.
            Defaults to ``(0.8,)``.

    Returns:
        ``DataFrame`` of per-threshold statistics (one row per
        probability) with columns *sample*, *cluster*,
        *threshold_prob*, *threshold_value*, *cells_lost*, and
        *cells_retained*.

    Raises:
        ValueError: If ``save=True`` and ``outdir`` is ``None``.
    """
    if threshold_probs is None:
        threshold_probs = (0.8,)
    if "mt_frac" not in adata.obs.columns:
        core.compute_mt_fraction(adata)
    if save and outdir is None:
        raise ValueError("`outdir` is required when `save=True`.")

    visualization.plot_mt_by_umi(
        adata,
        color_by=color_by,
        show=show,
        save=save,
        outdir=outdir,
        sample_id=sample_id,
    )

    if save:
        _ensure_umap(adata)

    _, stats_rows = _run_cluster_thresholds(
        adata,
        label=sample_id,
        cluster_id=None,
        threshold_probs=threshold_probs,
        outdir=outdir,
        save=save,
        show=show,
    )
    for prob in threshold_probs:
        prob = float(prob)
        column_name = _threshold_column_name(prob)
        adata.obs[column_name] = adata.obs[column_name].astype(bool)

        enrichment_path = None
        if save and outdir is not None:
            enrichment_path = _enrichment_plot_path(
                outdir,
                sample_id,
                None,
                prob,
            )
        enrichment.comparative_enrichment(
            adata_cluster=adata,
            outdir=outdir,
            sample_id=sample_id,
            cluster_id=None,
            threshold_label=column_name,
            exclude_mt_ribo=True,
            min_lfc=2,
            padj=0.05,
            top_n=6,
            show=show,
            save=save,
            save_path=enrichment_path,
        )

        if save and outdir is not None:
            _save_filtered_umap(
                adata,
                column_name=column_name,
                save_path=_filtered_umap_path(
                    outdir,
                    sample_id,
                    None,
                    prob,
                ),
                show=show,
                cluster_obs_names=None,
                cluster_id=None,
                prob=prob,
                sample_id=sample_id,
            )

    if save and outdir is not None:
        adata.write(_adata_path(outdir, sample_id))

    stats_df = pd.DataFrame(stats_rows)
    if show and not stats_df.empty:
        print(stats_df)
    return stats_df
