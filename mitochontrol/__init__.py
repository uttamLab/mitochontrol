"""MitoChontrol — clustering-aware mtRNA thresholding.

.. include:: ../README.md

Typical end-to-end workflow:
    1. Run ``clustering()`` on each sample ``AnnData`` object.
    2. Pass the returned mapping into ``mtctrl_with_clustering(adatas=...)``.
    3. Use the returned thresholded ``AnnData`` objects and saved outputs.

Pipeline expectations:
    - ``clustering()`` detects or creates a raw-count layer, creates an
      analyzed layer if needed, assigns ``obs['leiden']``, writes clustered
      outputs, and returns both the clustered ``adata`` and layer names.
    - ``mtctrl_with_clustering()`` expects clustered sample-level ``AnnData`` objects
      or the result dictionaries returned by ``clustering()``.
    - ``mtctrl_with_clustering()`` requires raw counts and ``obs['leiden']``. It
      creates ``obs['mt_frac']`` if missing and writes boolean threshold
      columns named ``mitochontrol_threshold_out_{prob}``, where ``True``
      means thresholded out and ``False`` means retained.
    - ``mtctrl_without_clustering()`` provides the same thresholding logic
      for one already-isolated population and uses the same output layout
      without cluster-specific filename components.

Default output layout:
    - ``{outdir}/clustered/...`` for clustering artifacts
    - ``{outdir}/mitochontrol/adata/{label}.h5ad`` for thresholded outputs
    - ``{outdir}/mitochontrol/cluster_overlays/{label}.pdf``
    - ``{outdir}/mitochontrol/threshold/{label}_cluster{cluster}_{prob}.pdf``
    - ``{outdir}/mitochontrol/enrichment/{label}_cluster{cluster}_{prob}.pdf``
    - ``{outdir}/mitochontrol/filtered_umap/
      {label}_cluster{cluster}_{prob}.pdf`` (``mtctrl_with_clustering`` only)
    - ``{outdir}/mitochontrol/scatter/{label}_mt_by_umi.png`` and per-threshold
      ``*_mt_by_umi_thresh_{prob}.png`` (``mtctrl_without_clustering``)
    - ``{outdir}/mitochontrol/threshold_stats.csv``
"""

from __future__ import annotations

import os
import sys

__version__ = "0.1.0"

_CITE_URL = "https://github.com/uttamLab/mitochontrol#readme"


def _emit_startup_message() -> None:
    if os.environ.get("MITOCHONTROL_QUIET", "").lower() in ("1", "true", "yes"):
        return
    print(
        f"mitochontrol v{__version__}\n"
        "If MitoChontrol is useful in your work, please cite it.\n"
        f"See {_CITE_URL} for citation information.\n",
        file=sys.stderr,
        end="",
    )


# Core functions
from .core import (
    transfer_metadata,
    compute_mt_fraction,
    get_cluster_dict,
    get_mt_dict,
    get_label_maps,
    assign_celltype_colors,
)

# Model fitting functions
from .models import (
    fit_empirical_histogram,
    kl_divergence,
    compute_kl_divergences,
    initialize_gmm_params,
    online_em_gmm,
    online_em_nbm,
    online_em_beta,
    online_em_poisson,
)

# Visualization functions
from .visualization import (
    plot_mt_dist,
    plot_mt_by_umi,
    plot_cluster_overlays,
    plot_mixture_fits,
    plot_threshold_umap,
)

# Thresholding functions
from .thresholding import (
    naive_bayes_threshold,
    manual_threshold,
)

# Enrichment functions
from .enrichment import (
    top_up_genes,
    prep_enrich_df,
    comparative_enrichment,
)

# Pipeline functions
from .pipelines import (
    mtctrl_with_clustering,
    mtctrl_without_clustering,
)

# Clustering pipeline functions
from .clustering import (
    clustering,
    construct_neighbors,
    optimal_res,
    cluster_data,
    differential_expression,
    read_marker_genes,
    assign_celltypes,
)

__all__ = [
    # Core
    "transfer_metadata",
    "compute_mt_fraction",
    "get_cluster_dict",
    "get_mt_dict",
    "get_label_maps",
    "assign_celltype_colors",
    # Models
    "fit_empirical_histogram",
    "kl_divergence",
    "compute_kl_divergences",
    "initialize_gmm_params",
    "online_em_gmm",
    "online_em_nbm",
    "online_em_beta",
    "online_em_poisson",
    # Visualization
    "plot_mt_dist",
    "plot_mt_by_umi",
    "plot_cluster_overlays",
    "plot_mixture_fits",
    "plot_threshold_umap",
    # Thresholding
    "naive_bayes_threshold",
    "manual_threshold",
    # Enrichment
    "top_up_genes",
    "prep_enrich_df",
    "comparative_enrichment",
    # Pipelines
    "mtctrl_with_clustering",
    "mtctrl_without_clustering",
    # Clustering pipeline
    "clustering",
    "construct_neighbors",
    "optimal_res",
    "cluster_data",
    "differential_expression",
    "read_marker_genes",
    "assign_celltypes",
]

_emit_startup_message()
