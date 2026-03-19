"""Gene enrichment analysis functions."""
import logging
import os
from pathlib import Path
from textwrap import fill
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import gseapy as gp  # pyright: ignore[reportMissingImports]

from anndata import AnnData

from . import thresholding
from . import visualization

logger = logging.getLogger(__name__)


def top_up_genes(
    adata: AnnData,
    group: str,
    n_top: int = 200,
    min_lfc: float = 2.0,
    padj: float = 0.01,
) -> List[str]:
    """Return top upregulated genes for a given group from differential
    expression results.

    Extracts and filters genes from scanpy's rank_genes_groups results based
    on log fold change and adjusted p-value thresholds. Returns genes sorted
    by effect size (log fold change) and significance.

    Args:
        adata: AnnData object containing differential expression results
            from ``sc.tl.rank_genes_groups()``. Must have
            ``uns['rank_genes_groups']`` populated.
        group: Group name (cluster/cell type) within rank_genes_groups results
            to extract genes from.
        n_top: Maximum number of genes to return. If fewer genes pass filters,
            returns all that pass. Defaults to ``200``.
        min_lfc: Minimum log fold change threshold. Only genes with
            logfoldchanges > this value are included. Defaults to ``2.0``.
        padj: Maximum adjusted p-value threshold. Only genes with
            pvals_adj < this value are included. Defaults to ``0.01``.

    Returns:
        List of gene names (strings) sorted by:
            1. Descending log fold change (effect size)
            2. Ascending adjusted p-value (significance)
        Returns empty list if no genes pass the filters.

    Raises:
        KeyError: If required columns are missing from rank_genes_groups_df
            output.

    Notes:
        - Requires ``sc.tl.rank_genes_groups()`` to be run on ``adata`` first.
        - Filters genes where logfoldchanges > min_lfc AND pvals_adj < padj.
        - If fewer than n_top genes pass filters, returns all passing genes.
    """
    # Extract differential expression results for the specified group
    df = sc.get.rank_genes_groups_df(adata, group=group)

    # Validate required columns are present
    required_cols = {"logfoldchanges", "pvals_adj", "names"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise KeyError(
            f"Missing required columns in rank_genes_groups_df output: "
            f"{missing}. Available columns: {list(df.columns)}"
        )

    # Filter genes by log fold change and adjusted p-value thresholds
    df_filtered = df[
        (df["logfoldchanges"] > min_lfc) & (df["pvals_adj"] < padj)
    ]

    # Sort by descending log fold change (effect size), then ascending
    # adjusted p-value (significance)
    df_filtered = df_filtered.sort_values(
        ["logfoldchanges", "pvals_adj"], ascending=[False, True]
    )

    # Return top N genes (or all if fewer than n_top pass filters)
    return df_filtered["names"].head(n_top).tolist()


def prep_enrich_df(
    df: pd.DataFrame,
    label: str,
    top_n: int = 10,
) -> pd.DataFrame:
    """Prepare enrichment results DataFrame for visualization or export.

    Processes gene set enrichment analysis results (e.g., from Enrichr) by
    adding cluster labels, computing derived metrics (log10FDR, GeneRatio),
    and selecting top enriched terms. Returns a standardized DataFrame ready
    for plotting or further analysis.

    Args:
        df: Enrichment result table from gene set enrichment analysis tool
            (e.g., Enrichr). Must contain columns: "Term", "Adjusted P-value",
            "Overlap", "Combined Score".
        label: Group or cluster label to annotate results with. Added as
            "Cluster" column.
        top_n: Number of top enriched terms to retain, sorted by adjusted
            p-value. If fewer than top_n terms exist, returns all terms.
            Defaults to ``10``.

    Returns:
        DataFrame with standardized columns:
            - "Term": Enrichment term/gene set name
            - "Cluster": Group label (from ``label`` parameter)
            - "Adjusted P-value": FDR-adjusted p-value
            - "log10FDR": Negative log10 of adjusted p-value (-log10(FDR))
            - "GeneRatio": Ratio of overlapping genes (k/M format converted
              to float)
            - "Combined Score": Enrichment combined score

    Raises:
        KeyError: If required columns are missing from input DataFrame.

    Notes:
        - Input DataFrame is copied (not modified in place).
        - Terms are sorted by adjusted p-value (ascending) before selecting
          top_n.
        - GeneRatio is computed from "Overlap" column (format: "k/M" where k
          is number of overlapping genes, M is total genes in set).
        - Invalid or missing Overlap values result in NaN for GeneRatio.
    """
    # Validate required columns are present
    required_cols = {"Term", "Adjusted P-value", "Overlap", "Combined Score"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise KeyError(
            f"Missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # Work on a copy to avoid modifying input
    df = df.copy()
    df["Cluster"] = label

    # Sort by significance (ascending adjusted p-value) and take top N terms
    df = df.sort_values("Adjusted P-value", ascending=True).head(top_n)

    # Compute -log10(FDR) for visualization (clipping to avoid log(0))
    df["log10FDR"] = -np.log10(
        df["Adjusted P-value"].clip(lower=np.finfo(float).tiny)
    )

    # Convert "Overlap" strings (format: "k/M") into numeric gene ratio
    def _gene_ratio(overlap_str: Union[str, float]) -> float:
        """Parse overlap string "k/M" to compute gene ratio k/M.

        Args:
            overlap_str: Overlap string in format "k/M" or numeric value.

        Returns:
            Gene ratio as float, or NaN if parsing fails.
        """
        try:
            n_overlapping, n_total = str(overlap_str).split("/")
            return float(n_overlapping) / float(n_total)
        except (ValueError, TypeError, ZeroDivisionError):
            return np.nan

    df["GeneRatio"] = df["Overlap"].map(_gene_ratio)

    return df[[
        "Term", "Cluster", "Adjusted P-value", "log10FDR", "GeneRatio",
        "Combined Score"
    ]]


def comparative_enrichment(
    adata_cluster: AnnData,
    outdir: Optional[Union[str, Path]] = None,
    sample_id: Optional[str] = None,
    cluster_id: Optional[str] = None,
    suptitle: Optional[str] = None,
    threshold_label: str = "MitoChontrol",
    norm: bool = True,
    exclude_mt_ribo: bool = True,
    min_lfc: float = 2,
    padj: float = 1,
    top_n: int = 3,
    show: bool = True,
    save: bool = False,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """Perform comparative gene set enrichment between thresholded and
    retained cells.

    Compares gene expression between cells flagged by thresholding vs. cells
    retained, performs differential expression analysis, extracts top
    upregulated genes for each group, and runs GO Biological Process
    enrichment analysis via Enrichr. Visualizes results as a bubble plot
    showing enriched pathways.

    Args:
        adata_cluster: AnnData object for one cluster (cells × genes) with a
            threshold column in ``obs``. Column name should be
            ``threshold_label`` or ``f"Threshold out {threshold_label}"``.
        outdir: Output directory for saving CSV and PNG files if
            ``save=True``. Required if ``save=True`` and ``save_path`` is None.
            Defaults to ``None``.
        sample_id: Sample identifier for filenames. Required if
            ``save=True`` and ``save_path`` is None. Defaults to ``None``.
        cluster_id: Cluster identifier for filenames. Required if
            ``save=True`` and ``save_path`` is None. Defaults to ``None``.
        suptitle: Title prefix for plots. Defaults to ``None``.
        threshold_label: Label used to identify threshold column in
            ``adata_cluster.obs``. Column name will be ``threshold_label`` or
            ``f"Threshold out {threshold_label}"``. Defaults to
            ``"MitoChontrol"``.
        norm: If True, normalize and log-transform data before differential
            expression. Defaults to ``True``.
        exclude_mt_ribo: If True, exclude mitochondrial (MT-*) and ribosomal
            (RPS*, RPL*) genes before differential expression analysis.
            Defaults to ``True``.
        min_lfc: Minimum log fold change threshold for filtering top genes.
            Defaults to ``2.0``.
        padj: Maximum adjusted p-value threshold for filtering top genes.
            Defaults to ``1.0`` (no p-value filtering).
        top_n: Number of top enriched terms per group to display in plot.
            Defaults to ``3``.
        show: Display plots interactively if True; otherwise close immediately.
            Defaults to ``True``.
        save: Save gene lists (CSV) and plots (PNG) if True. Requires outdir,
            sample_id, and cluster_id unless save_path is provided. Defaults to
            ``False``.
        save_path: Optional explicit file path for saving plot. If provided,
            overrides ``outdir``-based path for plot. CSV still uses outdir
            structure. Defaults to ``None``.

    Returns:
        None: Function creates plots and optionally saves files. Returns early
        if insufficient cells in either group (< 2) or no enrichment results.

    Notes:
        - Requires threshold column to have boolean or string values that can
          be converted to categories with 'True' and 'False' categories.
        - Each group (thresholded/retained) must have at least 2 cells.
        - Uses Wilcoxon rank-sum test for differential expression.
        - Enrichment uses GO Biological Process 2021 gene sets via Enrichr
          (or local GMT file if GO_BP_GMT_PATH environment variable is
          set).
        - Plot shows bubbles sized by -log10(FDR) and colored by GeneRatio.
    """

    # Identify threshold column (try direct name first, then with prefix)
    column = thresholding._get_threshold_column_name_from_label(
        adata_cluster, threshold_label
    )

    # Check group sizes (each must have ≥2 cells for valid comparison)
    bool_col = adata_cluster.obs[column].astype(bool)
    n_thresholded = int(bool_col.sum())
    n_retained = int((~bool_col).sum())
    if n_thresholded < 2 or n_retained < 2:
        logger.info(
            "%s has fewer than 2 cells in one or both groups "
            "(thresholded=%d, retained=%d); skipping.",
            column, n_thresholded, n_retained
        )
        return None

    # Work on a copy so the original adata's column dtype is not modified
    adata_cluster_de = adata_cluster.copy()
    adata_cluster_de.obs[column] = (
        adata_cluster_de.obs[column].astype(str).astype("category")
    )

    # Remove mitochondrial and ribosomal genes if specified
    if exclude_mt_ribo:
        # Find mitochondrial genes (MT-*)
        mt_genes = adata_cluster_de.var_names[
            adata_cluster_de.var_names.str.upper().str.startswith('MT-')
        ]
        # Find ribosomal genes (RPS*, RPL*)
        rb_genes = adata_cluster_de.var_names[
            (adata_cluster_de.var_names.str.upper().str.startswith('RPS')) |
            (adata_cluster_de.var_names.str.upper().str.startswith('RPL'))
        ]
        genes_to_remove = mt_genes.union(rb_genes)
        adata_cluster_de = adata_cluster_de[
            :, ~adata_cluster_de.var_names.isin(genes_to_remove)
        ].copy()

    # Normalize and log transform
    if norm:
        sc.pp.normalize_total(adata_cluster_de, target_sum=1e4)
        sc.pp.log1p(adata_cluster_de)

    # Rank genes
    sc.tl.rank_genes_groups(
        adata_cluster_de, groupby=column, method="wilcoxon"
    )

    # Get top genes
    excluded_genes = top_up_genes(
        adata_cluster_de, 'True', n_top=200, min_lfc=min_lfc, padj=padj
    )
    included_genes = top_up_genes(
        adata_cluster_de, 'False', n_top=200, min_lfc=min_lfc, padj=padj
    )

    output_df = pd.concat(
        [
            pd.DataFrame({"gene": excluded_genes, "set": "excluded"}),
            pd.DataFrame({"gene": included_genes, "set": "included"}),
        ],
        ignore_index=True,
    )
    # Determine CSV save path
    csv_path = None
    if save:
        if outdir is not None and sample_id and cluster_id:
            out = Path(outdir) / "mitochontrol" / "enrichment"
            out.mkdir(parents=True, exist_ok=True)
            csv_path = out / f"{sample_id}_{cluster_id}_{threshold_label}.csv"
    if csv_path is not None:
        output_df.to_csv(csv_path, index=False)

    # Optional: point to a local GO BP GMT to avoid network calls.
    # Check environment variable first, then use hardcoded default if not set
    GSEAPY_GO_BP_GMT_PATH = os.getenv(
        "GO_BP_GMT_PATH",
        "/ihome/suttam/cms496/refs/GO_Biological_Process_2021.gmt"
    )

    # --- Run Enrichr only for non-empty lists ---
    def _safe_enrich(gene_list: Sequence[str], label: str) -> pd.DataFrame:
        """Run Enrichr, returning an empty DataFrame on failure."""

        if not gene_list:
            return pd.DataFrame(columns=[
                "Term", "Adjusted P-value", "Overlap", "Cluster",
                "log10FDR", "GeneRatio"
            ])
        # Prefer local GMT if available; fall back to online library name
        gene_sets_arg = None
        if GSEAPY_GO_BP_GMT_PATH and Path(GSEAPY_GO_BP_GMT_PATH).exists():
            gene_sets_arg = str(GSEAPY_GO_BP_GMT_PATH)
        else:
            gene_sets_arg = ["GO_Biological_Process_2021"]
        try:
            enr = gp.enrichr(
                gene_list=gene_list,
                gene_sets=gene_sets_arg,
                organism="Human",
                outdir=None
            )
            # gseapy can expose .results (list[dict]) or .res2d (DataFrame)
            res = getattr(enr, "results", None)
            if res is None:
                res = getattr(enr, "res2d", None)
            if (res is None or (hasattr(res, "empty") and res.empty) or
                    (hasattr(res, "__len__") and len(res) == 0)):
                return pd.DataFrame(columns=[
                    "Term", "Adjusted P-value", "Overlap", "Cluster",
                    "log10FDR", "GeneRatio"
                ])
            return prep_enrich_df(res, label=label, top_n=top_n)
        except (ValueError, ConnectionError, TimeoutError, RuntimeError) as e:
            logger.error(
                "Enrichr failed for %s using %s: %s",
                label, gene_sets_arg, e
            )
            return pd.DataFrame(columns=[
                "Term", "Adjusted P-value", "Overlap", "Cluster",
                "log10FDR", "GeneRatio"
            ])

    # Run Enrichr for both groups
    enrichment_excluded = _safe_enrich(excluded_genes, "True")
    enrichment_included = _safe_enrich(included_genes, "False")

    # Collect non-empty enrichment results
    enrichment_dfs = [
        df for df in (enrichment_excluded, enrichment_included)
        if not df.empty
    ]
    if not enrichment_dfs:
        # Nothing to plot; exit
        logger.info(
            "No enrichment results for %s Cluster %s", sample_id, cluster_id
        )
        return None

    df = pd.concat(enrichment_dfs, ignore_index=True)

    # Create ordered list of unique terms (sorted by significance)
    # This determines y-axis ordering in the plot
    terms_order = (
        df.sort_values(["Adjusted P-value"])
        .drop_duplicates(subset=["Term"])["Term"]
        .tolist()
    )

    # Create bubble plot: bubbles sized by -log10(FDR), colored by GeneRatio
    # (Metascape-style visualization)
    n_terms = len(terms_order)
    # Dynamic figure height: grow with number of terms, clamp to [3, 14]
    fig_height = max(3.0, min(1.2 + 0.5 * n_terms, 14.0))
    fig, ax = plt.subplots(figsize=(6, fig_height))

    # Ensure consistent categorical ordering on y
    # top at top
    df["Term"] = pd.Categorical(
        df["Term"], categories=terms_order[::-1], ordered=True
    )

    # Map clusters to evenly spaced x positions in [0, 1]
    # Use only clusters that actually exist
    present_clusters = sorted(df["Cluster"].unique().tolist())
    if present_clusters == ["False"]:
        x_pos = {"False": 0.5}
        xticks = [0.5]
        xticklabels = ["Retained"]
    elif present_clusters == ["True"]:
        x_pos = {"True": 0.5}
        xticks = [0.5]
        xticklabels = ["Thresholded Out"]
    else:  # both
        x_pos = {"True": 0.25, "False": 0.75}
        xticks = [0.25, 0.75]
        xticklabels = ["Thresholded Out", "Retained"]
    df["x"] = df["Cluster"].map(x_pos)

    # Evenly spaced y positions in [0, 1] (top to bottom matches
    # terms_order[::-1])
    n_terms = len(terms_order)
    y_locs = np.linspace(0.9, 0.1, n_terms)
    y_map = {term: y_locs[i] for i, term in enumerate(terms_order[::-1])}
    df["y"] = df["Term"].map(y_map)

    # Create scatter plot (bubbles)
    scatter_plot = ax.scatter(
        df["x"], df["y"],
        s=(df["log10FDR"].fillna(0) + 0.1) * 120,
        c=df["GeneRatio"], cmap="viridis", edgecolor="black", linewidth=0.5
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_yticks(y_locs)
    # wrap pathway names
    ax.set_yticklabels([fill(t, width=28) for t in terms_order[::-1]])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Set plot title
    if suptitle:
        ax.set_title(
            f"{suptitle}\nComparative Pathway Enrichment (GO BP)\n"
            f"{threshold_label}"
        )
    else:
        ax.set_title(
            f"Comparative Pathway Enrichment (GO BP)\n{threshold_label}"
        )

    # Add colorbar for GeneRatio
    cbar = plt.colorbar(scatter_plot, ax=ax)
    cbar.set_label("GeneRatio (k/M)")

    # Legend for bubble size
    for size in [1, 2, 3]:  # -log10(FDR)
        ax.scatter(
            [], [], s=size*120, edgecolors="black", c="none",
            label=f"-log10(FDR)={size}"
        )
    ax.legend(
        scatterpoints=1, frameon=True, labelspacing=1, title="Bubble size",
        loc="upper right"
    )

    plt.tight_layout()

    # Determine plot save path and save/show
    filename = (
        f"{sample_id}_{cluster_id}_{threshold_label}.png"
        if sample_id and cluster_id
        else f"enrichment_{threshold_label}.png"
    )
    plot_path = visualization._determine_plot_path(
        save, save_path, outdir, "enrichment", filename,
        sample_id, cluster_id
    )
    if plot_path is not None:
        plot_path = Path(plot_path)
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        # Also save SVG
        svg_path = plot_path.with_suffix('.svg')
        fig.savefig(svg_path, bbox_inches='tight', format='svg')
    if show:
        plt.show()
    else:
        plt.close(fig)
