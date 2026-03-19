"""Visualization functions for mitochondrial RNA fraction analysis."""
import logging
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.stats import nbinom, beta as beta_dist, poisson

from anndata import AnnData

from . import core
from . import models

logger = logging.getLogger(__name__)


def _determine_plot_path(
        save: bool,
        save_path: Optional[Union[str, Path]],
        outdir: Optional[Union[str, Path]],
        subdir: str,
        filename: str,
        sample_id: Optional[str] = None,
        cluster_id: Optional[str] = None
) -> Optional[Path]:
    """Determine plot save path from save_path or outdir-based structure.

    Returns None if save=False or if required parameters are missing.
    """
    if not save:
        return None
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    if outdir is not None:
        base = Path(outdir) / "mitochontrol" / subdir
        base.mkdir(parents=True, exist_ok=True)
        return base / filename
    return None


def _save_and_show_figure(
        fig: plt.Figure,
        save: bool,
        save_path: Optional[Union[str, Path]],
        outdir: Optional[Union[str, Path]],
        subdir: str,
        filename: str,
        show: bool,
        sample_id: Optional[str] = None,
        cluster_id: Optional[str] = None,
        log_msg: Optional[str] = None
) -> None:
    """Save figure and show/close based on parameters.

    Handles save_path override, outdir-based paths, logging, and show/close.
    Saves to both PNG and SVG formats.
    """
    plot_path = _determine_plot_path(
        save, save_path, outdir, subdir, filename, sample_id, cluster_id
    )
    if plot_path is not None:
        plot_path = Path(plot_path)
        # Save PNG
        fig.savefig(plot_path, bbox_inches='tight', dpi=300)
        # Save SVG (same path with .svg extension)
        svg_path = plot_path.with_suffix('.svg')
        fig.savefig(svg_path, bbox_inches='tight', format='svg')
        if log_msg:
            logger.info(log_msg, plot_path)
            logger.info(log_msg.replace('%s', '%s (SVG)'), svg_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_mt_dist(adata: AnnData,
                 outdir: Optional[Union[str, Path]] = None,
                 sample_id: Optional[str] = None,
                 title: Optional[str] = None,
                 show: bool = True,
                 save: bool = False,
                 save_path: Optional[Union[str, Path]] = None,
                 figsize: Tuple[int, int] = (15, 10)) -> None:
    """Plot a histogram of mitochondrial RNA fraction for the full sample.

    Creates a probability histogram showing the distribution of mitochondrial
    RNA fractions across all cells in the sample. The x-axis is fixed to the
    range [0, 100] percent.

    Args:
        adata: Annotated data matrix containing an ``obs['mt_frac']`` column.
        outdir: Directory where plots should be saved if ``save=True``.
            Required if ``save=True`` and ``save_path`` is None.
        sample_id: Identifier of the sample being plotted. Used in default
            title and filename if provided. Required if ``save=True`` and
            ``save_path`` is None.
        title: Custom figure title. If None, generates a default title.
        show: Whether to display the plot interactively. Defaults to ``True``.
        save: Whether to save the plot (PNG and SVG). Defaults to ``False``.
        save_path: Optional explicit file path for saving plot. If provided,
            overrides ``outdir``-based path. Defaults to ``None``.
        figsize: Figure size in inches. Defaults to ``(15, 10)``.

    Returns:
        None: Displays or saves a histogram plot.

    Raises:
        KeyError: If ``obs['mt_frac']`` column is not present in ``adata``.

    Notes:
        - If ``save=True`` but ``outdir`` or ``sample_id`` is None, the plot
          will not be saved (no error is raised).
        - Saved plots are written to
          ``{outdir}/mitochontrol/mt_distribution/``.
        - The histogram uses probability density (normalized to sum to 1).
    """
    # Extract mitochondrial fraction data
    if "mt_frac" not in adata.obs:
        raise KeyError(
            "Column 'mt_frac' not found in adata.obs. "
            "Run compute_mt_fraction() first."
        )

    full_sample_mt_frac = pd.Series(
        adata.obs["mt_frac"].values, index=adata.obs_names
    ).astype(float)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot histogram with 50 bins (51 edges) evenly spaced from 0 to 100%
    sns.histplot(
        full_sample_mt_frac,
        bins=np.linspace(0, 100, 51),
        stat="probability",
        color="lightgray",
        edgecolor="grey",
        label="Full Sample",
        ax=ax,
    )

    # Title and axis labels
    default_title = (
        f"Distribution of mtRNA: {sample_id}" if sample_id
        else "Distribution of mtRNA"
    )
    ax.set_title(title or default_title, fontsize=16)
    ax.set_xlabel("Mitochondrial RNA Fraction (%)")
    ax.set_ylabel("Probability")
    ax.grid(True)
    ax.set_xlim(0, 100)
    fig.tight_layout()

    # Save or show
    filename = f"{sample_id}.png" if sample_id else "mt_distribution.png"
    _save_and_show_figure(
        fig, save, save_path, outdir, "mt_distribution", filename,
        show, sample_id, None, "Saved MT distribution plot to %s"
    )
    if save and save_path is None and (outdir is None or sample_id is None):
        logger.warning(
            "Plot not saved: save=True but outdir or sample_id is None"
        )


def plot_mt_by_umi(
    adata: AnnData,
    color_by: Optional[str] = None,             # e.g., "leiden" or "cell_type"
    log_y: bool = True,
    sample: Optional[int] = 50_000,  # subsample for speed; None = all
    alpha: float = 0.6,
    point_size: float = 8.0,
    title: Optional[str] = None,
    show: bool = True,
    save: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    outdir: Optional[Union[str, Path]] = None,
    sample_id: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    """Scatter plot of mtRNA fraction (%) vs total UMI counts per cell.

    Creates a scatter plot with mitochondrial RNA fraction on the x-axis and
    total UMI counts on the y-axis. Supports optional coloring by metadata
    columns or gene expression, and automatic subsampling for large datasets.

    Args:
        adata: AnnData with ``.obs['mt_frac']`` (percent). For UMI counts,
            uses ``.obs['total_counts']`` or ``.obs['n_counts']`` if available,
            otherwise sums the expression matrix ``X``.
        color_by: Optional column in ``adata.obs`` to color by (e.g., "leiden",
            "cell_type"), OR a gene name in ``adata.var_names`` for continuous
            expression coloring. If None, points are not colored.
        log_y: Whether to use log scale on the UMI axis. Defaults to ``True``.
        sample: Randomly subsample this many cells for plotting (None = use
            all cells). Defaults to 50,000 for performance with large datasets.
        alpha: Point transparency (0.0 to 1.0). Defaults to ``0.6``.
        point_size: Marker size in points. Defaults to ``8.0``.
        title: Optional title override. If None, generates a default title.
        show: Whether to display the plot interactively. Defaults to ``True``.
        save: Whether to save the plot (PNG and SVG). Defaults to ``False``.
        save_path: Optional explicit file path for saving. If provided and
            ``save=True``, overrides the default ``outdir``-based path.
        outdir: Root directory for saving when using default path structure.
            Required if ``save=True`` and ``save_path`` is None.
        sample_id: Optional identifier used in default filename and title.
        figsize: Figure size in inches (width, height). Defaults to ``(8, 6)``.

    Returns:
        None: Displays or saves a scatter plot.

    Raises:
        KeyError: If ``obs['mt_frac']`` is not present in ``adata``, or if
            ``color_by`` is specified but not found in ``adata.obs`` or
            ``adata.var_names``.
        ValueError: If no valid cells remain after filtering NaNs/Infs, or if
            ``save=True`` but ``outdir`` is None and ``save_path`` is None.

    Notes:
        - Cells with NaN or Inf values in mt_frac or UMI are excluded.
        - If both ``save_path`` and ``outdir`` are provided with ``save=True``,
          the plot is saved to both locations.
        - Subsampling uses a fixed random seed (0) for reproducibility.
        - X-axis is fixed to [0, 100] percent.
        - For log scale, y-axis minimum is set to max(1, min(UMI)) to avoid
          log(0) errors.
    """
    # Validate x (mt_frac)
    if "mt_frac" not in adata.obs:
        raise KeyError(
            "Column 'mt_frac' not found in adata.obs. "
            "Run compute_mt_fraction() first."
        )

    # Determine UMI counts (prefer precomputed, fallback to summing X)
    if "total_counts" in adata.obs:
        umi = adata.obs["total_counts"].to_numpy()
    elif "n_counts" in adata.obs:
        umi = adata.obs["n_counts"].to_numpy()
    else:
        # Compute total counts by summing expression matrix
        X = adata.X
        if sparse.issparse(X):
            umi = np.asarray(X.sum(axis=1)).ravel()
        else:
            umi = X.sum(axis=1)

    # Assemble plotting frame
    df = pd.DataFrame(
        {
            "mt_frac": pd.to_numeric(adata.obs["mt_frac"], errors="coerce"),
            "UMI": pd.to_numeric(umi, errors="coerce")
        },
        index=adata.obs_names,
    )

    # Configure coloring (metadata column, gene expression, or none)
    color_is_numeric_expr = False
    color_label = None

    if color_by is not None:
        if color_by in adata.obs:
            # Color by metadata column (categorical or numeric)
            df[color_by] = adata.obs[color_by]
            # Convert to categorical if non-numeric or if explicitly 'leiden'
            if (not pd.api.types.is_numeric_dtype(df[color_by]) or
                    color_by == 'leiden'):
                df[color_by] = df[color_by].astype("category")
            color_label = color_by

        elif color_by in adata.var_names:
            # Color by gene expression (continuous)
            expr = adata[:, color_by].X
            if sparse.issparse(expr):
                expr = expr.toarray().ravel()
            else:
                expr = np.asarray(expr).ravel()
            df[color_by] = pd.to_numeric(expr, errors="coerce")
            color_is_numeric_expr = True
            color_label = color_by
        else:
            raise KeyError(
                f"Requested color_by='{color_by}' not found in adata.obs or "
                f"adata.var_names."
            )

    # Clean data: remove Inf values and drop rows with NaN in mt_frac or UMI
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["mt_frac", "UMI"]
    )
    if df.empty:
        raise ValueError(
            "No valid cells to plot after filtering NaNs/Infs."
        )

    # Optional subsampling for large datasets (fixed seed for reproducibility)
    if sample is not None and len(df) > sample:
        df = df.sample(n=sample, random_state=0)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    if color_label is None:
        # no coloring
        sns.scatterplot(
            data=df,
            x="mt_frac",
            y="UMI",
            s=point_size,
            alpha=alpha,
            linewidth=0,
            ax=ax,
        )

    elif color_is_numeric_expr:
        # continuous gene expression coloring
        scat = ax.scatter(
            df["mt_frac"],
            df["UMI"],
            c=df[color_label],
            s=point_size,
            alpha=alpha,
            linewidths=0,
            cmap="viridis",
        )
        cbar = plt.colorbar(scat, ax=ax)
        cbar.set_label(color_label)

    else:
        # categorical / obs-based coloring
        sns.scatterplot(
            data=df,
            x="mt_frac",
            y="UMI",
            hue=color_label,
            s=point_size,
            alpha=alpha,
            linewidth=0,
            ax=ax,
        )

    # Axis labels and limits
    ax.set_xlabel("Mitochondrial RNA Fraction (%)")
    ax.set_ylabel("UMI (total counts)")
    ax.set_xlim(0, 100)  # x-axis fixed to 0–100%
    # Set y-axis minimum: for log scale, ensure >= 1 to avoid log(0)
    bottom = 1 if log_y else 0
    bottom = max(bottom, min(df["UMI"]))
    ax.set_ylim(bottom=bottom)
    if log_y:
        ax.set_yscale("log")

    # Title
    if title:
        ax.set_title(title)
    elif sample_id:
        ax.set_title(f"mtRNA vs UMI — {sample_id}")
    else:
        ax.set_title("mtRNA vs UMI")

    # Legend (only for categorical coloring; continuous uses colorbar)
    if color_label is not None and not color_is_numeric_expr:
        ax.legend(
            title=color_label, bbox_to_anchor=(1.02, 1),
            loc="upper left", borderaxespad=0.0
        )
    else:
        # Remove default legend for uncolored or continuous plots
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    plt.tight_layout()

    # Save plot (to default location and/or explicit path)
    filename = (
        f"{sample_id}_mt_by_umi.png" if sample_id
        else "mt_by_umi.png"
    )
    if save and not save_path:
        if outdir is None:
            raise ValueError(
                "outdir must be provided when save=True and save_path is None."
            )
        out = Path(outdir) / "mitochontrol" / "scatter"
        out.mkdir(parents=True, exist_ok=True)
        png_path = out / filename
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        # Also save SVG
        svg_path = png_path.with_suffix('.svg')
        fig.savefig(svg_path, bbox_inches="tight", format='svg')

    if save and save_path:
        # Save to explicit path (may also save to default location above)
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path_obj, dpi=300, bbox_inches="tight")
        # Also save SVG
        svg_path = save_path_obj.with_suffix('.svg')
        fig.savefig(svg_path, bbox_inches="tight", format='svg')

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_cluster_overlays(adata: AnnData,
                          outdir: Optional[Union[str, Path]] = None,
                          sample_id: Optional[str] = None,
                          title: Optional[str] = None,
                          label_map: Optional[Mapping[
                              str, Mapping[str, str]]] = None,
                          bins: int = 50,
                          show: bool = True,
                          save: bool = False,
                          save_path: Optional[Union[str, Path]] = None,
                          figsize: Tuple[int, int] = (15, 10)
                          ) -> Optional[Dict[str, Tuple[
                              float, float, float, float]]]:
    """Plot histograms of mitochondrial RNA fraction per cluster overlaid on
    the full sample.

    Creates a grid of subplots, one per cluster, showing the cluster's mtRNA
    distribution overlaid on the full sample distribution. The full sample is
    shown in the background (gray), and each cluster is shown in the
    foreground with cluster-specific coloring.

    Args:
        adata: AnnData with clusters in ``.obs['leiden']`` and
            ``.obs['mt_frac']`` column.
        outdir: Output directory if ``save=True``. Required if ``save=True``
            and ``save_path`` is None.
        sample_id: Sample identifier for filenames and titles. Required if
            ``save=True`` and ``save_path`` is None.
        title: Optional figure title. If None, uses default title.
        label_map: Optional nested mapping ``{sample_id: {cluster_id:
            cell_type}}`` for cell type labels and coloring.
        bins: Number of histogram bins. Defaults to ``50``.
        show: Whether to display the plot interactively. Defaults to ``True``.
        save: Whether to save the plot (PNG and SVG). Defaults to ``False``.
        save_path: Optional explicit file path for saving plot. If provided,
            overrides ``outdir``-based path. Defaults to ``None``.
        figsize: Figure size in inches (width, height).
            Defaults to ``(15, 10)``.

    Returns:
        Dictionary mapping cell type names to RGBA color tuples, or None if
        ``label_map`` is not provided. This can be used to maintain consistent
        coloring across plots.

    Raises:
        KeyError: If ``obs['mt_frac']`` or ``obs['leiden']`` is not present in
            ``adata``.

    Notes:
        - The full sample histogram uses "probability" stat (normalized to sum
          to 1), while cluster histograms use "density" stat.
        - Subplot grid size is automatically calculated to be approximately
          square based on the number of clusters.
        - If ``save=True`` but ``outdir`` or ``sample_id`` is None, the plot
          will not be saved (no error is raised).
        - Saved plots are written to
          ``{outdir}/mitochontrol/cluster_overlays/``.
    """

    # Validate required columns
    if "mt_frac" not in adata.obs:
        raise KeyError(
            "Column 'mt_frac' not found in adata.obs. "
            "Run compute_mt_fraction() first."
        )
    if "leiden" not in adata.obs:
        raise KeyError(
            "Column 'leiden' not found in adata.obs. "
            "Clusters are required for this plot."
        )

    # Extract mtRNA fractions for full sample and per cluster
    full_sample_mt_frac = pd.Series(
        data=adata.obs["mt_frac"].values, index=adata.obs_names
    ).astype(float)
    cluster_dict = core.get_cluster_dict(adata)
    mt_dict = core.get_mt_dict(cluster_dict)

    # Handle empty case
    if not mt_dict:
        logger.warning(
            "No clusters found in adata. Nothing to plot."
        )
        return None

    # Calculate subplot grid dimensions (approximately square)
    n_clusters = len(mt_dict)
    n_cols = math.ceil(np.sqrt(n_clusters))
    n_rows = math.ceil(n_clusters / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=figsize, sharex=True, sharey=True
    )
    fig.suptitle(title or "Distribution of mtRNA by Celltype", fontsize=16)
    axes = np.asarray(axes).ravel()

    if label_map:
        celltype_coloring = core.assign_celltype_colors(
            cluster_dict, label_map, sample_id
        )
    else:
        celltype_coloring = None

    # Precompute label map for this sample (convert keys to strings)
    sample_map = (label_map or {}).get(sample_id, {})
    sample_map_str_keys = {str(k): v for k, v in sample_map.items()}

    # Plot each cluster's distribution overlaid on full sample
    last_used_index = -1
    for i, (cluster_label, cluster_data) in enumerate(mt_dict.items()):
        ax = axes[i]
        last_used_index = i

        # Full sample histogram in background (probability normalization)
        sns.histplot(
            full_sample_mt_frac,
            bins=bins,
            stat="probability",
            color="lightgray",
            edgecolor="grey",
            label="All Cells",
            ax=ax,
        )

        # Cluster histogram in foreground (density normalization)
        legend_label = sample_map_str_keys.get(
            str(cluster_label), f'Cluster {cluster_label}'
        )
        cluster_color = (
            celltype_coloring[legend_label] if celltype_coloring
            else "lightskyblue"
        )
        sns.histplot(
            cluster_data,
            bins=bins,
            stat="density",
            color=cluster_color,
            edgecolor="grey",
            label=legend_label,
            ax=ax,
        )

        # Set subplot title and labels
        if label_map:
            ax.set_title(f"Cluster {cluster_label}: {legend_label}s")
        else:
            ax.set_title(f"Cluster {cluster_label}")
        ax.set_xlabel("Mitochondrial RNA Fraction (%)")
        ax.set_ylabel("Density")
        ax.grid(True, color="lightgrey", linewidth=0.5)
        ax.legend()
        # Set x-axis limit to full mtRNA range (0-100%)
        ax.set_xlim(0, min(100, float(full_sample_mt_frac.max())))

    # Remove unused subplots
    for j in range(last_used_index + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()

    # Save plot if requested
    filename = f"{sample_id}.png" if sample_id else "cluster_overlays.png"
    _save_and_show_figure(
        fig, save, save_path, outdir, "cluster_overlays", filename,
        show, sample_id, None, "Saved cluster overlays to %s"
    )
    if save and save_path is None and (outdir is None or sample_id is None):
        logger.warning(
            "Plot not saved: save=True but outdir/sample_id and "
            "save_path are None"
        )

    return celltype_coloring


def plot_mixture_fits(
    data: np.ndarray,
    outdir: Optional[Union[str, Path]] = None,
    sample_id: Optional[str] = None,
    cluster_id: Optional[str] = None,
    label_map: Optional[
        Mapping[str, Mapping[str, str]]
    ] = None,
    celltype_coloring: Optional[Mapping[str, Any]] = None,
    gmm_model: Optional[Mapping[str, Any]] = None,
    nb_model: Optional[Mapping[str, Any]] = None,
    beta_model: Optional[Mapping[str, Any]] = None,
    poisson_model: Optional[Mapping[str, Any]] = None,
    title: Optional[str] = None,
    save: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> Dict[str, float]:
    """Plot mixture model fits (Gaussian, Negative Binomial, Beta, Poisson)
    in 2×2 subplots.

    Creates a 2×2 grid of subplots showing empirical histograms overlaid
    with fitted mixture model components and composite densities. Each
    subplot displays a different distribution type with KL divergence scores.

    Args:
        data: 1D array of mtRNA fractions (percentages).
        outdir: Output directory for saving if ``save=True``. Required if
            ``save=True`` and ``save_path`` is None. Defaults to ``None``.
        sample_id: Sample identifier for title and filename. Required if
            ``save=True`` and ``save_path`` is None. Defaults to ``None``.
        cluster_id: Cluster identifier for title and filename. Required if
            ``save=True`` and ``save_path`` is None. Defaults to ``None``.
        label_map: Mapping from sample to cluster to cell type:
            ``{sample: {cluster: cell_type}}``. Used for title and color
            lookup. Defaults to ``None``.
        celltype_coloring: Optional mapping from cell type to color:
            ``{cell_type: color}``. Used to color histograms. Defaults to
            ``None``.
        gmm_model: Optional fitted Gaussian mixture model dictionary with
            keys "components", "weights", "k". Defaults to ``None``.
        nb_model: Optional fitted Negative Binomial mixture model dictionary
            with keys "components", "weights", "k". Defaults to ``None``.
        beta_model: Optional fitted Beta mixture model dictionary with keys
            "components", "weights", "k". Defaults to ``None``.
        poisson_model: Optional fitted Poisson mixture model dictionary with
            keys "components", "weights", "k". Defaults to ``None``.
        title: Suptitle override. If None, auto-generated from sample_id,
            cluster_id, and label_map. Defaults to ``None``.
        save: Save figure (PNG and SVG) if True. Requires
            outdir, sample_id, and cluster_id unless
            save_path is provided. Defaults to ``False``.
        save_path: Optional explicit file path for saving plot. If provided,
            overrides ``outdir``-based path. Defaults to ``None``.
        show: Display figure if True; otherwise close immediately. Defaults
            to ``True``.

    Returns:
        Dictionary mapping model names (e.g., "Gaussian", "NegBinomial") to
        their KL divergence values. Only includes models that were provided
        (non-None).

    Raises:
        ValueError: If data is empty.
        KeyError: If label_map or celltype_coloring keys are missing when
            accessed (if sample_id/cluster_id provided but not in mappings).
    """
    if data.size == 0:
        raise ValueError("`data` must be non-empty.")

    # Default color; override if mapping is provided
    celltype_color = "lightgrey"
    if (celltype_coloring and label_map and sample_id and cluster_id and
            sample_id in label_map and cluster_id in label_map[sample_id]):
        celltype = label_map[sample_id][cluster_id]
        if celltype in celltype_coloring:
            celltype_color = celltype_coloring[celltype]

    # Compute empirical histogram (used for KL divergence computation)
    pdf_emp, bin_centers, bin_edges = models.fit_empirical_histogram(data)
    # Continuous x-axis for PDF evaluation (Gaussian)
    data_max = float(np.max(data))
    x_vals = np.linspace(0, data_max, 500)
    # Discrete integer bins for PMF evaluation (Negative Binomial, Poisson)
    # Scale by 100 to preserve precision when rounding to integers
    bin_width = x_vals[1] - x_vals[0]  # Approximate bin width for PMF scaling
    max_int = int(np.ceil(data_max * 100))  # PMF domain [0, max_int]
    x_int = np.arange(0, max_int + 1)
    x_scaled = x_int / 100.0  # Convert back to [0, max_data] space

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()
    # Set title: use provided title or generate from metadata
    if title:
        fig.suptitle(title, fontsize=16)
    else:
        # Generate title from sample_id, cluster_id, and label_map
        if (sample_id and cluster_id and label_map and
                sample_id in label_map and
                cluster_id in label_map[sample_id]):
            celltype = label_map[sample_id][cluster_id]
            title = (
                f'Mixture Model Fits\n{sample_id}: Cluster {cluster_id} - '
                f'{celltype}s'
            )
        else:
            title = 'Mixture Model Fits'
        fig.suptitle(title, fontsize=16)

    # Compute KL divergences for all provided models
    kl_scores = models.compute_kl_divergences(
        data=data,
        gmm_model=gmm_model,
        nb_model=nb_model,
        beta_model=beta_model,
        poisson_model=poisson_model
    )

    # 🎨 Custom color palettes
    gaussian_colors = ["#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c"]
    composite_color_gaussian = "#08519c"

    negbin_colors = ["#c7e9c0", "#a1d99b", "#74c476", "#41ab5d", "#238b45"]
    composite_color_negbin = "#006d2c"

    beta_colors = ["#bcbddc", "#9e9ac8", "#807dba", "#6a51a3", "#54278f"]
    composite_color_beta = "#54278f"

    poisson_colors = ["#fbb4b9", "#f768a1", "#dd3497", "#ae017e", "#7a0177"]
    composite_color_poisson = "#49006a"

    # --------------------
    # 1. Gaussian Mixture
    # --------------------
    ax = axs[0]
    sns.histplot(
        data, bins=50, stat='density', alpha=0.5, color=celltype_color,
        edgecolor='darkgrey', label='Distribution of mtRNA', ax=ax
    )

    if gmm_model:
        gmm_pdf = np.zeros_like(x_vals)
        components = gmm_model["components"]
        weights = gmm_model["weights"]

        for i, (mean, var) in enumerate(components):
            std = np.sqrt(var)
            weight = weights[i]
            comp_pdf = (
                weight * (1 / (np.sqrt(2 * np.pi) * std)) *
                np.exp(-(x_vals - mean)**2 / (2 * std**2))
            )
            gmm_pdf += comp_pdf
            color = gaussian_colors[i % len(gaussian_colors)]
            ax.plot(
                x_vals, comp_pdf, color=color, linewidth=2.5,
                label=f'Component {i + 1}'
            )

        ax.plot(
            x_vals, gmm_pdf, linestyle='--', color=composite_color_gaussian,
            lw=1, label='Total Density'
        )
        kl_gmm = kl_scores.get("Gaussian", 0.0)
        ax.set_xlabel("Mitochondrial RNA Fraction (%)")
        ax.set_title(f"Gaussian\nKL Divergence={kl_gmm:.3f}")
        ax.legend()
        ax.set_xlim(0, 100)

    # ------------------------------
    # 2. Negative Binomial Mixture
    # ------------------------------
    ax = axs[1]
    sns.histplot(
        data, bins=50, stat='density', alpha=0.5, color=celltype_color,
        edgecolor='darkgrey', label='Distribution of mtRNA', ax=ax
    )

    if nb_model:
        nb_pdf = np.zeros_like(x_scaled, dtype=float)
        # Use model weights if available, otherwise assume uniform
        weights_nb = nb_model.get(
            "weights", np.ones(nb_model["k"]) / nb_model["k"]
        )
        for i, (r, p) in enumerate(nb_model["components"]):
            raw_pmf = nbinom.pmf(x_int, r, p)
            weight = (
                weights_nb[i] if i < len(weights_nb)
                else 1.0 / nb_model["k"]
            )
            # Convert PMF to density by dividing by bin width
            comp_pdf = weight * raw_pmf / bin_width
            nb_pdf += comp_pdf
            color = negbin_colors[i % len(negbin_colors)]
            ax.plot(
                x_scaled, comp_pdf, color=color,
                label=f'Component {i + 1}', linewidth=2.5
            )

        ax.plot(
            x_scaled, nb_pdf, linestyle='--', color=composite_color_negbin,
            lw=1, label='Total Density'
        )
        kl_nb = kl_scores.get("NegBinomial", 0.0)
        ax.set_title(f"Negative Binomial\nKL={kl_nb:.3f}")
        ax.legend()
        ax.set_xlim(0, 100)

    # --------------------
    # 3. Beta Mixture
    # --------------------
    ax = axs[2]

    # Scale data to [0, 1] for Beta distribution support
    data_max = float(np.max(data))
    if data_max > 0:
        data_scaled = data / data_max
    else:
        data_scaled = data  # All zeros; avoid division by zero
    beta_x_vals = np.linspace(0, 1, 500)

    # Plot histogram on [0, 1]
    sns.histplot(
        data_scaled, bins=50, stat='density', alpha=0.5,
        color=celltype_color, edgecolor='darkgrey',
        label='Distribution of mtRNA', ax=ax
    )

    # Plot Beta components
    if beta_model:
        beta_pdf = np.zeros_like(beta_x_vals)
        # Use model weights if available, otherwise assume uniform
        weights_beta = beta_model.get(
            "weights", np.ones(beta_model["k"]) / beta_model["k"]
        )
        for i, (a, b) in enumerate(beta_model["components"]):
            weight = (
                weights_beta[i] if i < len(weights_beta)
                else 1.0 / beta_model["k"]
            )
            comp_pdf = weight * beta_dist.pdf(beta_x_vals, a, b)
            beta_pdf += comp_pdf
            color = beta_colors[i % len(beta_colors)]
            ax.plot(
                beta_x_vals, comp_pdf, color=color, linewidth=2.5,
                label=f'Component {i + 1}'
            )

        # Plot composite line
        ax.plot(
            beta_x_vals, beta_pdf, linestyle='--', color=composite_color_beta,
            lw=1, label='Total Density'
        )

        kl_beta = kl_scores.get("Beta", 0.0)
        ax.set_title(f"Beta\nKL={kl_beta:.3f}")
        ax.legend()

    # ---------------------
    # 4. Poisson Mixture
    # ---------------------
    ax = axs[3]
    sns.histplot(
        data, bins=50, stat='density', alpha=0.5, color=celltype_color,
        edgecolor='darkgrey', label='Distribution of mtRNA', ax=ax
    )

    if poisson_model:
        pois_pdf = np.zeros_like(x_scaled, dtype=float)
        # Use model weights if available, otherwise assume uniform
        weights_pois = poisson_model.get(
            "weights", np.ones(poisson_model["k"]) / poisson_model["k"]
        )
        for i, lam in enumerate(poisson_model["components"]):
            raw_pmf = poisson.pmf(x_int, lam)
            weight = (
                weights_pois[i] if i < len(weights_pois)
                else 1.0 / poisson_model["k"]
            )
            # Convert PMF to density by dividing by bin width
            comp_pdf = weight * raw_pmf / bin_width
            pois_pdf += comp_pdf
            color = poisson_colors[i % len(poisson_colors)]
            ax.plot(
                x_scaled, comp_pdf, color=color, linewidth=2.5,
                label=f'Component {i + 1}'
            )

        ax.plot(
            x_scaled, pois_pdf, linestyle='--', color=composite_color_poisson,
            lw=1, label='Total Density'
        )
        kl_pois = kl_scores.get("Poisson", 0.0)
        ax.set_title(f"Poisson\nKL={kl_pois:.3f}")
        ax.legend()
        ax.set_xlim(0, 100)

    plt.tight_layout()

    filename = (
        f"{sample_id}_{cluster_id}.png"
        if sample_id and cluster_id
        else "mixture_fits.png"
    )
    _save_and_show_figure(
        fig, save, save_path, outdir, "MM_fits", filename,
        show, sample_id, cluster_id, "Saved mixture fit plot to %s"
    )

    return kl_scores


def plot_threshold_umap(
    adata: AnnData,
    threshold_column: str,
    cluster_obs_names: Optional[pd.Index] = None,
    cluster_id: Optional[str] = None,
    prob: Optional[float] = None,
    sample_id: Optional[str] = None,
    save: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (8, 6),
) -> None:
    """Plot a UMAP colored by threshold status with clear category distinction.

    When ``cluster_obs_names`` is provided the plot uses three categories so
    that "other-cluster" cells are visually separated from retained and flagged
    cells within the target cluster:

    * **Other** (light gray) — cells outside the target cluster
    * **Retained** (steel blue) — cells inside the cluster that passed
    * **Flagged** (tomato red) — cells inside the cluster that were flagged

    When ``cluster_obs_names`` is ``None`` the whole sample is treated as one
    group and only two categories are used (Retained / Flagged).

    Args:
        adata: Full-sample AnnData with a UMAP in ``obsm['X_umap']`` and a
            boolean threshold column in ``obs``.
        threshold_column: Name of the boolean column in ``adata.obs`` where
            ``True`` means flagged for removal.
        cluster_obs_names: ``obs_names`` of cells belonging to the target
            cluster.  If ``None``, all cells are considered part of the target.
        cluster_id: Cluster label used in the plot title.
        prob: Threshold probability used in the plot title.
        sample_id: Sample label used in the plot title.
        save: Whether to save the figure (PDF and SVG).
        save_path: Explicit path for saving.
        show: Whether to display the figure interactively.
        figsize: Figure size in inches.

    Notes:
        A temporary column ``_threshold_umap_cat`` is added to
        ``adata.obs`` during plotting and removed before the
        function returns.
    """
    import scanpy as sc

    cat_col = "_threshold_umap_cat"

    if cluster_obs_names is not None:
        in_cluster = adata.obs_names.isin(cluster_obs_names)
    else:
        in_cluster = pd.Series(True, index=adata.obs_names)

    flagged = adata.obs[threshold_column].astype(bool)

    cat_labels = ["Other", "Retained", "Flagged"]
    categories = pd.Categorical(
        ["Other"] * adata.n_obs, categories=cat_labels,
    )
    cat_series = pd.Series(categories, index=adata.obs_names)
    cat_series[in_cluster & ~flagged] = "Retained"
    cat_series[in_cluster & flagged] = "Flagged"
    adata.obs[cat_col] = cat_series

    palette = {"Other": "#d9d9d9", "Retained": "#4a90d9", "Flagged": "#e24a33"}

    title_parts = []
    if sample_id:
        title_parts.append(sample_id)
    if cluster_id is not None:
        title_parts.append(f"cluster {cluster_id}")
    if prob is not None:
        title_parts.append(f"prob {prob:g}")
    title = "Threshold UMAP"
    if title_parts:
        title += f" — {', '.join(title_parts)}"

    n_flagged = int((in_cluster & flagged).sum())
    n_cluster = int(in_cluster.sum())
    title += f"\n{n_flagged}/{n_cluster} cells flagged"

    fig = sc.pl.umap(
        adata,
        color=cat_col,
        palette=palette,
        title=title,
        frameon=False,
        show=False,
        return_fig=True,
        size=20,
    )

    if save and save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
        svg_path = save_path.with_suffix(".svg")
        fig.savefig(svg_path, bbox_inches="tight", format="svg")

    if show:
        plt.show()
    else:
        plt.close(fig)

    if cat_col in adata.obs.columns:
        del adata.obs[cat_col]
    uns_key = f"{cat_col}_colors"
    if uns_key in adata.uns:
        del adata.uns[uns_key]
