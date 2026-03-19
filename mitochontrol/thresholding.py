"""Thresholding functions for mitochondrial RNA fraction QC."""
import logging
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

from anndata import AnnData

from . import core


logger = logging.getLogger(__name__)


def _create_model_pdfs(
    model: Mapping[str, Any],
) -> Tuple[List[Callable[[np.ndarray], np.ndarray]], np.ndarray]:
    """Create PDF functions from mixture model components."""
    model_pdfs = []

    for mean, var in model["components"]:
        std = np.sqrt(var)
        model_pdfs.append(lambda x, m=mean, s=std: norm.pdf(x, m, s))
    return model_pdfs, model["weights"]


def _get_threshold_column_name(
        threshold: float,
        threshold_labels: Optional[Mapping[float, str]] = None
) -> str:
    """Generate threshold column name from value and optional label."""
    if threshold_labels and threshold in threshold_labels:
        return f"Threshold out {threshold_labels[threshold]}"
    return f"Threshold out {threshold:g}"


def _create_evaluation_grid(
        x_values: np.ndarray,
        thresholds: Optional[Union[float, Sequence[float]]] = None,
        n_points: int = 1000
) -> np.ndarray:
    """Create evaluation grid from x_values, optionally extending for
    thresholds."""
    x_min = float(np.min(x_values))
    x_max = float(np.max(x_values))
    if thresholds is not None:
        if isinstance(thresholds, (list, np.ndarray, Sequence)):
            max_threshold = float(np.max(thresholds))
        else:
            max_threshold = float(thresholds)
        if max_threshold >= x_max:
            x_max = max_threshold * 1.25
    return np.linspace(x_min, x_max, n_points)


def _get_threshold_column_name_from_label(
        adata: AnnData,
        threshold_label: str
) -> str:
    """Get threshold column name, trying direct name then prefixed format.

    Returns the column name if found, otherwise raises KeyError.
    """
    if threshold_label in adata.obs.columns:
        return threshold_label
    prefixed = f"Threshold out {threshold_label}"
    if prefixed in adata.obs.columns:
        return prefixed
    raise KeyError(
        f"Expected column '{threshold_label}' or '{prefixed}' in "
        f"adata.obs. Available columns: {list(adata.obs.columns)}"
    )


def _compute_separation_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str = "euclidean",
) -> Dict[str, float]:
    """Compute separation metrics between two groups.

    Calculates three clustering quality metrics to assess how well
    two groups of cells are separated in feature space.

    Args:
        X: Feature matrix (e.g., PCA embedding) of shape
            ``(n_cells, n_features)``. May be dense or sparse.
        labels: Binary group membership array of shape
            ``(n_cells,)``. ``0`` = retained, ``1`` = removed.
        metric: Distance metric for silhouette score.
            Defaults to ``"euclidean"``.

    Returns:
        Dictionary with float values for ``"silhouette"``
        (higher is better, range [-1, 1]),
        ``"calinski_harabasz"`` (higher is better), and
        ``"davies_bouldin"`` (lower is better).  All values
        are ``NaN`` when a group has fewer than 2 cells.

    Notes:
        Metrics are computed independently; a failure in one does
        not affect the others.
    """

    # Initialize metrics dictionary with NaN (will be updated if computation
    # succeeds)
    metrics = {
        "silhouette": np.nan,
        "calinski_harabasz": np.nan,
        "davies_bouldin": np.nan,
    }

    # Validate and convert inputs
    labels = np.asarray(labels)
    if X.shape[0] != len(labels):
        raise ValueError(
            f"Shape mismatch: X has {X.shape[0]} rows but labels has "
            f"{len(labels)} elements."
        )

    # Count cells in each group
    n_removed = int(labels.sum())
    n_retained = len(labels) - n_removed

    # Require at least 2 cells per group for meaningful metrics
    if n_removed < 2 or n_retained < 2:
        logger.info(
            "Separation metrics skipped: one group has fewer than 2 cells "
            "(removed=%d, retained=%d).", n_removed, n_retained
        )
        return metrics

    # Silhouette score: measures how similar cells are to their own group
    # vs. the other group (range [-1, 1], higher is better)
    try:
        metrics["silhouette"] = silhouette_score(X, labels, metric=metric)
    except (ValueError, RuntimeError) as e:
        logger.warning("Silhouette score failed: %s", e)

    # Calinski-Harabasz index: ratio of between-cluster to within-cluster
    # variance (higher is better)
    try:
        metrics["calinski_harabasz"] = calinski_harabasz_score(X, labels)
    except (ValueError, RuntimeError) as e:
        logger.warning("Calinski–Harabasz score failed: %s", e)

    # Davies-Bouldin index: average similarity ratio of each cluster to its
    # most similar cluster (lower is better)
    try:
        metrics["davies_bouldin"] = davies_bouldin_score(X, labels)
    except (ValueError, RuntimeError) as e:
        logger.warning("Davies–Bouldin score failed: %s", e)

    return metrics


def naive_bayes_threshold(
    adata_cluster: AnnData,
    x_values: np.ndarray,  # 1D array of QC metric for all cells
    model: Mapping[str, Any],
    outdir: Optional[Union[str, Path]] = None,
    sample_id: Optional[str] = None,
    cluster_id: Optional[str] = None,
    compromised_component: Optional[Union[int, Sequence[float]]] = None,
    # index/indices of 'bad' comp(s)
    threshold_prob: float = 0.75,     # desired posterior probability cutoff
    grid: Optional[np.ndarray] = None,    # optional: x-axis to evaluate
    visualize: bool = True,
    suptitle: Optional[str] = None,
    celltype_color: str = "lightgrey",
    save: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    threshold_label: Optional[str] = None,  # label for column name
    column_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Apply posterior-probability cutoff from mixture model to flag cells.

    Uses a fitted mixture model to compute posterior probabilities for each
    component and flags cells that fall in regions where the compromised
    component(s) have posterior probability above a threshold. Thresholds are
    identified as contiguous regions in the x-value space.

    Args:
        adata_cluster: Cluster-specific AnnData object (cells × genes).
            Modified in place by adding threshold columns to ``obs``.
        x_values: 1D array of QC metric values (typically mtRNA fraction) for
            all cells, aligned to ``adata_cluster.obs_names``.
        model: Fitted mixture model dictionary with keys "components" (list
            of (mean, variance) tuples) and "weights" (1D array).
        outdir: Output directory for saving figures and results if
            ``save=True``. Required if ``save=True`` and ``save_path`` is None.
            Defaults to ``None``.
        sample_id: Sample identifier for filenames. Required if
            ``save=True`` and ``save_path`` is None. Defaults to ``None``.
        cluster_id: Cluster identifier for filenames. Required if
            ``save=True`` and ``save_path`` is None. Defaults to ``None``.
        compromised_component: Index or sequence of indices designating
            "compromised" (bad) component(s). If None, automatically selected
            as up to 2 components with highest posterior probability at the
            highest mtRNA levels (rightmost extending components), accounting
            for bimodal components. Defaults to ``None``.
        threshold_prob: Posterior probability cutoff for flagging cells.
            Cells in regions where compromised component posterior >= this
            value are flagged. Defaults to ``0.75``.
        grid: Optional x-axis grid for evaluating posterior curves. If None,
            created from min/max of x_values with 1000 points. Defaults to
            ``None``.
        visualize: Whether to create visualization plots. Defaults to
            ``True``.
        suptitle: Figure super-title override. Defaults to ``None``.
        celltype_color: Color for cell-type histogram overlay. Defaults to
            ``"lightgrey"``.
        save: Save thresholded labels, plots, and metrics to CSV/PNG.
            Defaults to ``False``.
        save_path: Optional explicit file path for saving plot. If provided,
            overrides ``outdir``-based path. Defaults to ``None``.
        show: Display plots interactively if True; otherwise close
            immediately. Defaults to ``True``.
        threshold_label: Label for threshold column name. If None,
            auto-generated from ``threshold_prob`` as
            "MitoChontrol: {int(threshold_prob*100)}% Posterior Prob".
            Defaults to ``None``.
        column_name: Exact column name to write into ``adata_cluster.obs``.
            If provided, overrides ``threshold_label``-based naming. Defaults
            to ``None``.

    Returns:
        Dictionary with keys:
            - "threshold": Primary threshold value (float).
            - "posterior_grid": Posterior probabilities for compromised
              component(s) evaluated on grid (1D array).
            - "component_posteriors": Posterior probabilities for all
              components on grid (2D array, shape [len(grid), n_components]).
            - "grid": X-axis grid values (1D array).
            - "Thresholded Cells": Number of flagged cells (int).
            - "silhouette", "calinski_harabasz", "davies_bouldin": Separation
              metrics (float, may be NaN).

    Notes:
        - Creates column ``"Threshold out {label}"`` in ``adata_cluster.obs``
          where {label} is determined by ``threshold_label`` parameter or
          auto-generated from ``threshold_prob``. See module-level
          documentation for column naming conventions.
        - Boolean values: True = flagged/removed, False = retained.
        - If no regions exceed threshold, defaults to threshold of 10%
          (fallback behavior).
        - Separation metrics computed only if PCA embedding available in
          ``adata_cluster.obsm["X_pca"]``.
    """

    # Determine save paths
    plot_path = None
    csv_path = None
    if save:
        if save_path is not None:
            plot_path = Path(save_path)
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            csv_path = plot_path.parent / (plot_path.stem + ".csv")
        elif (outdir is not None and sample_id is not None and
              cluster_id is not None):
            outdir_base = Path(outdir) / "mitochontrol" / "naive_bayes"
            outdir_base.mkdir(parents=True, exist_ok=True)
            plot_path = (
                outdir_base / f"{sample_id}_{cluster_id}_{threshold_prob}.png"
            )
            # Note: CSV uses .2f formatting while PNG uses raw threshold_prob
            # to preserve precision in filenames. This is intentional for
            # backward compatibility with existing scripts that parse
            # filenames.
            csv_path = (
                outdir_base /
                f"{sample_id}_{cluster_id}_{threshold_prob:.2f}.csv"
            )

    # Create PDF functions for each component (use default args to avoid
    # closure issues)
    model_pdfs, model_weights = _create_model_pdfs(model)

    # Create evaluation grid if not provided
    if grid is None:
        grid = _create_evaluation_grid(x_values)

    # Compute unnormalized posterior probabilities (likelihood * prior)
    posteriors = np.zeros((len(grid), len(model_pdfs)))
    for i, (pdf, weight) in enumerate(zip(model_pdfs, model_weights)):
        posteriors[:, i] = pdf(grid) * weight
    # Normalize to get true posterior probabilities (sum to 1 per grid point)
    posteriors /= np.sum(posteriors, axis=1, keepdims=True)

    # Auto-select compromised component(s) if not provided: choose up to 2
    # components with highest posterior probability at highest mtRNA levels
    # (rightmost extending components), accounting for bimodal components
    if compromised_component is None:
        # Evaluate posteriors at highest and lowest mtRNA levels
        # Use top 2% of actual cells for high mtRNA region
        # Find 98th percentile of actual cell mtRNA values
        cell_98th_percentile = np.percentile(x_values, 98)
        # Find grid points at or above this percentile
        top_indices = np.where(grid >= cell_98th_percentile)[0]
        # Ensure at least one grid point is selected
        if len(top_indices) == 0:
            top_indices = [np.argmin(np.abs(grid - cell_98th_percentile))]

        # For low mtRNA region, use bottom 10% of grid points
        n_bottom = max(1, int(len(grid) * 0.1))
        sorted_indices = np.argsort(grid)
        bottom_indices = sorted_indices[:n_bottom]

        # Calculate average posterior probabilities for each component
        avg_posterior_high = np.mean(posteriors[top_indices, :], axis=0)
        avg_posterior_low = np.mean(posteriors[bottom_indices, :], axis=0)

        # Thresholds for identifying compromised and bimodal components
        high_threshold = 0.2  # Minimum avg posterior in high mtRNA region
        low_threshold = 0.2  # Minimum avg posterior in low mtRNA region

        # Identify bimodal components (high posterior at both extremes)
        bimodal_components = []
        high_only_components = []
        for i in range(len(model_pdfs)):
            if avg_posterior_high[i] > high_threshold:
                if avg_posterior_low[i] > low_threshold:
                    # Bimodal component
                    bimodal_components.append(i)
                else:
                    # High-only component
                    high_only_components.append(i)

        # Selection logic (maximum 2 components)
        selected_components = []
        if bimodal_components:
            # Case A: Bimodal component(s) found
            # Include first bimodal component (if multiple, take the one with
            # highest high posterior)
            if len(bimodal_components) > 1:
                bimodal_scores = [
                    avg_posterior_high[i] for i in bimodal_components
                ]
                best_bimodal_idx = bimodal_components[
                    np.argmax(bimodal_scores)
                ]
                selected_components.append(best_bimodal_idx)
            else:
                selected_components.append(bimodal_components[0])

            # Include best high-only component if available
            if high_only_components:
                high_only_scores = [
                    avg_posterior_high[i] for i in high_only_components
                ]
                best_high_only_idx = high_only_components[
                    np.argmax(high_only_scores)
                ]
                selected_components.append(best_high_only_idx)
        else:
            # Case B: No bimodal component - select top 1-2 high-only
            # components
            if high_only_components:
                # Sort by avg_posterior_high (descending)
                sorted_high_only = sorted(
                    high_only_components,
                    key=lambda i: avg_posterior_high[i],
                    reverse=True
                )
                # Select up to 2
                selected_components = sorted_high_only[:2]
            else:
                # Case C: Fallback - select component with highest high
                # posterior
                selected_components = [int(np.argmax(avg_posterior_high))]

        # Ensure maximum 2 components
        if len(selected_components) > 2:
            # Keep top 2 by avg_posterior_high
            scores = [
                avg_posterior_high[i] for i in selected_components
            ]
            top_two_indices = np.argsort(scores)[-2:][::-1]
            selected_components = [
                selected_components[i] for i in top_two_indices
            ]

        # Set compromised_component (single int if one, list if multiple)
        if len(selected_components) == 1:
            compromised_component = selected_components[0]
        else:
            compromised_component = selected_components

    # Extract posterior for compromised component(s)
    # Handle both single component and multiple components
    if isinstance(compromised_component, (list, np.ndarray, Sequence)):
        # Multiple compromised components - sum their posteriors
        compromised_indices = np.asarray(compromised_component)
        compromised_post = np.sum(
            posteriors[:, compromised_indices], axis=1
        )
    else:
        # Single compromised component
        compromised_post = posteriors[:, compromised_component]

    # Identify contiguous regions where posterior >= threshold_prob
    mask = compromised_post >= threshold_prob
    if not np.any(mask):
        logger.info(
            "No grid values exceed the desired posterior threshold."
        )

    # Find start and end indices of contiguous high-probability regions
    # Group consecutive indices together
    high_prob_indices = np.where(mask)[0]
    regions = []
    if len(high_prob_indices) > 0:
        for k, g in groupby(
            enumerate(high_prob_indices), lambda ix: ix[0] - ix[1]
        ):
            group = list(map(itemgetter(1), g))
            start_idx, end_idx = group[0], group[-1]
            # Convert indices to x-values
            regions.append((grid[start_idx], grid[end_idx]))

    # Fallback: if no regions found, use default threshold
    if not regions:
        logger.info(
            "Insufficient data for appropriate threshold selection. "
            "Defaulting to 10%%"
        )
        # Fallback threshold: manually select resolution based on histogram
        regions.append((10.0, np.max(x_values)))

    # Select only the rightmost region (highest mtRNA) for thresholding
    # This is the region that extends furthest to the right
    if len(regions) > 1:
        # Find the region with the highest right boundary (hi value)
        rightmost_region = max(regions, key=lambda x: x[1])
        logger.info(
            f"Multiple regions found with posterior >= {threshold_prob}. "
            f"Using only the rightmost region: [{rightmost_region[0]:.2f}, "
            f"{rightmost_region[1]:.2f}]% mtRNA"
        )
    else:
        rightmost_region = regions[0]

    # Flag cells that fall into the rightmost threshold region only
    lo, hi = rightmost_region
    threshold_mask = (x_values >= lo) & (x_values <= hi)

    # Store thresholding results in AnnData
    if column_name is None:
        if threshold_label is None:
            threshold_label = (
                f"MitoChontrol: {int(threshold_prob*100)}% Posterior Prob"
            )
        column_name = f"Threshold out {threshold_label}"
    adata_cluster.obs[column_name] = threshold_mask.astype(bool)

    n_flagged = np.sum(threshold_mask)
    n_total = len(x_values)

    # Compute separation metrics if PCA embedding available
    metrics = {
        "silhouette": np.nan,
        "calinski_harabasz": np.nan,
        "davies_bouldin": np.nan,
    }
    labels = threshold_mask.astype(int)

    X = core._get_pca_embedding(adata_cluster)
    if X is not None:
        metrics = _compute_separation_metrics(X, labels, metric="euclidean")

    # Optional plot
    if visualize:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 15), sharex=True)

        gaussian_colors = [
            "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c"
        ]

        # --- Subplot 1: Posterior Probabilities with Threshold Regions ---
        ax1.plot(
            grid, compromised_post, color='red', linestyle='--',
            linewidth=3.5, label='Compromised Component'
        )
        for i in range(len(model_pdfs)):
            color = gaussian_colors[i % len(gaussian_colors)]
            ax1.plot(
                grid, posteriors[:, i], color=color, linewidth=2.5,
                label=f"Component {i+1} Posterior"
            )

        # Show all regions in light red for reference, highlight rightmost
        for start, end in regions:
            ax1.axvspan(start, end, color='pink', alpha=0.2)
        # Highlight the rightmost region that is actually used for thresholding
        if regions:
            ax1.axvspan(rightmost_region[0], rightmost_region[1],
                        color='red', alpha=0.3, label='Threshold region')

        ax1.set_ylabel("Posterior Probability", fontsize=12)
        # Use rightmost (primary) threshold region for title
        if regions:
            lower, upper = rightmost_region
            ax1.set_title(
                f"Naive Bayes Posterior Probability\n"
                f"Posterior Probability of Compromised Component Cutoff at "
                f"{threshold_prob:.2f}\n Threshold: {lower:.2f}% mtRNA"
            )
        else:
            ax1.set_title(
                f"Naive Bayes Posterior Probability\n"
                f"Posterior Probability of Compromised Component Cutoff at "
                f"{threshold_prob:.2f}"
            )
        ax1.legend()

        # --- Subplot 2: Histogram + Mixture Model + Threshold Regions ---
        sns.histplot(
            x_values, bins=50, stat='density', color=celltype_color,
            edgecolor='grey', alpha=0.5, label='Distribution of mtRNA',
            ax=ax2
        )

        # Show only the rightmost region that is actually used for thresholding
        if regions:
            ax2.axvspan(rightmost_region[0], rightmost_region[1],
                        color='red', alpha=0.3, label='Threshold region')

        for i, pdf in enumerate(model_pdfs):
            comp_pdf = pdf(grid) * model_weights[i]
            color = gaussian_colors[i % len(gaussian_colors)]
            ax2.plot(
                grid, comp_pdf, color=color, linewidth=2.5,
                label=f'Component {i + 1}'
            )
        ax2.set_xlabel("Mitochondrial RNA Fraction (%)", fontsize=12)
        ax2.set_ylabel("Density", fontsize=12)
        percent_flagged = (
            100.0 * n_flagged / n_total if n_total > 0 else 0.0
        )
        ax2.set_title(
            f"Thresholded Cells: {n_flagged} / {n_total} "
            f"({percent_flagged:.1f}%)"
        )

        handles, labels = ax2.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax2.legend(unique.values(), unique.keys())

        # Set x-axis limits to full mtRNA range (0-100%)
        ax2.set_xlim(0, min(100, x_values.max()))

        # --- Subplot 3: Combined Histogram, GMM, Posterior, and Threshold ---
        # Create dual y-axes for density and posterior probability
        ax3_density = ax3
        ax3_posterior = ax3.twinx()
        # ax3_posterior.set_zorder(ax3_density.get_zorder() - 1)
        # ax3_posterior.patch.set_visible(False)

        # Histogram with fill color matching first component
        first_component_color = gaussian_colors[0]  # "#9ecae1"
        sns.histplot(
            x_values, bins=50, stat='density', color=first_component_color,
            edgecolor='grey', alpha=0.5, ax=ax3_density
        )

        # GMM KDEs in uniform blue (matching component 3 color from plots 1-2)
        gmm_blue = "#4292c6"  # Component 3 color
        gmm_label_added = False
        compromised_label_added = False
        # Store component PDFs and grids for annotation
        component_data = []

        for i, pdf in enumerate(model_pdfs):
            comp_pdf = pdf(grid) * model_weights[i]
            # Store component data for finding nearest points
            component_data.append((grid, comp_pdf))

            # Check if this is a compromised component
            if isinstance(compromised_component, (list, np.ndarray, Sequence)):
                is_compromised = i in compromised_component
            else:
                is_compromised = i == compromised_component

            if is_compromised:
                # Compromised components: dashed blue line
                shading_label = (
                    'Compromised Component' if not compromised_label_added
                    else ''
                )
                if not compromised_label_added:
                    compromised_label_added = True

                # Blue dashed line for compromised component
                ax3_density.plot(
                    grid, comp_pdf, color=gmm_blue, linewidth=3,
                    linestyle='--', label=shading_label, zorder=4
                )
            else:
                # Non-compromised components: solid blue line
                if not gmm_label_added:
                    ax3_density.plot(
                        grid, comp_pdf, color=gmm_blue, linewidth=2.5,
                        label='Gaussian Mixture Model'
                    )
                    gmm_label_added = True
                else:
                    ax3_density.plot(
                        grid, comp_pdf, color=gmm_blue, linewidth=2.5
                    )

        # Compromised posterior probability (on right y-axis)
        # Don't add to legend (will be handled by axis label)
        ax3_posterior.plot(
            grid, compromised_post, color='red', linewidth=2.5, zorder=1
        )

        # Threshold as thin dotted red vertical line
        threshold_value = rightmost_region[0]
        ax3_density.axvline(
            threshold_value, color='red', linestyle=':', linewidth=1.5,
            alpha=0.7
        )

        # Thresholded region shaded in light red (twice as light: 0.2 -> 0.1)
        ax3_density.axvspan(
            rightmost_region[0], rightmost_region[1],
            color='red', alpha=0.1, label='Thresholded Region'
        )

        # Find intersection point: where threshold vertical line intersects
        # posterior probability curve
        intersection_idx = np.argmin(np.abs(grid - threshold_value))
        intersection_mtrna = grid[intersection_idx]
        intersection_posterior = compromised_post[intersection_idx]

        # Red point at intersection
        ax3_posterior.scatter(
            [intersection_mtrna], [intersection_posterior],
            color='red', s=100, zorder=5
        )

        # Annotation for red dot (no arrow, text without character limit)
        # Format: {threshold probability}% Probability of Compromise at
        # {threshold}% mtRNA. {removed cells}/{total cells} Cells above
        # threshold.
        annotation_text = (
            f"{int(threshold_prob*100)}% Probability of \n "
            f"Compromise at {threshold_value:.2f}% mtRNA. \n"
            f"{n_flagged}/{n_total} Cells above threshold."
        )
        # Use annotation text directly without character limit
        wrapped_text = annotation_text

        buffer_pts = 10  # same buffer for x/y (points)
        buffer_px = buffer_pts * ax3_posterior.figure.dpi / 72.0

        # 1) Place annotation "below" the dot, left-aligned to dot (candidate)
        ann = ax3_posterior.annotate(
            wrapped_text,
            xy=(intersection_mtrna, intersection_posterior),
            xycoords="data",
            xytext=(buffer_pts, -buffer_pts),      # right and down in points
            textcoords="offset points",
            ha="left",
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.5", facecolor="white", alpha=1.0,
                edgecolor="red", linewidth=1.5
            ),
            fontsize=9,
            clip_on=True,
            zorder=10,
        )

        # 2) Draw once so the annotation has a real size
        fig = ax3_posterior.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        # 3) Measure annotation bbox in display coords
        bbox = ann.get_window_extent(renderer=renderer)
        ax_bbox = ax3_posterior.get_window_extent(renderer=renderer)
        # If the annotation overflows right-align it and move left
        if bbox.x1 > (ax_bbox.x1 - buffer_px):
            ann.set_ha("right")
            # left and down from the dot move annotation left
            ann.set_position((-buffer_pts, -buffer_pts))

            # Optional: re-draw if you want to be exact after the change
            fig.canvas.draw()

        # Set labels and titles (axis labels size 12)
        ax3_density.set_xlabel("Mitochondrial RNA Fraction (%)", fontsize=12)
        ax3_density.set_ylabel("Density", color=gmm_blue, fontsize=12)
        ax3_posterior.set_ylabel(
            "Posterior Probability of Compromise", color='red', fontsize=12
        )
        ax3_density.tick_params(axis='y', labelcolor='black')
        ax3_posterior.tick_params(axis='y', labelcolor='black')

        # Set x-axis limits to full mtRNA range (0-100%)
        ax3_density.set_xlim(0, min(100, x_values.max()))

        # Add annotation just to the right of threshold with arrows to each
        # component
        if component_data:
            # Position annotation just to the right of threshold
            x_range = x_values.max() - x_values.min()
            x_annotation = threshold_value + 0.05 * x_range

            # Find maximum y value of all GMM components at annotation x
            # position
            max_y_at_x = 0.0
            for comp_grid, comp_pdf in component_data:
                # Find grid point nearest to annotation x position
                nearest_idx = np.argmin(np.abs(comp_grid - x_annotation))
                y_at_x = comp_pdf[nearest_idx]
                max_y_at_x = max(max_y_at_x, y_at_x)

            # Position annotation above the maximum GMM value with buffer
            # Add ~10% of y-axis range as vertical buffer
            y_range = ax3_density.get_ylim()[1] - ax3_density.get_ylim()[0]
            buffer = 0.2 * y_range
            y_annotation = max_y_at_x + buffer

            # Check which components are compromised for special labeling
            if isinstance(compromised_component, (list, np.ndarray, Sequence)):
                compromised_indices = list(compromised_component)
            else:
                compromised_indices = (
                    [compromised_component]
                    if compromised_component is not None
                    else []
                )

            # Collect all component endpoints and sort so compromised are last
            component_endpoints = []
            for i, (comp_grid, comp_pdf) in enumerate(component_data):
                # Find grid point nearest to annotation x position
                nearest_idx = np.argmin(np.abs(comp_grid - x_annotation))
                nearest_x = comp_grid[nearest_idx]
                nearest_y = comp_pdf[nearest_idx]
                is_compromised = i in compromised_indices
                component_endpoints.append(
                    (i, nearest_x, nearest_y, is_compromised, comp_grid,
                     comp_pdf)
                )

            # Sort so compromised components are drawn last (rightmost)
            component_endpoints.sort(key=lambda x: (x[3], x[1]))

            # Add text annotation first (only once)
            text_obj = ax3_density.text(
                x_annotation, y_annotation, 'Gaussian Mixture Model',
                fontsize=12,
                ha='left',
                va='center',
                color=gmm_blue,
                weight='bold',
                zorder=10
            )
            # Get text width to center arrows on text
            fig = ax3_density.figure
            fig.canvas.draw()
            text_bbox = text_obj.get_window_extent(
                renderer=fig.canvas.get_renderer()
            )
            text_bbox_data = text_bbox.transformed(
                ax3_density.transData.inverted()
            )

            text_width = text_bbox_data.x1 - text_bbox_data.x0
            text_center = text_bbox_data.x0 + text_width / 2

            # Center point of text for arrow origin x
            arrow_origin_x = text_center - text_width / 4
            arrow_origin_y = text_bbox_data.y0 - 0.02 * y_range
            # Position compromised component text below GMM text
            comp_text_x = text_center + text_width / 4
            comp_text_y = text_bbox_data.y0 - 0.05 * y_range

            # Position arrow origin below compromised component text
            # (we'll calculate this after creating compromised text)

            # Check if any components are compromised
            has_compromised = any(ep[3] for ep in component_endpoints)

            # Add compromised component text below GMM text if exists
            compromised_text_bbox = None
            if has_compromised:
                from matplotlib.patches import FancyBboxPatch
                compromised_text_obj = ax3_density.text(
                    comp_text_x, comp_text_y,
                    'Compromised \n Component',
                    fontsize=10,
                    ha='center',
                    va='center',
                    color=gmm_blue,
                    zorder=11
                )
                # Get compromised text bounding box
                fig = ax3_density.figure
                fig.canvas.draw()
                compromised_bbox = compromised_text_obj.get_window_extent(
                    renderer=fig.canvas.get_renderer()
                )
                compromised_text_bbox = compromised_bbox.transformed(
                    ax3_density.transData.inverted()
                )
                # Create dashed border patch with minimal padding
                border_patch = FancyBboxPatch(
                    (compromised_text_bbox.x0, compromised_text_bbox.y0),
                    compromised_text_bbox.width,
                    compromised_text_bbox.height,
                    boxstyle='round,pad=0.01',
                    facecolor='white',
                    edgecolor=gmm_blue,
                    linewidth=1.5,
                    linestyle='--',
                    alpha=0.85,
                    zorder=10,
                    transform=ax3_density.transData
                )
                ax3_density.add_patch(border_patch)

                comp_width = compromised_text_bbox.x1-compromised_text_bbox.x0
                comp_text_center = compromised_text_bbox.x0 + comp_width / 2
                # Position arrow origin below compromised component text
                comp_arrow_origin_x = comp_text_center - comp_width / 4
                comp_arrow_origin_y = compromised_text_bbox.y0 - 0.02 * y_range

            # Draw all arrows from the same origin point (center of text)
            for (i, nearest_x, nearest_y, is_compromised, comp_grid,
                 comp_pdf) in component_endpoints:
                # Draw arrow from annotation center to component point
                if is_compromised:
                    # get nearest point of compromised component
                    nearest_idx = np.argmin(np.abs(comp_grid-text_center))
                    nearest_x = comp_grid[nearest_idx]
                    nearest_y = comp_pdf[nearest_idx]
                    ax3_density.annotate(
                        '',
                        xy=(text_center, nearest_y),
                        xytext=(comp_arrow_origin_x, comp_arrow_origin_y),
                        arrowprops=dict(arrowstyle='->', color=gmm_blue,
                                        lw=2.5),
                        zorder=10
                    )
                else:
                    ax3_density.annotate(
                        '',
                        xy=(nearest_x, nearest_y),
                        xytext=(arrow_origin_x, arrow_origin_y),
                        arrowprops=dict(arrowstyle='->', color=gmm_blue,
                                        lw=2.5),
                        zorder=10
                    )

        if suptitle:
            plt.suptitle(suptitle)
        plt.tight_layout()

        if plot_path is not None:
            plot_path = Path(plot_path)
            fig.savefig(plot_path, bbox_inches='tight', dpi=300)
            # Also save SVG
            svg_path = plot_path.with_suffix('.svg')
            fig.savefig(svg_path, bbox_inches='tight', format='svg')
        if show:
            plt.show()
        else:
            plt.close(fig)

    # Use left boundary of rightmost region as threshold
    threshold = rightmost_region[0]
    results = {
        "threshold": threshold,
        "posterior_grid": compromised_post,
        "component_posteriors": posteriors,
        "grid": grid,
        "Thresholded Cells": n_flagged,
        **metrics,
    }
    if csv_path is not None:
        pd.DataFrame(
            [(
                threshold_prob, threshold, n_flagged,
                100.0 * n_flagged / n_total if n_total > 0 else 0.0,
                metrics["silhouette"], metrics["calinski_harabasz"],
                metrics["davies_bouldin"]
            )],
            columns=[
                "posterior_prob_cutoff", "threshold", "n_lost", "percent_lost",
                "silhouette", "calinski_harabasz", "davies_bouldin"
            ],
        ).to_csv(csv_path, index=False)

    return results


def manual_threshold(
    adata_cluster: AnnData,
    x_values: np.ndarray,  # 1D array of QC metric for all cells
    model: Mapping[str, Any],  # used only to draw component PDFs
    thresholds: Union[float, Sequence[float]],  # manual threshold(s)
    outdir: Optional[Union[str, Path]] = None,
    sample_id: Optional[str] = None,
    cluster_id: Optional[str] = None,
    suptitle: Optional[str] = None,
    threshold_labels: Optional[Mapping[float, str]] = None,
    grid: Optional[np.ndarray] = None,    # optional: x-axis to evaluate PDFs
    visualize: bool = True,
    celltype_color: str = "lightgrey",
    show_MM: bool = False,
    save: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> None:

    """Apply user-specified threshold(s) on QC metric and flag cells.

    Applies one or more manual thresholds to flag cells based on a QC metric
    (e.g., mtRNA fraction). For each threshold, cells with values >= threshold
    are flagged and stored as boolean columns in the AnnData object. Optionally
    visualizes the thresholds overlaid on the data distribution and mixture
    model components.

    Args:
        adata_cluster: Cluster-specific AnnData object (cells × genes).
            Modified in place by adding boolean columns to ``obs`` for each
            threshold.
        x_values: 1D array of QC metric values (typically mtRNA fraction) for
            all cells, aligned to ``adata_cluster.obs_names``.
        model: Fitted mixture model dictionary with keys "components" (list
            of (mean, variance) tuples) and "weights" (1D array). Used only
            for visualization overlays if ``show_MM=True``.
        thresholds: Single threshold value or sequence of threshold values in
            the same units as ``x_values``. Duplicate values are automatically
            removed and values are sorted.
        outdir: Output directory for saving figures and CSV results if
            ``save=True``. Required if ``save=True`` and ``save_path`` is None.
            Defaults to ``None``.
        sample_id: Sample identifier for filenames. Required if
            ``save=True`` and ``save_path`` is None. Defaults to ``None``.
        cluster_id: Cluster identifier for filenames. Required if
            ``save=True`` and ``save_path`` is None. Defaults to ``None``.
        suptitle: Figure title override. If provided, prepended to default
            title. Defaults to ``None``.
        threshold_labels: Optional mapping from threshold values to custom
            labels for display and column names. See module-level
            documentation for column naming conventions. Defaults to ``None``.
        grid: Optional x-axis grid for evaluating PDFs. If None, created from
            min/max of x_values and thresholds. Defaults to ``None``.
        visualize: Whether to create visualization plots. Defaults to ``True``.
        celltype_color: Color for cell-type histogram overlay. Defaults to
            ``"lightgrey"``.
        show_MM: Whether to overlay mixture model component PDFs on the
            histogram. Defaults to ``False``.
        save: Save thresholded labels, plots, and metrics to CSV/PNG.
            Defaults to ``False``.
        save_path: Optional explicit file path for saving plot. If provided,
            overrides ``outdir``-based path. CSV will be saved to same
            directory with .csv extension. Defaults to ``None``.
        show: Display plots interactively if True; otherwise close
            immediately. Defaults to ``True``.

    Returns:
        None: Function modifies ``adata_cluster.obs`` in place and optionally
        saves files.

    Notes:
        - For each threshold, adds a column ``"Threshold out {label}"`` to
          ``adata_cluster.obs`` with boolean values (True = flagged). See
          module-level documentation for column naming conventions.
        - Separation metrics (silhouette, Calinski-Harabasz, Davies-Bouldin)
          are computed only if PCA embedding is available in
          ``adata_cluster.obsm["X_pca"]``.
        - Figure height automatically adjusts based on number of thresholds.
    """

    # Determine save paths
    plot_path = None
    csv_path = None
    if save:
        if save_path is not None:
            plot_path = Path(save_path)
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            # CSV path derived from plot path
            csv_path = plot_path.parent / (plot_path.stem + ".csv")
        elif (outdir is not None and sample_id is not None and
              cluster_id is not None):
            outdir_base = Path(outdir) / "mitochontrol" / "thresholds"
            outdir_base.mkdir(parents=True, exist_ok=True)
            plot_path = outdir_base / f"{sample_id}_{cluster_id}.png"
            csv_path = outdir_base / f"{sample_id}_{cluster_id}.csv"

    # Create PDF functions for each component (use default args to avoid
    # closure issues)
    model_pdfs, model_weights = _create_model_pdfs(model)

    # Create evaluation grid if not provided (extends for thresholds)
    if grid is None:
        grid = _create_evaluation_grid(x_values, thresholds)

    # Normalize thresholds to sorted list of unique floats
    if np.isscalar(thresholds):
        thresholds = [float(thresholds)]
    thresholds = sorted({float(t) for t in thresholds})

    # Process each threshold: flag cells and compute metrics
    n_total = len(x_values)
    if n_total == 0:
        raise ValueError("x_values must be non-empty.")
    rows = []  # For CSV output (includes metrics for each threshold)
    threshold_info = []  # For plot annotations: (threshold, n_flagged, pct)

    # Pull embedding once
    X_embed = core._get_pca_embedding(adata_cluster)

    for t in thresholds:
        # Flag cells at or above threshold
        threshold_mask = x_values >= t
        # Generate column name with optional label
        column_title = _get_threshold_column_name(t, threshold_labels)
        adata_cluster.obs[column_title] = threshold_mask

        n_flagged = int(threshold_mask.sum())
        pct_flagged = 100.0 * n_flagged / n_total if n_total > 0 else 0.0
        threshold_info.append((t, n_flagged, pct_flagged))

        # Compute separation metrics if PCA embedding available
        metrics = {
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan,
        }
        if X_embed is not None:
            labels = threshold_mask.astype(int)
            metrics = _compute_separation_metrics(
                X_embed, labels, metric="euclidean"
            )

        rows.append({
            "threshold label": column_title,
            "threshold": t,
            "n_lost": n_flagged,
            "percent_lost": pct_flagged,
            "silhouette": metrics["silhouette"],
            "calinski_harabasz": metrics["calinski_harabasz"],
            "davies_bouldin": metrics["davies_bouldin"],
        })

    if visualize:
        # Dynamic figure height: base height + growth per annotation
        # (grows once you have >2 thresholds)
        n_ann = len(threshold_info)
        fig_height = max(5.0, 5.0 + 0.55 * max(0, n_ann - 2))

        # Use constrained layout instead of tight_layout (prevents
        # label/legend overlap)
        fig, ax2 = plt.subplots(
            1, 1, figsize=(9, fig_height), constrained_layout=True
        )

        gaussian_colors = [
            "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c"
        ]

        # --- Histogram + Mixture Model + Threshold Region (ONLY subplot) ---
        sns.histplot(
            x_values, bins=50, stat='density', color=celltype_color,
            edgecolor='grey', alpha=0.5, label='Distribution of mtRNA',
            ax=ax2
        )

        if show_MM:
            # Overlay component PDFs from the model
            for i, pdf in enumerate(model_pdfs):
                comp_pdf = pdf(grid) * model_weights[i]
                color = gaussian_colors[i % len(gaussian_colors)]
                ax2.plot(
                    grid, comp_pdf, color=color, linewidth=2.0,
                    label=f'Component {i + 1}'
                )

        # Draw threshold lines, shaded regions, and annotations
        x_min_ax, x_max_ax = ax2.get_xlim()
        for idx, (t, n_flagged, pct) in enumerate(threshold_info):
            label_val = threshold_labels.get(t) if threshold_labels else None
            if label_val:
                header = f"{label_val}: {t:g}% mtRNA"
                fw = 'bold'
            else:
                header = f"{t:g}% mtRNA:"
                fw = None

            # Draw vertical line and shaded region for threshold
            ax2.axvline(t, color='red', linestyle='--', linewidth=0.9)
            ax2.axvspan(t, x_max_ax, color='red', alpha=0.1)

            # Place annotation near threshold line
            y = 0.98 - 0.05 * idx
            # Position annotation: near the line with small offset, but don't
            # exceed plot boundaries
            x_pos = min(
                t + 0.02 * (x_max_ax - x_min_ax),
                x_max_ax - 0.01 * (x_max_ax - x_min_ax)
            )
            ax2.text(
                x_pos,
                y,
                f"{header}\nCells Lost: {n_flagged}/{n_total} ({pct:.1f}%)",
                transform=ax2.get_xaxis_transform(),
                # x in data coords, y in axes fraction
                ha='left', va='top', fontsize=10, fontweight=fw
            )

        ax2.set_xlabel("Mitochondrial RNA Fraction (%)")
        ax2.set_ylabel("Density")
        if suptitle:
            ax2.set_title(f"{suptitle}Thresholded Cells")
        else:
            ax2.set_title("Thresholded Cells")

        # De-duplicate legend
        handles, labels = ax2.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax2.legend(uniq.values(), uniq.keys())

        # Set x-axis limits to full mtRNA range (0-100%)
        ax2.set_xlim(0, min(100, x_values.max()))

        if plot_path is not None:
            plot_path = Path(plot_path)
            fig.savefig(plot_path, bbox_inches='tight', dpi=300)
            # Also save SVG
            svg_path = plot_path.with_suffix('.svg')
            fig.savefig(svg_path, bbox_inches='tight', format='svg')
        if show:
            plt.show()
        else:
            plt.close(fig)

    # Save per-threshold results table to CSV
    if csv_path is not None:
        df = pd.DataFrame(rows, columns=[
            "threshold label", "threshold", "n_lost", "percent_lost",
            "silhouette", "calinski_harabasz", "davies_bouldin",
        ])
        df.to_csv(csv_path, index=False)
