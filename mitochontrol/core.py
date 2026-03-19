"""Core utilities for mitochondrial RNA fraction thresholding.

This module provides basic data manipulation, metadata transfer, and
data structure utilities.
"""

import csv
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from anndata import AnnData
from scipy import sparse

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

PathLike = Union[str, Path]

logger = logging.getLogger(__name__)


def _to_path(path: PathLike) -> Path:
    """Return ``path`` as a ``Path`` instance."""
    return Path(path)


def _ensure_dir(path: PathLike) -> Path:
    """Create ``path`` if needed and return it."""
    path = _to_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _copy_matrix(X: Any):
    """Return a copy of a dense or sparse matrix."""
    return X.copy() if sparse.issparse(X) else np.array(X, copy=True)


def _matrix_values(X, max_items: int = 10000) -> np.ndarray:
    """Sample numeric values from a matrix for lightweight heuristics."""
    if sparse.issparse(X):
        values = np.asarray(X.data)
    else:
        values = np.asarray(X).ravel()
    if values.size == 0:
        return values.astype(float)
    if values.size > max_items:
        idx = np.linspace(0, values.size - 1, max_items, dtype=int)
        values = values[idx]
    return values.astype(float, copy=False)


def _is_count_like(X: Any, tol: float = 1e-6) -> bool:
    """Heuristically detect whether a matrix looks like raw counts."""
    values = _matrix_values(X)
    if values.size == 0:
        return True
    if np.any(~np.isfinite(values)) or np.any(values < -tol):
        return False
    return bool(np.all(np.abs(values - np.round(values)) <= tol))


def _get_pca_embedding(adata: AnnData) -> Optional[np.ndarray]:
    """Get PCA embedding if available, otherwise log and return None."""
    if "X_pca" in adata.obsm:
        return adata.obsm["X_pca"]
    logger.info(
        "adata.obsm['X_pca'] not found; skipping expression-based "
        "separation metrics."
    )
    return None


def transfer_metadata(
    source_adata: AnnData,
    target_adata: AnnData,
    metadata_labels: Sequence[str],
) -> None:
    """Transfer selected `.obs` columns from one AnnData to another, in place.

    Aligns on `.obs_names`, preserves categorical dtypes (including order),
    and writes only the intersection of cells. Non-overlapping cells in
    `target_adata` are set to NA for transferred columns.

    Args:
        source_adata: Source AnnData object containing metadata to transfer.
        target_adata: Target AnnData object to receive metadata (modified
            in place).
        metadata_labels: Sequence of column names to transfer from
            `source_adata.obs`.

    Raises:
        ValueError: If none of the requested columns are present in
            `source_adata.obs`, or if there are no overlapping cells between
            the two AnnData objects.

    Notes:
        - Columns not present in `source_adata.obs` are silently ignored.
        - Categorical dtypes preserve their categories and ordering.
        - If dtype conversion fails, both source and target columns are
          converted to object dtype as a fallback.
        - New columns are initialized with NA values for all cells, then
          populated with source data for overlapping cells.
    """

    if len(metadata_labels) == 0:
        return

    present_columns = [
        m for m in metadata_labels if m in source_adata.obs.columns
    ]
    if not present_columns:
        raise ValueError(
            "None of the requested metadata columns are present in "
            f"source_adata.obs. Requested: {metadata_labels}. "
            f"Available: {list(source_adata.obs.columns)}"
        )

    common_cells = source_adata.obs_names.intersection(
        target_adata.obs_names
    )
    if len(common_cells) == 0:
        raise ValueError(
            "No overlapping cells between source_adata and target_adata."
        )

    # Sort for consistent ordering
    common_cells = common_cells.sort_values()

    for col in present_columns:
        src_data = source_adata.obs.loc[common_cells, col]

        if isinstance(src_data.dtype, pd.CategoricalDtype):
            # Preserve categorical structure (categories and ordering)
            cat_dtype = pd.CategoricalDtype(
                categories=src_data.cat.categories,
                ordered=src_data.cat.ordered
            )

            # Initialize column with NA values if missing or wrong dtype
            target_is_cat = isinstance(
                target_adata.obs.get(col, pd.Series()).dtype,
                pd.CategoricalDtype,
            )
            if col not in target_adata.obs.columns or not target_is_cat:
                target_adata.obs[col] = pd.Series(
                    pd.Categorical(
                        [pd.NA] * target_adata.n_obs, dtype=cat_dtype
                    ),
                    index=target_adata.obs.index
                )
            else:
                # Ensure target column matches source categorical structure
                target_adata.obs[col] = (
                    target_adata.obs[col].astype(cat_dtype)
                )

            # Transfer data for overlapping cells
            target_adata.obs.loc[common_cells, col] = (
                src_data.astype(cat_dtype).values
            )

        else:
            # Handle non-categorical (numeric, string, etc.) columns
            src_dtype = src_data.dtype

            if col not in target_adata.obs.columns:
                # Initialize new column with NA values
                target_adata.obs[col] = pd.Series(
                    pd.NA, index=target_adata.obs.index, dtype=src_dtype
                )
            else:
                # Attempt to match source dtype, fallback to object if needed
                if target_adata.obs[col].dtype != src_dtype:
                    try:
                        target_adata.obs[col] = (
                            target_adata.obs[col].astype(src_dtype)
                        )
                    except TypeError:
                        # Type conversion failed; use object dtype for both
                        target_adata.obs[col] = (
                            target_adata.obs[col].astype("object")
                        )
                        src_data = src_data.astype("object")

            # Transfer data for overlapping cells
            target_adata.obs.loc[common_cells, col] = src_data.to_numpy()


def compute_mt_fraction(adata: AnnData, mt_prefix: str = "MT-") -> None:
    """Compute the per-cell mitochondrial RNA fraction and store it in
    ``adata.obs["mt_frac"]``.

    Computes the percentage of total counts that come from mitochondrial
    genes (identified by prefix) for each cell. The result is stored as a
    float in the range [0.0, 100.0], where NaN values are replaced with 0.0.

    Args:
        adata: Annotated data matrix containing gene expression counts.
        mt_prefix: Prefix identifying mitochondrial genes (case-insensitive).
            Defaults to ``"MT-"``.

    Returns:
        None: Updates ``adata.obs`` in place with a new column ``"mt_frac"``.

    Notes:
        - Also modifies ``adata.var["mt"]`` to mark mitochondrial genes.
        - Uses Scanpy's ``calculate_qc_metrics`` to compute percentages.
        - If no genes match the prefix, all cells will have ``mt_frac = 0.0``.
        - NaN values (e.g., from cells with zero total counts) are set to 0.0.
    """
    # Identify mitochondrial genes by prefix (case-insensitive)
    adata.var["mt"] = (
        adata.var_names.str.upper().str.startswith(mt_prefix.upper())
    )

    # Check if any mitochondrial genes were found
    n_mt_genes = adata.var["mt"].sum()
    if n_mt_genes == 0:
        logger.warning(
            f"No genes found matching prefix '{mt_prefix}'. "
            f"All cells will have mt_frac = 0.0"
        )

    # Compute QC metrics using Scanpy (creates pct_counts_mt column)
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    # Convert to float and handle NaN values (set to 0.0 for cells with
    # zero total counts or other edge cases)
    adata.obs["mt_frac"] = (
        adata.obs["pct_counts_mt"].astype(float).fillna(0.0)
    )


def get_cluster_dict(
    adata: AnnData, cluster_label: str = "leiden"
) -> Dict[str, AnnData]:
    """Return a dictionary of per-cluster AnnData objects sorted by cluster
    label.

    Splits the input AnnData into separate AnnData objects, one per unique
    cluster value. Each returned AnnData is an independent copy containing only
    cells belonging to that cluster.

    Args:
        adata: Annotated data matrix containing cluster assignments.
        cluster_label: Column in ``adata.obs`` with cluster IDs.
            Defaults to ``"leiden"``.

    Returns:
        Dictionary mapping each cluster label (as string) to a corresponding
        AnnData subset. Keys are sorted lexicographically after converting to
        strings (handles mixed numeric/string cluster labels).

    Raises:
        KeyError: If ``cluster_label`` is not present in ``adata.obs.columns``.

    Notes:
        - Returns an empty dictionary if ``adata`` has no observations.
        - NaN values in ``cluster_label`` are treated as a distinct cluster.
        - Each returned AnnData is a copy; modifications won't affect the
          original.
    """
    if cluster_label not in adata.obs.columns:
        raise KeyError(
            f"Cluster label column '{cluster_label}' not found in "
            f"adata.obs. Available columns: {list(adata.obs.columns)}"
        )

    # Create dictionary of cluster subsets (each is an independent copy)
    cluster_dict = {
        cluster: adata[
            adata.obs[cluster_label] == cluster, :
        ].copy()
        for cluster in adata.obs[cluster_label].unique()
    }

    # Sort keys lexicographically after string conversion to handle mixed
    # types (e.g., numeric clusters like 0, 1, 2 and string clusters)
    sorted_keys = sorted(
        cluster_dict.keys(), key=lambda cluster_id: str(cluster_id)
    )
    return {cluster_id: cluster_dict[cluster_id] for cluster_id in sorted_keys}


def get_mt_dict(
    cluster_dict: Mapping[str, AnnData], mt_label: str = "mt_frac"
) -> Dict[str, pd.Series]:
    """Return a dictionary of per-cell mitochondrial RNA fractions for each
    cluster.

    Extracts the mitochondrial RNA fraction values from each cluster's AnnData
    object and returns them as pandas Series indexed by cell IDs.

    Args:
        cluster_dict: Dictionary mapping cluster labels to AnnData objects.
        mt_label: Column name in ``adata.obs`` containing mitochondrial
            fractions. Defaults to ``"mt_frac"``.

    Returns:
        Dictionary mapping each cluster label to a pandas Series of
        mitochondrial fractions. Each Series is indexed by the corresponding
        cell IDs (``adata.obs_names``).

    Raises:
        KeyError: If ``mt_label`` is not present in any AnnData's
            ``obs.columns``.

    Notes:
        - Returns an empty dictionary if ``cluster_dict`` is empty.
        - All AnnData objects in ``cluster_dict`` must have the ``mt_label``
          column; the function will raise KeyError on the first missing column.
    """
    if not cluster_dict:
        return {}

    # Extract mt_frac values for each cluster, preserving cell ID indices
    result = {}
    for cluster, adata in cluster_dict.items():
        if mt_label not in adata.obs.columns:
            raise KeyError(
                f"Column '{mt_label}' not found in cluster '{cluster}' "
                f"AnnData.obs. Available columns: {list(adata.obs.columns)}"
            )
        result[cluster] = pd.Series(
            adata.obs[mt_label].values, index=adata.obs_names
        )
    return result


def get_label_maps(outdir: Union[str, Path]) -> Dict[str, Dict[str, str]]:
    """Load a mapping of cluster ('leiden') labels to cell-type annotations
    from a CSV file.

    Reads a CSV file with columns "Sample", "Cluster", and "Label 1" and
    constructs a nested dictionary mapping samples to cluster-to-cell-type
    mappings.

    Args:
        outdir: Directory path containing ``cluster_results.csv``.

    Returns:
        Nested dictionary mapping
        ``{sample: {cluster_label: cell_type_label}}``. Returns an empty
        dictionary if the file is empty or all rows are skipped.

    Raises:
        FileNotFoundError: If ``cluster_results.csv`` does not exist in
            ``outdir``.
        PermissionError: If the file cannot be read due to permissions.

    Notes:
        - Expected CSV columns: "Sample", "Cluster", "Label 1".
        - Rows with missing or empty values in any required column are
          skipped.
        - If duplicate cluster labels exist for the same sample, the last
          occurrence in the CSV overwrites previous values.
        - Column names are case-sensitive and must match exactly.
    """
    label_maps: Dict[str, Dict[str, str]] = {}
    csv_path = Path(outdir) / "cluster_results.csv"

    # Read CSV file with expected columns: "Sample", "Cluster", "Label 1"
    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            sample = row.get("Sample")
            cluster = row.get("Cluster")
            celltype = row.get("Label 1")
            # Skip rows with missing or empty required fields
            if not (sample and cluster and celltype):
                continue
            # Create nested dict structure: sample -> cluster -> celltype
            label_maps.setdefault(sample, {})[cluster] = celltype

    return label_maps


def assign_celltype_colors(
    cluster_dict: Mapping[str, Any],
    label_map: Mapping[str, Mapping[str, str]],
    sample_id: str,
    cmap: str = "tab20",
) -> Dict[str, Tuple[float, float, float, float]]:
    """Assign a unique color to each cell type for a given sample.

    Maps each unique cell type (from label_map or fallback cluster names) to
    a unique RGBA color from the specified colormap. Colors are evenly
    distributed across the colormap range.

    Args:
        cluster_dict: Dictionary of cluster identifiers for a sample. Keys
            are used to look up cell types in label_map.
        label_map: Nested mapping ``{sample_id: {cluster_id: cell_type}}``.
        sample_id: Sample identifier used to access the relevant mapping in
            label_map.
        cmap: Matplotlib colormap name. Defaults to ``"tab20"``.

    Returns:
        Dictionary mapping each unique cell type to an RGBA color tuple
        ``(r, g, b, a)`` with values in the range [0.0, 1.0]. Returns an empty
        dictionary if cluster_dict is empty or no cell types are found.

    Raises:
        ValueError: If ``cmap`` is not a valid matplotlib colormap name.

    Notes:
        - If a cluster ID is not found in label_map, it uses the fallback
          name ``"Cluster {cluster_id}"``.
        - If ``sample_id`` is not in label_map, all clusters use fallback
          names.
        - Cluster IDs are converted to strings for matching (handles numeric
          cluster labels).
        - Cell types are sorted alphabetically before color assignment.
    """
    # Get sample-specific mapping (empty dict if sample_id not found)
    sample_map = label_map.get(sample_id, {})
    # Convert cluster IDs to strings for consistent matching
    # (handles cases where cluster IDs are numeric)
    sample_map_str_keys = {str(k): v for k, v in sample_map.items()}

    # Map each cluster to its cell type (or fallback name if not found)
    cell_types = [
        sample_map_str_keys.get(str(cluster_id), f"Cluster {cluster_id}")
        for cluster_id in cluster_dict.keys()
    ]

    # Get unique cell types, sorted for consistent color assignment
    unique_types = sorted(set(cell_types))
    if not unique_types:
        return {}

    # Generate evenly-spaced colors from the colormap
    cmap_obj = plt.get_cmap(cmap)
    colors = [
        cmap_obj(i / len(unique_types))
        for i in range(len(unique_types))
    ]

    return dict(zip(unique_types, colors))
