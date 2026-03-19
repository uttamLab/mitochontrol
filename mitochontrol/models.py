"""Mixture model fitting for mitochondrial RNA fraction distributions.

This module provides functions for fitting Gaussian, Negative Binomial,
Beta, and Poisson mixture models using online EM algorithms.
"""

import logging
from typing import (
    Any, Callable, Dict, Literal, Mapping, Optional,
    Sequence, Tuple, Union,
)

import numpy as np
from scipy.special import loggamma

from scipy.stats import (
    nbinom,
    beta as beta_dist,
    poisson,
    norm,
)

from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


def fit_empirical_histogram(
    data: Sequence[float],
    bins: Union[int, Sequence[float]] = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate an empirical probability density from sample data.

    Computes a normalized histogram (probability density) from the input data
    and returns the density values along with bin centers and edges for
    visualization or further analysis.

    Args:
        data: 1D array-like of numeric observations.
        bins: Number of bins (int) or custom bin edges (sequence of floats).
            If int, bins are automatically computed. Defaults to ``50``.

    Returns:
        Tuple containing:
            - pdf_empirical: Normalized density values (area under curve = 1).
                Shape: ``(n_bins,)``.
            - bin_centers: Midpoint of each histogram bin.
                Shape: ``(n_bins,)``.
            - bin_edges: Bin boundaries (n_bins + 1 edges define n_bins bins).
                Shape: ``(n_bins + 1,)``.

    Notes:
        - Uses ``density=True`` in numpy histogram, which normalizes the
          histogram so the integral over the range equals 1.
        - If ``data`` is empty, returns empty arrays with shapes determined
          by the bin specification.
        - Bin centers are computed as the midpoint between consecutive edges.
    """
    # Compute normalized histogram (density=True normalizes to area = 1)
    pdf_empirical, bin_edges = np.histogram(
        data, bins=bins, density=True
    )
    # Calculate bin centers as midpoints between consecutive edges
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return pdf_empirical, bin_centers, bin_edges


def kl_divergence(
    pdf_empirical: np.ndarray,
    bin_centers: np.ndarray,
    model_pdf_fn: Callable[[np.ndarray], np.ndarray],
    bin_width: Optional[float] = None,
    epsilon: float = 1e-10,
) -> float:
    """Compute KL divergence between an empirical histogram PDF and a model
    PDF.

    Computes the Kullback-Leibler divergence D_KL(P||Q) using a discrete
    Riemann sum approximation, where P is the empirical distribution and Q
    is the model distribution. Lower values indicate better agreement between
    the distributions.

    Args:
        pdf_empirical: Empirical density values per bin. Must have the same
            length as ``bin_centers``.
        bin_centers: X locations (midpoints) corresponding to the empirical
            bins. Used to evaluate the model PDF and infer bin width.
        model_pdf_fn: Function that takes an array of x-values and returns
            the model PDF evaluated at those points. Must return values with
            the same shape as the input.
        bin_width: Override for bin width used in the Riemann sum
            approximation. If None, inferred from spacing of ``bin_centers``
            (assumes uniform spacing).
        epsilon: Small constant used to clip PDF values away from zero to
            avoid log(0). Defaults to ``1e-10``.

    Returns:
        KL divergence value: ``∑ p(x) log[p(x)/q(x)] Δx``, where p is the
        empirical PDF, q is the model PDF, and Δx is the bin width.

    Raises:
        ValueError: If ``pdf_empirical`` and ``model_pdf_fn(bin_centers)``
            have different shapes, or if ``bin_width`` is None and
            ``bin_centers`` has fewer than 2 points.

    Notes:
        - The computation assumes uniform bin spacing when inferring
          ``bin_width`` from ``bin_centers``.
        - PDF values are clipped to be at least ``epsilon`` to ensure
          numerical stability in the logarithm.
        - KL divergence is non-negative and equals zero only when the
          distributions are identical (almost everywhere).
    """
    # Evaluate model PDF on the same grid as empirical data
    pdf_model = np.asarray(model_pdf_fn(bin_centers))

    # Validate that shapes match (prevents silent broadcasting errors)
    if pdf_empirical.shape != pdf_model.shape:
        raise ValueError(
            "pdf_empirical and model_pdf_fn(bin_centers) must have the "
            "same shape."
        )

    # Clip PDF values away from zero to avoid log(0) in KL divergence
    # p = empirical PDF, q = model PDF
    p = np.clip(np.asarray(pdf_empirical, dtype=float), epsilon, None)
    q = np.clip(pdf_model.astype(float), epsilon, None)

    # Infer bin width from bin_centers spacing (assumes uniform bins)
    if bin_width is None:
        if bin_centers.size < 2:
            raise ValueError(
                "bin_width is None and bin_centers has < 2 points; cannot "
                "infer bin width."
            )
        bin_width = float(bin_centers[1] - bin_centers[0])

    # Compute KL divergence: D_KL(P||Q) = ∫ p(x) log[p(x)/q(x)] dx
    # Approximated as discrete sum: ∑ p(x) log[p(x)/q(x)] Δx
    kl = float(np.sum(p * (np.log(p) - np.log(q))) * bin_width)
    return kl


def compute_kl_divergences(
    data: np.ndarray,
    gmm_model: Optional[Mapping[str, Any]] = None,
    nb_model: Optional[Mapping[str, Any]] = None,
    beta_model: Optional[Mapping[str, Any]] = None,
    poisson_model: Optional[Mapping[str, Any]] = None,
    bins: int = 50,
    scale_beta: bool = True,
) -> Dict[str, float]:
    """Compute KL divergence between an empirical histogram and each
    provided mixture model.

    Evaluates how well different mixture models (Gaussian, Negative Binomial,
    Beta, Poisson) fit the empirical distribution of the data by computing
    KL divergence. Models must be pre-fitted and provided in the expected
    dictionary format.

    Args:
        data: Original mtRNA data array (raw values, not KDE).
        gmm_model: Optional Gaussian mixture model with
            ``components=[(mean, var), ...]`` and ``weights=[...]``.
        nb_model: Optional Negative Binomial mixture with
            ``components=[(r, p), ...]`` and ``weights=[...]``.
        beta_model: Optional Beta mixture with ``components=[(a, b), ...]``
            and scalar ``k`` (number of components).
        poisson_model: Optional Poisson mixture with
            ``components=[lam1, lam2, ...]`` and ``weights=[...]``.
        bins: Number of bins for the empirical histogram. Defaults to ``50``.
        scale_beta: If True, normalize data to [0, 1] for the beta model
            (required for Beta distribution support). Defaults to ``True``.

    Returns:
        Dictionary mapping model names (e.g., "Gaussian", "NegBinomial") to
        their KL divergence values. Only includes models that were provided
        (non-None).

    Raises:
        ValueError: If ``scale_beta=False`` but beta_model is provided, or if
            data is empty and bin width cannot be inferred.

    Notes:
        - Discrete models (Negative Binomial, Poisson) round values to
          integers for PMF evaluation (scaled by 100 to preserve precision).
        - Beta model requires data on [0, 1] interval; automatic scaling is
          applied if ``scale_beta=True``.
        - All models are evaluated on the same bin grid as the empirical
          histogram.
    """
    results: Dict[str, float] = {}

    # Compute empirical distribution from histogram
    pdf_empirical, bin_centers, bin_edges = fit_empirical_histogram(
        data, bins
    )
    if len(bin_edges) < 2:
        raise ValueError(
            "Cannot compute bin width: histogram has fewer than 2 bin edges."
        )
    bin_width = float(bin_edges[1] - bin_edges[0])

    # Gaussian Mixture
    if gmm_model:
        def gaussian_mixture_pdf(x_vals: np.ndarray) -> np.ndarray:
            total = np.zeros_like(x_vals, dtype=float)
            for i, (mean, var) in enumerate(gmm_model["components"]):
                std = np.sqrt(var)
                weight = gmm_model["weights"][i]
                total += weight * norm.pdf(x_vals, mean, std)
            return total

        results["Gaussian"] = kl_divergence(
            pdf_empirical, bin_centers, gaussian_mixture_pdf, bin_width
        )

    # Negative Binomial Mixture (discrete distribution on rounded support)
    if nb_model:
        def nb_mixture_pdf(x_vals: np.ndarray) -> np.ndarray:
            total = np.zeros_like(x_vals, dtype=float)
            # Round to integers (scale by 100 to preserve precision)
            x_discrete = np.round(x_vals * 100).astype(int)
            for i, (r, p) in enumerate(nb_model["components"]):
                weight = nb_model["weights"][i]
                total += weight * nbinom.pmf(x_discrete, r, p)
            return total

        results["NegBinomial"] = kl_divergence(
            pdf_empirical, bin_centers, nb_mixture_pdf, bin_width
        )

    # Beta Mixture (requires data scaled to [0, 1] interval)
    if beta_model:
        if scale_beta:
            max_val = float(np.max(data))
            # Scale to [0, 1]; avoid division-by-zero if all values are zero
            data_scaled = (data / max_val) if max_val > 0 else data
            pdf_empirical_beta, bin_centers_beta, bin_edges_beta = (
                fit_empirical_histogram(data_scaled, bins)
            )
            if len(bin_edges_beta) < 2:
                raise ValueError(
                    "Cannot compute bin width for scaled beta data."
                )
            bin_width_beta = float(
                bin_edges_beta[1] - bin_edges_beta[0]
            )
        else:
            raise ValueError("Beta model requires data scaled to [0, 1]")

        def beta_mixture_pdf(x_vals: np.ndarray) -> np.ndarray:
            total = np.zeros_like(x_vals, dtype=float)
            for a, b in beta_model["components"]:
                total += beta_dist.pdf(x_vals, a, b) / beta_model["k"]
            return total

        results["Beta"] = kl_divergence(
            pdf_empirical_beta, bin_centers_beta, beta_mixture_pdf,
            bin_width_beta
        )

    # Poisson Mixture (discrete distribution on rounded support)
    if poisson_model:
        def poisson_mixture_pdf(x_vals: np.ndarray) -> np.ndarray:
            total = np.zeros_like(x_vals, dtype=float)
            # Round to integers (scale by 100 to preserve precision)
            x_discrete = np.round(x_vals * 100).astype(int)
            for i, lam in enumerate(poisson_model["components"]):
                weight = poisson_model["weights"][i]
                total += weight * poisson.pmf(x_discrete, lam)
            return total

        results["Poisson"] = kl_divergence(
            pdf_empirical, bin_centers, poisson_mixture_pdf, bin_width
        )

    return results


def initialize_gmm_params(
    data: np.ndarray,
    k: int,
    method: Literal["kmeans", "quantile", "random"] = "kmeans",
    min_var: float = 1e-6,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialize GMM parameters (means, variances, weights) using specified
    method.

    Provides three initialization strategies for Gaussian Mixture Models:
    k-means clustering, quantile-based binning, or random selection. All
    methods assign data points to clusters and compute initial parameters
    from these assignments.

    Args:
        data: 1D array of observations to cluster.
        k: Number of mixture components (must be between 1 and number of
            observations).
        method: Initialization strategy. "kmeans" uses KMeans clustering,
            "quantile" bins data by quantiles, "random" selects random data
            points as means. Defaults to ``"kmeans"``.
        min_var: Minimum variance floor applied to each component to ensure
            numerical stability. Defaults to ``1e-6``.
        seed: Random seed for reproducibility when using "random" method.
            Ignored for other methods. Defaults to ``None``.

    Returns:
        Tuple of three 1D arrays:
            - means: Component means, shape ``(k,)``.
            - variances: Component variances, shape ``(k,)``.
            - weights: Component mixing weights (sum to 1), shape ``(k,)``.

    Raises:
        ValueError: If ``k < 1`` or ``k > len(data)``.

    Notes:
        - K-means method uses a fixed random seed (0) for reproducibility.
        - Quantile method may produce empty clusters; empty clusters use the
          global median as mean and global variance as variance.
        - All variances are clipped to be at least ``min_var``.
        - Weights are computed as the fraction of points assigned to each
          cluster (minimum 1 point per cluster for numerical stability).
    """
    # Reshape to column vector for sklearn compatibility
    data_reshaped = np.asarray(data, dtype=float).reshape(-1, 1)
    if k < 1 or k > data_reshaped.shape[0]:
        raise ValueError(
            "k must be between 1 and the number of observations."
        )

    if method == "kmeans":
        # Use fixed seed for reproducibility (seed parameter only affects
        # "random" method)
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels = kmeans.fit_predict(data_reshaped)
        means = kmeans.cluster_centers_.ravel()

    elif method == "quantile":
        # Create k+1 quantiles to define k bins
        quantiles = np.quantile(
            data_reshaped.ravel(), np.linspace(0, 1, k + 1)
        )
        # Bin data using quantile boundaries (exclude first and last)
        labels = np.digitize(
            data_reshaped.ravel(), quantiles[1:-1], right=True
        )
        # Compute means per cluster; use global median if cluster is empty
        means = np.array([
            data_reshaped[labels == j].mean()
            if np.any(labels == j)
            else np.median(data_reshaped)
            for j in range(k)
        ]).ravel()

    else:  # "random"
        rng = np.random.default_rng(seed)
        # Select k random data points as initial means
        means = rng.choice(data_reshaped.ravel(), size=k, replace=False)
        # Assign each point to nearest mean (Euclidean distance)
        labels = np.argmin(
            (data_reshaped - means.reshape(1, -1))**2, axis=1
        )

    # Compute variances and weights from cluster assignments
    variances = np.empty(k, dtype=float)
    weights = np.empty(k, dtype=float)
    n_total = len(data_reshaped)
    for j in range(k):
        cluster_mask = (labels == j)
        # Ensure at least 1 point per cluster for weight calculation
        n_cluster = max(cluster_mask.sum(), 1)
        # Compute variance from cluster points; use global variance if empty
        cluster_points = data_reshaped[cluster_mask].ravel()
        diff = cluster_points - means[j]
        cluster_var = (
            diff.var() if cluster_mask.any() else data_reshaped.var()
        )
        # Clip variance to minimum floor for numerical stability
        variances[j] = max(cluster_var, min_var)
        # Weight is fraction of points in cluster
        weights[j] = n_cluster / n_total
    return means, variances, weights


def online_em_gmm(
    data: np.ndarray,
    init_method: str = "kmeans",
    max_components: int = 5,
    max_iter: int = 3,
    lr: float = 0.05,
    epsilon: float = 1e-6,
    seed: Optional[int] = None,
) -> Tuple[
    Optional[Dict[str, Any]], Optional[np.ndarray], float
]:
    """Fit Gaussian mixture models via online EM and select best by BIC.

    Fits GMMs with 1 to max_components components using an online EM algorithm
    that processes data point-by-point with stochastic gradient updates. The
    model with the lowest Bayesian Information Criterion (BIC) is selected.

    Args:
        data: 1D numeric array of observations. Non-finite values are filtered
            out.
        init_method: Initialization strategy for component parameters.
            Options: "kmeans", "quantile", "random". Defaults to ``"kmeans"``.
        max_components: Maximum number of mixture components to try. Models
            with 1 to max_components components are evaluated. Defaults to
            ``5``.
        max_iter: Number of online passes over the data for each component
            count. Each pass processes all data points once. Defaults to ``3``.
        lr: Learning rate for online parameter updates. Controls how quickly
            parameters adapt to new data. Defaults to ``0.05``.
        epsilon: Small constant added to weights and probabilities to prevent
            numerical issues (log(0), division by zero). Defaults to ``1e-6``.
        seed: Random seed for reproducibility when using "random"
            initialization. Defaults to ``None``.

    Returns:
        Tuple of three elements:
            - best_model: Dictionary with keys "components" (list of
              (mean, var) tuples), "weights" (1D array), and "k" (number of
              components). None if no valid model found.
            - best_labels: 1D array of cluster assignments (argmax of
              responsibilities) for the best model. None if no valid model
              found.
            - best_bic: BIC value for the best model. Lower is better.
              np.inf if no valid model found.

    Raises:
        ValueError: If data contains no finite observations.

    Notes:
        - Uses log-space computation for numerical stability when computing
          responsibilities.
        - Variance is clipped to a floor based on data variance to prevent
          collapse.
        - BIC penalty uses 3 parameters per component (mean, variance, weight),
          though weights sum to 1.
        - Online EM updates parameters incrementally as each data point is
          processed, rather than in batch.
    """
    # Filter to finite values and ensure 1D
    data_flat = np.asarray(data, dtype=float).ravel()
    data_flat = data_flat[np.isfinite(data_flat)]
    n = data_flat.size
    if n == 0:
        return None, None, np.inf
    # Set variance floor: use data variance if available, otherwise absolute
    # minimum
    if n > 1:
        var_floor = max(1e-12, 1e-6 * np.var(data_flat))
    else:
        var_floor = 1e-12
    log2pi = np.log(2.0 * np.pi)

    best_bic = np.inf
    best_model = None
    best_labels = None

    for k in range(1, max_components + 1):
        # Initialize parameters for k-component model
        means, variances, weights = initialize_gmm_params(
            data_flat, k, method=init_method, seed=seed
        )
        # Online EM: multiple passes over data
        for _ in range(max_iter):
            for x in data_flat:
                # Compute responsibilities in log-space for numerical stability
                # Standardize: z = (x - mean) / std
                scales = np.sqrt(np.maximum(variances, var_floor))
                z = (x - means) / scales
                z = np.clip(z, -100.0, 100.0)  # Prevent extreme values
                # Log probability: log(weight) + log(N(x|mean,var))
                log_probs = (
                    np.log(weights + epsilon) - 0.5 * (z**2) -
                    np.log(scales) - 0.5 * log2pi
                )
                # Log-sum-exp trick: subtract max before exp to prevent
                # overflow
                m = np.max(log_probs)
                probs = np.exp(log_probs - m)
                # Normalize to get responsibilities (posterior probabilities)
                gamma = probs / (np.sum(probs) + 1e-12)
                # Online parameter updates (stochastic gradient ascent)
                for j in range(k):
                    weights[j] += lr * (gamma[j] - weights[j])
                    means[j] += lr * gamma[j] * (x - means[j])
                    variances[j] += (
                        lr * gamma[j] * ((x - means[j])**2 - variances[j])
                    )
                    # Prevent variance collapse to zero
                    if variances[j] < var_floor:
                        variances[j] = var_floor

        # Compute final log-likelihood and responsibilities for BIC
        log_likelihood = 0.0
        responsibilities = np.zeros((n, k))
        for i, x in enumerate(data_flat):
            # Same log-space computation as in EM loop
            scales = np.sqrt(np.maximum(variances, var_floor))
            z = (x - means) / scales
            z = np.clip(z, -100.0, 100.0)
            log_probs = (
                np.log(weights + epsilon) - 0.5 * (z**2) -
                np.log(scales) - 0.5 * log2pi
            )
            m = np.max(log_probs)
            probs = np.exp(log_probs - m)
            prob_sum = np.sum(probs) + epsilon
            # Accumulate log-likelihood using log-sum-exp
            log_likelihood += (m + np.log(prob_sum))
            responsibilities[i] = probs / prob_sum

        # BIC = -2 * log_likelihood + penalty
        # Penalty: 3 parameters per component (mean, variance, weight)
        bic = -2 * log_likelihood + k * 3 * np.log(n)

        if bic < best_bic:
            best_bic = bic
            best_model = {
                "components": list(
                    zip(means.tolist(), variances.tolist())
                ),
                "weights": weights,
                "k": k
            }
            best_labels = np.argmax(responsibilities, axis=1)

    return best_model, best_labels, best_bic


def online_em_nbm(
    data: np.ndarray,
    max_components: int = 5,
    max_iter: int = 3,
    lr: float = 0.01,
    epsilon: float = 1e-6,
    seed: Optional[int] = None
) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray], float]:
    """Fit Negative Binomial mixture model via online EM with BIC selection.

    Fits NB mixture models with 1 to max_components components using an online
    EM algorithm. Data is discretized to integer counts (scaled by 100) for
    NB distribution support. The model with the lowest BIC is selected.

    Args:
        data: 1D array of observations in original mtRNA scale (float).
            Non-finite and negative values are filtered out.
        max_components: Maximum number of mixture components to try. Models
            with 1 to max_components components are evaluated. Defaults to
            ``5``.
        max_iter: Number of online passes over the data for each component
            count. Defaults to ``3``.
        lr: Learning rate for online parameter updates. Defaults to ``0.01``.
        epsilon: Small constant added to probabilities to prevent numerical
            issues. Defaults to ``1e-6``.
        seed: Random seed for parameter initialization. Defaults to ``None``.

    Returns:
        Tuple of three elements:
            - best_model: Dictionary with keys "components" (list of (r, p)
              tuples), "weights" (1D array), and "k" (number of components).
              None if no valid model found.
            - best_labels: 1D array of cluster assignments (argmax of
              responsibilities) for the best model. None if no valid model
              found.
            - best_bic: BIC value for the best model. Lower is better.
              np.inf if no valid model found.

    Notes:
        - Data is discretized by rounding to integers after scaling by 100.
        - Parameters are initialized randomly (means and variances).
        - NB parameters (r, p) are derived from mean and variance using
          method of moments: r = mean²/(variance - mean), p = mean/variance.
        - Requires variance > mean for valid NB parameterization.
    """
    # Discretize to NB support: round to integers (scale by 100 to preserve
    # precision) and filter to nonnegative counts
    data_int = np.round(np.asarray(data, dtype=float) * 100).astype(int)
    data_int = data_int[data_int >= 0]
    n = data_int.size
    if n == 0:
        raise ValueError(
            "No valid observations after filtering (all non-finite or "
            "negative)."
        )
    best_bic = np.inf
    best_model: Optional[Dict[str, Any]] = None
    best_labels: Optional[np.ndarray] = None

    def nb_logpmf(x: float, r: float, p: float) -> float:
        """Compute log PMF of Negative Binomial distribution.

        Formula: log(Γ(x+r)/(Γ(r)Γ(x+1))) + r*log(p) + x*log(1-p)
        """
        return (
            loggamma(x + r)
            - loggamma(r)
            - loggamma(x + 1)
            + r * np.log(p)
            + x * np.log(1 - p)
        )

    rng = np.random.default_rng(seed)
    for k in range(1, max_components + 1):
        # Initialize parameters: uniform weights, random means and variances
        weights = np.ones(k) / k
        means = rng.uniform(1, 10, k)
        variances = means + rng.uniform(1, 10, k)  # Ensure variance > mean

        # Online EM: multiple passes over data
        for _ in range(max_iter):
            for x_val in data_int:
                x_val = float(x_val)
                # Compute component probabilities (log-space for stability)
                probs = np.zeros(k)
                for j in range(k):
                    mean_j = means[j]
                    var_j = variances[j]
                    # Method of moments: convert (mean, variance) to (r, p)
                    # Requires variance > mean for valid NB
                    r = mean_j**2 / (var_j - mean_j)
                    p = mean_j / var_j
                    probs[j] = np.exp(nb_logpmf(x_val, r, p))
                # Weighted probabilities and responsibilities
                prob_weighted = probs * weights
                gamma = prob_weighted / (np.sum(prob_weighted) + 1e-12)
                # Online parameter updates
                for j in range(k):
                    weights[j] += lr * (gamma[j] - weights[j])
                    means[j] += lr * gamma[j] * (x_val - means[j])
                    variances[j] += (
                        lr * gamma[j] * ((x_val - means[j])**2 - variances[j])
                    )

        # Compute final log-likelihood and responsibilities for BIC
        log_likelihood = 0.0
        responsibilities = np.zeros((n, k))
        for i, x_val in enumerate(data_int):
            x_val = float(x_val)
            # Compute component probabilities
            probs = np.zeros(k)
            for j in range(k):
                mean_j = means[j]
                var_j = variances[j]
                r = mean_j**2 / (var_j - mean_j)
                p = mean_j / var_j
                probs[j] = np.exp(nb_logpmf(x_val, r, p))
            # Weighted probabilities and normalize
            prob_weighted = probs * weights
            prob_sum = np.sum(prob_weighted) + epsilon
            log_likelihood += np.log(prob_sum)
            responsibilities[i] = prob_weighted / prob_sum

        # BIC = -2 * log_likelihood + penalty
        # Penalty: 3 parameters per component (mean, variance, weight)
        bic = -2 * log_likelihood + k * 3 * np.log(n)

        if bic < best_bic:
            best_bic = bic
            # Convert (mean, variance) to (r, p) for final model
            best_model = {
                "components": [
                    (
                        means[i]**2 / (variances[i] - means[i]),  # r
                        means[i] / variances[i]  # p
                    )
                    for i in range(k)
                ],
                "weights": weights,
                "k": k
            }
            best_labels = np.argmax(responsibilities, axis=1)

    return best_model, best_labels, best_bic


def online_em_beta(
    data: np.ndarray,
    max_components: int = 5,
    max_iter: int = 3,
    lr: float = 0.01,
    epsilon: float = 1e-6,
    seed: Optional[int] = None
) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray], float]:
    """Fit Beta mixture model via online EM with BIC selection.

    Fits Beta mixture models with 1 to max_components components using an
    online EM algorithm. Data is scaled to [0, 1] interval (required for
    Beta distribution support) and clipped away from boundaries. The model
    with the lowest BIC is selected.

    Args:
        data: 1D array of observations (float). Non-finite values are
            filtered out.
        max_components: Maximum number of mixture components to try. Models
            with 1 to max_components components are evaluated. Defaults to
            ``5``.
        max_iter: Number of online passes over the data for each component
            count. Defaults to ``3``.
        lr: Learning rate for online parameter updates. Defaults to ``0.01``.
        epsilon: Small constant used for clipping data away from [0, 1]
            boundaries and numerical stability. Defaults to ``1e-6``.
        seed: Random seed for parameter initialization. Defaults to ``None``.

    Returns:
        Tuple of three elements:
            - best_model: Dictionary with keys "components" (list of (alpha,
              beta) tuples), "weights" (1D array), and "k" (number of
              components). None if no valid model found.
            - best_labels: 1D array of cluster assignments (argmax of
              responsibilities) for the best model. None if no valid model
              found.
            - best_bic: BIC value for the best model. Lower is better.
              np.inf if no valid model found.

    Raises:
        ValueError: If data contains no finite observations.

    Notes:
        - Data is automatically scaled to [0, 1] by dividing by maximum value.
        - Values are clipped to [epsilon, 1-epsilon] to avoid boundary issues.
        - Parameters are updated via method of moments (mean/variance) then
          converted back to (alpha, beta).
    """
    # Scale to [0, 1] interval (required for Beta distribution)
    data = np.asarray(data, dtype=float).ravel()
    data = data[np.isfinite(data)]
    if data.size == 0:
        raise ValueError(
            "No valid observations after filtering (all non-finite)."
        )
    max_val = float(np.max(data))
    # Scale by maximum; avoid division by zero if all values are zero
    data = (data / max_val) if max_val > 0 else data
    # Clip away from boundaries to avoid numerical issues
    data = np.clip(data, epsilon, 1 - epsilon)

    n = data.size
    best_bic = np.inf
    best_model = None
    best_labels = None

    rng = np.random.default_rng(seed)
    for k in range(1, max_components + 1):
        # Initialize parameters: uniform weights, random alpha and beta
        weights = np.ones(k) / k
        alphas = rng.uniform(1, 5, k)
        betas = rng.uniform(1, 5, k)

        # Online EM: multiple passes over data
        for _ in range(max_iter):
            for x_val in data:
                x_val = float(x_val)
                # Compute component probabilities (Beta PDF)
                probs = np.array([
                    beta_dist.pdf(x_val, a, b)
                    for a, b in zip(alphas, betas)
                ])
                # Responsibilities (posterior probabilities)
                prob_weighted = probs * weights
                gamma = prob_weighted / (np.sum(prob_weighted) + 1e-12)

                # Online parameter updates via method of moments
                for j in range(k):
                    weights[j] += lr * (gamma[j] - weights[j])
                    # Convert (alpha, beta) to (mean, variance)
                    mean = alphas[j] / (alphas[j] + betas[j])
                    var = (
                        (alphas[j] * betas[j]) /
                        ((alphas[j] + betas[j])**2 *
                         (alphas[j] + betas[j] + 1))
                    )
                    # Update mean and variance
                    mean += lr * gamma[j] * (x_val - mean)
                    var += lr * gamma[j] * ((x_val - mean)**2 - var)
                    # Convert back to (alpha, beta) from (mean, variance)
                    alpha = ((1 - mean) / var - 1 / mean) * mean**2
                    beta_ = alpha * (1 / mean - 1)
                    # Only update if parameters are valid (positive)
                    if alpha > 0 and beta_ > 0:
                        alphas[j] = alpha
                        betas[j] = beta_

        # Compute final log-likelihood and responsibilities for BIC
        log_likelihood = 0.0
        responsibilities = np.zeros((n, k))
        for i, x_val in enumerate(data):
            x_val = float(x_val)
            # Compute component probabilities
            probs = np.array([
                beta_dist.pdf(x_val, a, b) for a, b in zip(alphas, betas)
            ])
            # Weighted probabilities and normalize
            prob_weighted = probs * weights
            prob_sum = np.sum(prob_weighted) + epsilon
            log_likelihood += np.log(prob_sum)
            responsibilities[i] = prob_weighted / prob_sum

        # BIC = -2 * log_likelihood + penalty
        # Penalty: 3 parameters per component (alpha, beta, weight)
        bic = -2 * log_likelihood + k * 3 * np.log(n)

        if bic < best_bic:
            best_bic = bic
            best_model = {
                "components": list(zip(alphas, betas)),
                "weights": weights,
                "k": k
            }
            best_labels = np.argmax(responsibilities, axis=1)

    return best_model, best_labels, best_bic


def online_em_poisson(
    data: np.ndarray,
    max_components: int = 5,
    max_iter: int = 3,
    lr: float = 0.05,
    epsilon: float = 1e-6,
    seed: Optional[int] = None
) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray], float]:
    """Fit Poisson mixture model via online EM with BIC selection.

    Fits Poisson mixture models with 1 to max_components components using an
    online EM algorithm. Data is discretized to integer counts (scaled by
    100) for Poisson distribution support. The model with the lowest BIC is
    selected.

    Args:
        data: 1D array of observations in original mtRNA scale (float).
            Non-finite and negative values are filtered out.
        max_components: Maximum number of mixture components to try. Models
            with 1 to max_components components are evaluated. Defaults to
            ``5``.
        max_iter: Number of online passes over the data for each component
            count. Defaults to ``3``.
        lr: Learning rate for online parameter updates. Defaults to ``0.05``.
        epsilon: Small constant added to probabilities to prevent numerical
            issues. Defaults to ``1e-6``.
        seed: Random seed for parameter initialization. Defaults to ``None``.

    Returns:
        Tuple of three elements:
            - best_model: Dictionary with keys "components" (list of lambda
              values), "weights" (1D array), and "k" (number of components).
              None if no valid model found.
            - best_labels: 1D array of cluster assignments (argmax of
              responsibilities) for the best model. None if no valid model
              found.
            - best_bic: BIC value for the best model. Lower is better.
              np.inf if no valid model found.

    Notes:
        - Data is discretized by rounding to integers after scaling by 100.
        - Parameters (lambda values) are initialized randomly.
        - BIC penalty uses 2 parameters per component (lambda, weight), though
          weights sum to 1.
    """
    # Discretize to Poisson support: round to integers (scale by 100 to
    # preserve precision) and filter to nonnegative counts
    data_int = np.round(np.asarray(data, dtype=float) * 100).astype(int)
    data_int = data_int[data_int >= 0]
    n = data_int.size
    if n == 0:
        raise ValueError(
            "No valid observations after filtering (all non-finite or "
            "negative)."
        )
    best_bic = np.inf
    best_model: Optional[Dict[str, Any]] = None
    best_labels: Optional[np.ndarray] = None

    rng = np.random.default_rng(seed)
    for k in range(1, max_components + 1):
        # Initialize parameters: uniform weights, random lambda values
        weights = np.ones(k) / k
        lambdas = rng.uniform(1, 10, k)

        # Online EM: multiple passes over data
        for _ in range(max_iter):
            for x_val in data_int:
                x_val = float(x_val)
                # Compute component probabilities (Poisson PMF)
                probs = np.array([poisson.pmf(x_val, lam) for lam in lambdas])
                # Weighted probabilities and responsibilities
                prob_weighted = probs * weights
                gamma = prob_weighted / (np.sum(prob_weighted) + 1e-12)
                # Online parameter updates
                for j in range(k):
                    weights[j] += lr * (gamma[j] - weights[j])
                    lambdas[j] += lr * gamma[j] * (x_val - lambdas[j])

        # Compute final log-likelihood and responsibilities for BIC
        log_likelihood = 0.0
        responsibilities = np.zeros((n, k))
        for i, x_val in enumerate(data_int):
            x_val = float(x_val)
            # Compute component probabilities
            probs = np.array([poisson.pmf(x_val, lam) for lam in lambdas])
            # Weighted probabilities and normalize
            prob_weighted = probs * weights
            prob_sum = np.sum(prob_weighted) + epsilon
            log_likelihood += np.log(prob_sum)
            responsibilities[i] = prob_weighted / prob_sum

        # BIC = -2 * log_likelihood + penalty
        # Penalty: 2 parameters per component (lambda, weight)
        bic = -2 * log_likelihood + k * 2 * np.log(n)

        if bic < best_bic:
            best_bic = bic
            best_model = {
                "components": lambdas.tolist(),
                "weights": weights,
                "k": k
            }
            best_labels = np.argmax(responsibilities, axis=1)

    return best_model, best_labels, best_bic
