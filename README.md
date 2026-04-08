# MitoChontrol

Clustering-aware mitochondrial RNA thresholding for single-cell RNA-seq
quality control.

MitoChontrol uses per-cluster naive Bayes modeling to identify
tissue- and cell-type-specific mtRNA thresholds, replacing the
conventional fixed 10 % cutoff with data-driven, probability-based
filtering.

## Installation

Create and activate the mitochontrol python environment
```bash
conda env create -f mitochontrol.yml
conda activate mitochontrol
```

Install using pip

```bash
pip install mitochontrol
```

Or, install from source:

```bash
git clone https://github.com/uttamLab/mitochontrol.git
cd mitochontrol
pip install -e .
```

## Quick start

### Cluster-aware pipeline

The recommended workflow clusters the data first, then applies
per-cluster thresholds:

```python
import scanpy as sc
from mitochontrol import clustering, mtctrl_with_clustering

adata = sc.read_h5ad("sample.h5ad")

# Step 1 — cluster
result = clustering(adata, label="Sample1", outdir="output")

# Step 2 — threshold per cluster
thresholds = mtctrl_with_clustering(
    adatas={"Sample1": result},
    outdir="output",
    threshold_probs=(0.8,),
)

adata_out = thresholds["Sample1"]["adata"]
```

### Single-cluster pipeline

For pre-isolated populations or quick exploration:

```python
from mitochontrol import mtctrl_without_clustering

stats = mtctrl_without_clustering(
    adata,
    sample_id="Sample1",
    outdir="output",
)
```

### Optional cell-type annotation

If marker genes are available, `clustering()` can assign cell types
automatically:

```python
result = clustering(
    adata,
    label="Sample1",
    outdir="output",
    marker_genes="markers.csv",   # or a {celltype: [genes]} dict
)
```

## Output layout

Both pipelines write results under `outdir`:

```
outdir/
├── clustered/
│   ├── adata/Sample1.h5ad
│   ├── umap/Sample1.pdf
│   ├── res_selection/Sample1.pdf
│   ├── DEG/Sample1.csv
│   └── celltype_labels/Sample1.csv
└── mitochontrol/
    ├── adata/Sample1.h5ad
    ├── cluster_overlays/Sample1.pdf
    ├── threshold/Sample1_cluster0_0.8.pdf
    ├── enrichment/Sample1_cluster0_0.8.pdf
    ├── filtered_umap/Sample1_cluster0_0.8.pdf
    └── threshold_stats.csv
```

The single-sample pipeline ``mtctrl_without_clustering`` does not write
cluster overlays or filtered UMAPs; it saves mt-vs-UMI scatters under
``mitochontrol/scatter/`` (initial plot plus one threshold-colored file per
probability).

## Tutorial

A step-by-step Jupyter notebook is included in
[`MitoChontrol_tut.ipynb`](MitoChontrol_tut.ipynb),
demonstrating both the clustered and single-cluster workflows with
heuristic overrides and result visualization. The data used in the notebook can be downloaded from [`https://doi.org/10.5281/zenodo.19423054`](https://doi.org/10.5281/zenodo.19423054)

## Citation

If you use MitoChontrol in your research, please cite the associated
publication:

> Strassburg *et al.* (2026). MitoChontrol: Adaptive mitochondrial filtering for robust single-cell RNA sequencing quality control.
> *Journal*, **volume**, pages. doi:XXXX

BibTeX:

```bibtex
@article {Strassburg20260404,
	author = {Strassburg, Caitlin and Pitlor, Danielle and Singhi, Aatur D and Gottschalk, Rachel and Uttam, Shikhar},
	title = {MitoChontrol: Adaptive mitochondrial filtering for robust single-cell RNA sequencing quality control},
	elocation-id = {2026.04.04.716517},
	year = {2026},
	doi = {10.64898/2026.04.04.716517},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2026/04/07/2026.04.04.716517},
	eprint = {https://www.biorxiv.org/content/early/2026/04/07/2026.04.04.716517.full.pdf},
	journal = {bioRxiv}
}

```

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.
