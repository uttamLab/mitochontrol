# MitoChontrol

Clustering-aware mitochondrial RNA thresholding for single-cell RNA-seq
quality control.

MitoChontrol uses per-cluster naive Bayes modeling to identify
tissue- and cell-type-specific mtRNA thresholds, replacing the
conventional fixed 10 % cutoff with data-driven, probability-based
filtering.

## Installation

```bash
pip install mitochontrol
```

Or install from source:

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
from mitochontrol import clustering, get_thresholds

adata = sc.read_h5ad("sample.h5ad")

# Step 1 — cluster
result = clustering(adata, label="Sample1", outdir="output")

# Step 2 — threshold per cluster
thresholds = get_thresholds(
    adatas={"Sample1": result},
    outdir="output",
    threshold_probs=(0.8,),
)

adata_out = thresholds["Sample1"]["adata"]
```

### Single-cluster pipeline

For pre-isolated populations or quick exploration:

```python
from mitochontrol import single_cluster_mitochontrol

stats = single_cluster_mitochontrol(
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

## Tutorial

A step-by-step Jupyter notebook is included in
[`examples/MitoChontrol_tut.ipynb`](examples/MitoChontrol_tut.ipynb),
demonstrating both the clustered and single-cluster workflows with
heuristic overrides and result visualization.

## Citation

If you use MitoChontrol in your research, please cite the associated
publication:

> Strassburg *et al.* (2026). MitoChontrol: Adaptive mitochondrial filtering for robust single-cell RNA sequencing quality control.
> *Journal*, **volume**, pages. doi:XXXX

BibTeX:

```bibtex
@article{strauss2026mitochontrol,
  title   = {MitoChontrol: Adaptive mitochondrial filtering for 
             robust single-cell RNA sequencing quality control},
  author  = {Strassburg, C. M. and others},
  journal = {Journal},
  year    = {2026},
  doi     = {XXXX}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
