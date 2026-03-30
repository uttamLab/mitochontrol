# Setup: conda env `sc_preprocessing` from sc_preprocessing.yml (override name via
# SC_PREPROCESSING_CONDA_ENV). Auto-installs missing R deps (rlang>=1.1.3, Bioc
# DropletUtils/SingleCellExperiment, GitHub DoubletFinder, CRAN scCustomize).
# If rlang is too old and already loaded, restart R after updating rlang.
# Reticulate always uses that conda env's Python.

SC_PREPROCESSING_CONDA_ENV <- Sys.getenv(
  "SC_PREPROCESSING_CONDA_ENV",
  unset = "sc_preprocessing"
)

{
  r <- "https://cloud.r-project.org"
  inst <- function(p) install.packages(p, repos = r, quiet = TRUE)
  need <- function(p, min = NULL) {
    ok <- requireNamespace(p, quietly = TRUE) &&
      (is.null(min) || packageVersion(p) >= package_version(min))
    if (ok) {
      return()
    }
    if (requireNamespace(p, quietly = TRUE) && isNamespaceLoaded(p)) {
      stop(
        "Restart R, update ", p, if (!is.null(min)) paste0(" (>= ", min, ")"),
        " (e.g. install.packages or conda update r-rlang), re-source.",
        call. = FALSE
      )
    }
    inst(p)
  }
  need("rlang", "1.1.3")
  for (p in c("DropletUtils", "SingleCellExperiment")) {
    if (!requireNamespace(p, quietly = TRUE)) {
      if (!requireNamespace("BiocManager", quietly = TRUE)) {
        inst("BiocManager")
      }
      BiocManager::install(p, ask = FALSE, update = FALSE)
    }
  }
  if (!requireNamespace("DoubletFinder", quietly = TRUE)) {
    if (!requireNamespace("remotes", quietly = TRUE)) {
      inst("remotes")
    }
    remotes::install_github(
      "chris-mcginnis-ucsf/DoubletFinder",
      upgrade = "never"
    )
  }
  if (!requireNamespace("scCustomize", quietly = TRUE)) {
    inst("scCustomize")
  }
}

library(reticulate)
tryCatch(
  use_condaenv(SC_PREPROCESSING_CONDA_ENV, required = TRUE),
  error = function(e) {
    stop(
      "use_condaenv('", SC_PREPROCESSING_CONDA_ENV, "') failed — ",
      "create env: conda env create -f sc_preprocessing.yml; put conda on PATH.\n",
      conditionMessage(e),
      call. = FALSE
    )
  }
)
if (nzchar(Sys.getenv("CONDA_DEFAULT_ENV", "")) &&
    Sys.getenv("CONDA_DEFAULT_ENV") != SC_PREPROCESSING_CONDA_ENV) {
  warning(
    "Activate ", SC_PREPROCESSING_CONDA_ENV, " before R so R and Python libs match.",
    call. = FALSE
  )
}
py_config()

library(Matrix)
library(Seurat)
library(DropletUtils)
library(DoubletFinder)
library(SoupX)
library(SingleCellExperiment)
library(scCustomize)

# ---------------------------------------------------------------------------
# MitoChontrol preprocessing helper
#
# Recommended workflow:
#   1. Run `sc_preprocessing()` on one or more Cell Ranger output folders.
#   2. Confirm the exported .h5ad file contains:
#        - layers["raw_counts"]
#        - layers["adjusted_counts"]
#        - layers["lognorm"]
#   3. Pass the resulting .h5ad into the Python `clustering()` pipeline.
#
# Notes:
#   - This script uses `scCustomize::as.anndata()` as the base .h5ad export
#     step, then re-opens the file with reticulate/anndata to attach clearly
#     named layers for downstream Python use.
#   - Python is always the interpreter from conda env `sc_preprocessing`
#     (or SC_PREPROCESSING_CONDA_ENV). Per-call overrides still go through
#     `conda_env` in `sc_preprocessing()` / `write_seurat_h5ad()`.
#   - `raw_counts` stores the original count matrix after cell filtering.
#   - `adjusted_counts` stores the post-SoupX matrix used to create the Seurat
#     object. If SoupX is disabled, this is identical to `raw_counts`.
#   - `lognorm` stores normalized, log-transformed values for analysis.
# ---------------------------------------------------------------------------


resolve_python_binary <- function(conda_env = NULL) {
  active_python <- Sys.which("python")
  active_env <- Sys.getenv("CONDA_DEFAULT_ENV", unset = "")
  active_prefix <- Sys.getenv("CONDA_PREFIX", unset = "")

  if (is.null(conda_env) || identical(conda_env, "")) {
    cfg <- reticulate::py_config()
    bound <- cfg$python
    if (nzchar(bound)) {
      return(bound)
    }
    if (nzchar(active_python)) {
      return(active_python)
    }
    stop(
      "No Python interpreter is configured. ",
      "use_condaenv('", SC_PREPROCESSING_CONDA_ENV, "') should run at startup.",
      call. = FALSE
    )
  }

  if (file.exists(conda_env)) {
    return(conda_env)
  }

  if (dir.exists(conda_env)) {
    env_python <- file.path(conda_env, "bin", "python")
    if (file.exists(env_python)) {
      return(env_python)
    }
  }

  if (nzchar(active_python) &&
      (identical(active_prefix, conda_env) ||
       identical(active_env, conda_env) ||
       identical(active_env, basename(conda_env)))) {
    return(active_python)
  }

  stop(
    paste(
      "Could not resolve a Python binary for `conda_env =",
      shQuote(conda_env),
      "`. Use an activated conda environment or pass a full env path"
    )
  )
}


configure_python <- function(conda_env = NULL, required = FALSE) {
  python_bin <- resolve_python_binary(conda_env = conda_env)
  reticulate::use_python(python_bin, required = required)
}


get_python_modules <- function(conda_env = NULL, required = FALSE) {
  configure_python(conda_env = conda_env, required = required)
  list(
    ad = reticulate::import("anndata", convert = FALSE),
    sp = reticulate::import("scipy.sparse", convert = FALSE)
  )
}


get_expression_matrix <- function(x, feature_type = "Gene Expression") {
  if (!is.list(x)) {
    return(methods::as(x, "dgCMatrix"))
  }
  if (!is.null(feature_type) && feature_type %in% names(x)) {
    return(methods::as(x[[feature_type]], "dgCMatrix"))
  }
  if ("Gene Expression" %in% names(x)) {
    return(methods::as(x[["Gene Expression"]], "dgCMatrix"))
  }
  methods::as(x[[1]], "dgCMatrix")
}


get_assay_matrix <- function(seurat_obj, assay = NULL, layer = c("counts", "data")) {
  assay <- assay %||% DefaultAssay(seurat_obj)
  layer <- match.arg(layer)

  if ("LayerData" %in% getNamespaceExports("SeuratObject")) {
    return(SeuratObject::LayerData(seurat_obj, assay = assay, layer = layer))
  }

  Seurat::GetAssayData(seurat_obj, assay = assay, slot = layer)
}


`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}


r_sparse_to_py_csc <- function(mat, sp_module) {
  mat <- methods::as(mat, "dgCMatrix")
  sp_module$csc_matrix(
    reticulate::tuple(list(
      mat@x,
      as.integer(mat@i),
      as.integer(mat@p)
    )),
    shape = reticulate::tuple(as.integer(dim(mat)))
  )
}


write_seurat_h5ad <- function(
    seurat_obj,
    raw_counts,
    adjusted_counts,
    output_path,
    assay = NULL,
    raw_layer_name = "raw_counts",
    adjusted_layer_name = "adjusted_counts",
    analyzed_layer_name = "lognorm",
    conda_env = NULL,
    python_required = FALSE
) {
  # scCustomize handles the initial .h5ad conversion reliably from Seurat.
  # We then annotate the exported AnnData object with explicit layer names
  # that the downstream MitoChontrol Python pipeline expects.
  modules <- get_python_modules(conda_env = conda_env, required = python_required)
  ad <- modules$ad
  sp <- modules$sp

  output_dir <- dirname(output_path)
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  }

  tmp_path <- tempfile(pattern = "seurat_export_", fileext = ".h5ad")
  scCustomize::as.anndata(
    x = seurat_obj,
    file_path = dirname(tmp_path),
    file_name = basename(tmp_path)
  )

  adata <- ad$read_h5ad(tmp_path)

  assay <- assay %||% DefaultAssay(seurat_obj)
  final_cells <- colnames(seurat_obj)
  final_genes <- rownames(seurat_obj)

  raw_counts <- methods::as(
    raw_counts[final_genes, final_cells, drop = FALSE],
    "dgCMatrix"
  )
  adjusted_counts <- methods::as(
    adjusted_counts[final_genes, final_cells, drop = FALSE],
    "dgCMatrix"
  )
  analyzed_counts <- methods::as(
    get_assay_matrix(seurat_obj, assay = assay, layer = "data")[
      final_genes, final_cells, drop = FALSE
    ],
    "dgCMatrix"
  )

  adata$layers[[raw_layer_name]] <- r_sparse_to_py_csc(t(raw_counts), sp)
  adata$layers[[adjusted_layer_name]] <- r_sparse_to_py_csc(t(adjusted_counts), sp)
  adata$layers[[analyzed_layer_name]] <- r_sparse_to_py_csc(t(analyzed_counts), sp)
  adata$X <- adata$layers[[analyzed_layer_name]]
  adata$uns[["preprocessing_layers"]] <- reticulate::dict(
    raw = raw_layer_name,
    adjusted = adjusted_layer_name,
    analyzed = analyzed_layer_name
  )

  adata$write_h5ad(output_path)
  unlink(tmp_path)
}


sc_preprocessing <- function(
    sample_folders = c("SAMPLE_001"),
    base_dir = "/path/to/project_root",
    out_folder = "preprocessing_run",
    cellranger_subdir = "CellRanger",
    outs_subdir = "outs",
    raw_matrix_subdir = "raw_feature_bc_matrix",
    output_subdir = "preprocessed",
    gene_expression_feature = "Gene Expression",
    assay_name = "RNA",
    use_doubletfinder = TRUE,
    use_ED = TRUE,
    use_soupx = FALSE,
    high_cont = FALSE,
    emptydrops_fdr = 0.01,
    min_cells = 3,
    min_features = 200,
    nfeatures = 2000,
    npcs = 20,
    raw_layer_name = "raw_counts",
    adjusted_layer_name = "adjusted_counts",
    analyzed_layer_name = "lognorm",
    conda_env = NULL,
    python_required = TRUE,
    seed = 42
) {
  # `sample_folders` can be either:
  #   - an unnamed character vector of Cell Ranger folder names, or
  #   - a named character vector where names are output labels and values are
  #     Cell Ranger folder names.
  #
  # Example:
  #   c("Sample_A" = "SRR000001", "Sample_B" = "SRR000002")

  set.seed(seed)

  destination <- file.path(base_dir, out_folder, output_subdir)
  if (!dir.exists(destination)) {
    dir.create(destination, recursive = TRUE, showWarnings = FALSE)
  }

  sample_labels <- names(sample_folders)
  if (is.null(sample_labels)) {
    sample_labels <- sample_folders
  }

  outputs <- vector("list", length(sample_folders))
  names(outputs) <- sample_labels

  for (idx in seq_along(sample_folders)) {
    folder <- sample_folders[[idx]]
    sample_label <- sample_labels[[idx]]
    message("\nProcessing sample: ", sample_label, " (folder: ", folder, ")")

    outputs[[sample_label]] <- tryCatch({
      cellranger_output <- file.path(base_dir, cellranger_subdir, folder, outs_subdir)
      raw_outputs <- file.path(cellranger_output, raw_matrix_subdir)
      out_path <- file.path(destination, paste0(sample_label, ".h5ad"))

      raw_counts <- get_expression_matrix(
        Read10X(raw_outputs),
        feature_type = gene_expression_feature
      )

      if (use_ED) {
        message(" - Performing EmptyDrops filtering...")
        ed_result <- emptyDrops(raw_counts)
        cell_barcodes <- rownames(ed_result)[which(ed_result$FDR < emptydrops_fdr)]
      } else {
        message(" - Skipping EmptyDrops: using all cell barcodes in raw matrix...")
        cell_barcodes <- colnames(raw_counts)
      }

      cell_barcodes <- intersect(colnames(raw_counts), cell_barcodes)
      raw_counts_filtered <- raw_counts[, cell_barcodes, drop = FALSE]

      if (use_soupx) {
        message(" - Performing SoupX correction...")
        soup_channel <- SoupX::load10X(cellranger_output)
        if (high_cont) {
          soup_channel <- SoupX::autoEstCont(
            soup_channel,
            contaminationRange = c(0.01, 0.4)
          )
        } else {
          soup_channel <- SoupX::autoEstCont(soup_channel)
        }
        adjusted_counts <- get_expression_matrix(
          SoupX::adjustCounts(soup_channel),
          feature_type = gene_expression_feature
        )
        adjusted_counts_filtered <- adjusted_counts[, cell_barcodes, drop = FALSE]
      } else {
        adjusted_counts_filtered <- raw_counts_filtered
      }

      seurat_obj <- CreateSeuratObject(
        counts = adjusted_counts_filtered,
        assay = assay_name,
        min.cells = min_cells,
        min.features = min_features
      )

      seurat_obj <- NormalizeData(seurat_obj, verbose = FALSE)
      seurat_obj <- FindVariableFeatures(
        seurat_obj,
        selection.method = "vst",
        nfeatures = nfeatures,
        verbose = FALSE
      )
      seurat_obj <- ScaleData(seurat_obj, verbose = FALSE)
      seurat_obj <- RunPCA(seurat_obj, npcs = npcs, verbose = FALSE)

      if (use_doubletfinder) {
        message(" - Running DoubletFinder...")
        n_cells <- ncol(seurat_obj)
        doublet_rate <- ifelse(n_cells <= 10000, 0.008 * (n_cells / 1000), 0.1)
        nExp <- round(doublet_rate * n_cells)

        sweep.res.list <- paramSweep(seurat_obj, PCs = 1:npcs, sct = FALSE)
        sweep.stats <- summarizeSweep(sweep.res.list, GT = FALSE)
        bcmvn <- find.pK(sweep.stats)
        optimal_pK <- as.numeric(
          as.character(bcmvn$pK[which.max(bcmvn$BCmetric)])
        )

        seurat_obj <- doubletFinder(
          seurat_obj,
          PCs = 1:npcs,
          pN = 0.25,
          pK = optimal_pK,
          nExp = nExp,
          reuse.pANN = FALSE,
          sct = FALSE
        )

        doublet_col <- grep(
          "DF.classifications",
          colnames(seurat_obj@meta.data),
          value = TRUE
        )
        singlet_cells <- colnames(seurat_obj)[
          seurat_obj@meta.data[[doublet_col]] == "Singlet"
        ]
        seurat_obj <- subset(seurat_obj, cells = singlet_cells)
        seurat_obj <- NormalizeData(seurat_obj, verbose = FALSE)
      } else {
        message(" - Skipping DoubletFinder step...")
      }

      write_seurat_h5ad(
        seurat_obj = seurat_obj,
        raw_counts = raw_counts_filtered,
        adjusted_counts = adjusted_counts_filtered,
        output_path = out_path,
        assay = assay_name,
        raw_layer_name = raw_layer_name,
        adjusted_layer_name = adjusted_layer_name,
        analyzed_layer_name = analyzed_layer_name,
        conda_env = conda_env,
        python_required = python_required
      )

      message("Successfully processed: ", sample_label)
      list(
        sample = sample_label,
        source_folder = folder,
        output_path = out_path,
        n_cells = ncol(seurat_obj),
        n_genes = nrow(seurat_obj),
        raw_layer = raw_layer_name,
        adjusted_layer = adjusted_layer_name,
        analyzed_layer = analyzed_layer_name
      )
    }, error = function(e) {
      message("\n! Error in sample ", sample_label, ": ", e$message)
      list(
        sample = sample_label,
        source_folder = folder,
        error = e$message
      )
    })
  }

  message("\nProcessing completed. Output saved to: ", destination)
  invisible(outputs)
}


# ---------------------------------------------------------------------------
# Usage templates
# ---------------------------------------------------------------------------

# Minimal example:
# results <- sc_preprocessing(
#   sample_folders = c("Sample_A" = "CELLRANGER_FOLDER_A"),
#   base_dir = "/path/to/project_root",
#   out_folder = "example_run",
#   conda_env = "YOUR_RETICULATE_ENV",
#   python_required = TRUE
# )

# Multi-sample example:
# sample_map <- c(
#   "Sample_A" = "CELLRANGER_FOLDER_A",
#   "Sample_B" = "CELLRANGER_FOLDER_B",
#   "Sample_C" = "CELLRANGER_FOLDER_C"
# )
#
# results <- sc_preprocessing(
#   sample_folders = sample_map,
#   base_dir = "/path/to/project_root",
#   out_folder = "study_preprocessing",
#   cellranger_subdir = "CellRanger",
#   use_doubletfinder = TRUE,
#   use_ED = TRUE,
#   use_soupx = TRUE,
#   conda_env = "YOUR_RETICULATE_ENV",
#   python_required = TRUE,
#   raw_layer_name = "raw_counts",
#   adjusted_layer_name = "adjusted_counts",
#   analyzed_layer_name = "lognorm"
# )

# After preprocessing, the intended Python-side handoff is:
#   clustered[label] = clustering(adata, label=label, outdir=...)
#   thresholds = get_thresholds(adatas=clustered, outdir=...)
