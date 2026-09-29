# MicroDNA Map v2.0

MicroDNA Map v2.0 is an integrated computational platform for extrachromosomal circular DNA (eccDNA / microDNA) identification, annotation, and benchmark evaluation. The framework integrates a 1D ResNet-SelfAttention deep learning classifier, a two-stage sliding-window refinement mechanism, and micro-scale local coverage profiling algorithm (`micro_coverage`) designed specifically to address the physical resolution of microDNA.

---

## Project Structure

```text
MicroDNA_Map/
├── config.py                  # Global configurations and parameters
├── requirements.txt           # Python dependency specifications
├── data/                      # Raw and processed sequence data
│   ├── raw/                   # Raw annotations and FASTQ files
│   └── processed/             # Cleaned datasets (eccDNA.fa, otherDNA.fa)
├── refs/                      # Reference genomes and indices
├── models/                    # Model checkpoints and training logs
│   └── cv/                    # 5-fold cross-validation checkpoints and summary metrics
├── results/                   # Predictions, metrics, and intermediate outputs
├── src/                       # Core algorithm implementation
│   ├── model.py               # ResNetSelfAttention network architecture
│   ├── dataloader.py          # Lazy binary-seeking dataset loader with chromosome CV splitting
│   ├── dataprocess.py         # One-hot encoding and FASTA parsing utilities
│   ├── hnm.py                 # Multi-round hard negative mining engine
│   ├── utils.py               # External process execution and file utilities
│   └── pipeline/              # Upstream candidate region screening pipelines
│       ├── micro_coverage_pipeline.py # Project-developed micro-coverage scanner
│       └── cnvkit_pipeline.py         # Traditional macro-CNV calling pipeline
├── scripts/                   # Workflow scripts
│   ├── process_data.py        # Sequence extraction from coordinate tables
│   ├── sample_negatives.py    # Genomic background negative sequence sampling
│   ├── train.py               # Chromosome 5-fold CV training with multi-round HNM and AMP
│   ├── verify.py              # Performance evaluation and metrics visualization
│   ├── compare_hnm.py         # HNM probability distribution comparison
│   ├── predict.py             # High-throughput batch and ensemble inference
│   ├── batch_process.py       # End-to-end processing pipeline from FASTQ inputs
│   └── ablation_layer_size.py # Channel capacity ablation experiments
└── benchmark/                 # Spike-in simulation, detection, and evaluation
    ├── config_real.yaml       # Real-data spike-in simulation configuration
    ├── config_random.yaml     # Random-data spike-in simulation configuration
    ├── run_benchmark.py       # Orchestrated 10-phase benchmark pipeline
    ├── microdna_junction_rescuer.py # Head-to-tail junction read rescuer
    ├── perturbation_test.py   # Sequence feature perturbation and mutagenesis testing
    ├── models_comparison/     # Comparative baselines (ResNet50, Transformer)
    ├── simulate/              # Circle-seq and WGS in silico sequencing simulators
    ├── detect/                # Detection wrappers (MicroDNA Map, Circle-Map)
    ├── evaluate/              # Overlap resolution and metrics calculator
    └── visualize/             # Benchmark visualization tools

```

---

## Installation

Create a dedicated Conda environment and install dependencies:

```bash
conda create -n microdna python=3.9 -y
conda activate microdna
pip install -r requirements.txt

# Install PyTorch matching your CUDA runtime (example: CUDA 12.1)
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install seaborn cnvkit pysam openpyxl

```

Ensure the following external binaries are installed and available in your system `PATH`:

* `bowtie2` and `bowtie2-build`
* `bwa`
* `samtools`
* `cnvkit.py` (optional, required only when using the legacy CNVkit pipeline)
* `Circle-Map` (optional, required only for benchmark evaluations)

---

## Quick Start

### 1. Sequence Inference Prediction (`scripts/predict.py`)

Directly score or scan given FASTA sequences using trained model checkpoints or 5-fold cross-validation soft-ensemble models.

Common Arguments:

* `--input`: Path to input FASTA file or directory containing FASTA files.
* `--mode`: Inference mode. `short` for fixed-length (400 bp) single-sequence classification; `long` for sliding-window scanning along longer sequences. Default: `long`.
* `--model`: Path to model weights (`.pth`) or a directory containing CV fold weights (e.g., `models/cv`) for ensemble voting. Default: `models/cv` or `models/best_model.pth`.
* `--limit`: Probability threshold for positive classification. Default: `0.75`.
* `--batch-size`: Batch size for GPU inference. Default: `512`.
* `--min-region-len`: Minimum retained candidate length (bp) in `long` mode. Default: `150`.
* `--no-merge`: Disable automatic merging of adjacent/overlapping predicted windows.
* `--output-dir`: Output directory for generated BED and FASTA files.

```bash
# Short sequence mode: Outputs classification probability TSV
python scripts/predict.py --input data/test_short.fa --mode short --batch-size 512

# Long sequence mode (Single model): Scans sequences and outputs candidate BED and FASTA
python scripts/predict.py --input data/test_long.fa --mode long --model models/best_model.pth --limit 0.75 --output-dir results/predictions

# Long sequence mode (5-Fold Soft Ensemble): Smooths variance across chromosomal folds
python scripts/predict.py --input data/test_long.fa --mode long --model models/cv --limit 0.75 --output-dir results/predictions

```

### 2. Direct Reference Genome Scanning (`benchmark/detect/run_microdna_map_direct.py`) [Not Recommended]

> **Note:** Direct scanning across whole chromosomes without read alignment or coverage pre-screening leads to excessive computational overhead and a high risk of false positives. This script is intended primarily for theoretical ablation and methodology comparisons. For standard biological sample analysis, please use the end-to-end workflow `scripts/batch_process.py`.

Directly segments and scans designated reference genomic regions without read alignment or coverage pre-screening.

Common Arguments:

* `--reference`: Path to reference genome FASTA (must be indexed or indexable via `samtools faidx`).
* `--output_dir`: Output directory.
* `--model_path`: Path to model checkpoint. Default: `models/6.pth`.
* `--limit`: Probability cutoff. Default: `0.99`.
* `--region`: Genomic coordinate or chromosome name (e.g., `chr21` or `chr21:10000000-20000000`). Repeatable.
* `--region_file`: Path to a BED file containing regions to scan.
* `--segment_length`: Slice length in bp for streaming memory chunking. Default: `1000000`.

```bash
python benchmark/detect/run_microdna_map_direct.py \
    --reference refs/hg19.fa \
    --output_dir results/direct_scan \
    --region chr21 --region chr22 \
    --limit 0.90

```

### 3. End-to-End Analysis Pipeline (`scripts/batch_process.py`)

Processes raw paired-end FASTQ data through read alignment, candidate enrichment pre-screening, and sliding-window neural network classification.

Our project-developed `micro_coverage` algorithm serves as the default backend, specifically designed for the physical size of microDNA (150–1000 bp) by capturing local micro-enrichment bins and clustering them. The `cnvkit` pipeline remains available as an optional comparative baseline.

Common Arguments:

* `--input-dir`: Directory containing paired-end FASTQ files.
* `--pipeline`: Pre-screening algorithm backend. Options: `micro_coverage` (project-developed micro-scale local coverage scanner, default) or `cnvkit` (traditional macro-CNV calling).
* `--model`: Path to model checkpoint file or fold directory (`models/cv`) for ensemble voting.
* `--threads`: Number of CPU threads for alignment and operations.
* `--limit`: Probability threshold for positive microDNA calls. Default: `0.75`.
* `--window-size`: Window size (bp) for `micro_coverage` scanning. Default: `200`.
* `--fold-change`: Fold-change threshold relative to chromosome median depth for `micro_coverage`. Default: `1.3`.
* `--cluster-max-len`: Maximum span allowed when clustering adjacent enriched micro-bins (bp). Default: `5000`.
* `--min-cnv-size` / `--max-cnv-size`: Candidate size boundaries.
* `--cleanup`: Automatically remove intermediate SAM and temporary files.
* `--keep-bam`: Retain sorted BAM files when `--cleanup` is active.
* `--output-dir`: Directory for final BED and FASTA outputs.

```bash
# Default: Self-developed micro_coverage pipeline
python scripts/batch_process.py --input-dir data/raw --model models/cv --threads 16 --cleanup --keep-bam

# Optional: Traditional CNVkit pipeline
python scripts/batch_process.py --input-dir data/raw --pipeline cnvkit --threads 16

```

---

## Developer Guide

### 1. Data Preprocessing

Extract and prepare standardized datasets from coordinate spreadsheets and reference backgrounds.

`scripts/process_data.py` (Excel Coordinate Processing):

* `--input`: Path to input Excel table (`.xlsx`).
* `--label`: Class label. Options: `ecc` (maps to `eccDNA`) or `other` (maps to `otherDNA`).
* `--mode`: Length adjustment strategy. Options: `expand` (extend flanks outward), `middle` (center segment of large spans), `raw` (keep original). Default: `expand`.
* `--target-len`: Target sequence length in bp. Default: `400`.
* `--min-span`: Minimum span threshold for `middle` mode. Default: `600`.
* `--ffill-chrom`: Forward-fill chromosome values for merged cells.

`scripts/sample_negatives.py` (Background Sampling):

* `--ratio`: Target negative-to-positive ratio (`negatives / positives`).
* `--count`: Fixed count of negative sequences to sample if `--ratio` is not set. Default: `100000`.
* `--ref`: Reference genome FASTA.
* `--target-len`: Sampled sequence length in bp. Default: `400`.

```bash
# Extract known positive and negative samples
python scripts/process_data.py --input data/raw/eccDNA_annotations.xlsx --label ecc --mode expand
python scripts/process_data.py --input data/raw/otherDNA_annotations.xlsx --label other --mode middle --min-span 600

# Sample random genomic background negatives
python scripts/sample_negatives.py --ratio 1.2 --ref refs/hg19.fa

```

### 2. Model Training (`scripts/train.py`)

Trains the ResNet-SelfAttention classifier using balanced chromosome-grouped 5-fold cross-validation, multi-round online Hard Negative Mining (HNM), and Automatic Mixed Precision (AMP).

Common Arguments:

* `--cv`: Run the automated balanced chromosome 5-fold cross-validation pipeline across all 24 human chromosomes.
* `--fold`: Train only a specific chromosome fold (1 to 5).
* `--split-mode`: Splitting strategy. Options: `chromosome` (balanced chromosome isolation, default) or `random` (random sequence split).
* `--base-epochs`: Training epochs for the initial base stage. Default: `30`.
* `--hnm-rounds`: Iterative HNM rounds (0 disables HNM). Default: `1`.
* `--hnm-epochs`: Additional training epochs for each HNM round. Default: `20`.
* `--hnm-threshold`: Probability cutoff for flagging false positives as hard negatives. Default: `0.1`.
* `--hnm-keep-easy`: Ratio of easy negatives retained to prevent catastrophic forgetting. Default: `0.1`.
* `--batch-size`: Batch size. Default: `256`.
* `--lr`: Initial learning rate. Default: `0.001`.
* `--layer-size`: ResNet base channel width. Default: `8`.
* `--balanced`: Enable weighted random sampling for class balancing.
* `--output-dir`: Output directory for checkpoints and metrics.

```bash
# 1. Automated balanced chromosome-grouped 5-fold cross-validation (Recommended)
python scripts/train.py \
    --cv \
    --base-epochs 30 \
    --hnm-rounds 1 \
    --hnm-epochs 15 \
    --layer-size 8 \
    --balanced \
    --output-dir models/cv

# 2. Train a single specific chromosome fold (e.g., Fold 1)
python scripts/train.py \
    --fold 1 \
    --base-epochs 30 \
    --hnm-rounds 1 \
    --hnm-epochs 15 \
    --layer-size 8 \
    --balanced \
    --output-dir models/fold_1

# 3. Traditional random-split training
python scripts/train.py \
    --split-mode random \
    --base-epochs 30 \
    --hnm-rounds 1 \
    --hnm-epochs 15 \
    --layer-size 8 \
    --balanced \
    --output-dir models/run_random

```

### 3. Model Evaluation (`scripts/verify.py`)

Quantitatively evaluates model predictions on unseen held-out chromosome sets or full datasets, exporting performance reports and confusion matrix plots.

Common Arguments:

* `--model`: Path to model weights (`.pth`) or directory containing CV fold models (`models/cv`).
* `--fold`: Evaluate exclusively on the held-out test chromosomes of the specified fold (1 to 5).
* `--threshold`: Decision threshold. Default: `0.5`.
* `--output-dir`: Directory for evaluation reports and figures.
* `--no-plot`: Suppress figure generation and export metrics text only.

```bash
# Evaluate a single fold model exclusively on its unseen test chromosomes
python scripts/verify.py \
    --model models/cv/fold1_best_model.pth \
    --fold 1 \
    --threshold 0.5 \
    --output-dir results/metrics/fold1

# Evaluate full dataset using 5-fold soft ensemble
python scripts/verify.py \
    --model models/cv \
    --threshold 0.5 \
    --output-dir results/metrics/ensemble

```

### 4. Hard Negative Mining Analysis (`scripts/compare_hnm.py`)

Compares prediction probability distributions between baseline and HNM models to evaluate false positive suppression and tail polarization.

Common Arguments:

* `--task`: Execution mode. Options: `infer` (generate predictions CSV), `plot` (render KDE figures from CSV), `all` (run both). Default: `all`.
* `--model-base`: Baseline model weights (without HNM).
* `--model-hnm`: HNM-trained model weights.
* `--csv-base` / `--csv-hnm`: Input CSV files when running in `plot` mode.
* `--output-dir`: Output directory for generated CSVs and plots.

```bash
# Full execution
python scripts/compare_hnm.py \
    --task all \
    --model-base models/base.pth \
    --model-hnm models/best_model.pth

```

---

## Benchmark

The benchmark directory (`benchmark/`) provides end-to-end evaluation, comparative baselines, and feature dependency verification:

```text
benchmark/
├── config_real.yaml       # Real-data spike-in simulation configuration
├── config_random.yaml     # Random-data spike-in simulation configuration
├── run_benchmark.py       # Orchestrated 10-phase benchmark pipeline
├── microdna_junction_rescuer.py # Head-to-tail junction read rescuer
├── perturbation_test.py   # Sequence feature perturbation and mutagenesis testing
├── models_comparison/     # Comparative baselines (ResNet50, Transformer)
├── simulate/              # Circle-seq and WGS in silico sequencing simulators
├── detect/                # Detection wrappers (MicroDNA Map, Circle-Map)
├── evaluate/              # Overlap resolution and metrics calculation
└── visualize/             # Benchmark visualization tools

```

### 1. Channel Capacity Ablation (`scripts/ablation_layer_size.py`)

Tests different base channel configurations to determine optimal network capacity.

Common Arguments:

* `--sizes`: List of base layer channel widths to evaluate. Default: `2 4 8 16 32 64 128 256 512`.
* `--epochs`: Epochs per training run. Default: `40`.
* `--batch-size`: Batch size. Default: `256`.
* `--balanced`: Enable class-balanced sampling.
* `--dry-run`: Preview parameter matrix without training.

```bash
python scripts/ablation_layer_size.py --sizes 8 16 32 64 --epochs 40 --balanced

```

### 2. Comparative Model Validation (`benchmark/models_comparison/train_eval.py`)

Evaluates comparative baseline network architectures under strictly controlled data splits and seeds.

Common Arguments:

* `--model`: Target architecture. Options: `resnet50` (half-width ~6M params), `resnet_no_att` (ablation model without Self-Attention), `transformer` (depth=8, embed=256).
* `--epochs`: Training epochs. Default: `30`.
* `--batch-size`: Batch size. Default: `256`.

```bash
python benchmark/models_comparison/train_eval.py --model resnet50 --epochs 30
python benchmark/models_comparison/train_eval.py --model resnet_no_att --epochs 30
python benchmark/models_comparison/train_eval.py --model transformer --epochs 30

```

### 3. Sequence Feature Perturbation Test (`benchmark/perturbation_test.py`)

Assesses whether model predictions rely on structural sequence motifs versus overall nucleotide composition via constant-GC sequence shuffling and in silico saturation mutagenesis.

Common Arguments:

* `--task`: Operational mode. Options: `infer`, `plot`, `all`. Default: `all`.
* `--model`: Path to model checkpoint.
* `--input`: Path to positive FASTA dataset.
* `--shuffling-n`: Number of sequences for GC-preserved shuffling test. Default: `5000`.
* `--mutagenesis-n`: Number of sequences for saturation mutagenesis. Default: `500`.

```bash
python benchmark/perturbation_test.py \
    --task all \
    --model models/best_model.pth \
    --input data/processed/eccDNA.fa \
    --output-dir benchmark/results/perturbation

```

### 4. Spike-in Simulation Benchmark (`benchmark/run_benchmark.py`)

Executes an automated, 10-phase benchmark pipeline covering in silico truth generation, Circle-seq/WGS simulation, multi-tool detection, and performance evaluation across copy numbers.

Pipeline Phases:

* `Phase 0`: Environment and index integrity check (`.fai`, BWA, Bowtie2).
* `Phase 1`: MicroDNA truth set generation (`generate_microdna.py`).
* `Phase 2`: Circle-seq read simulation (`simulate_circseq.py`).
* `Phase 3`: WGS read simulation with genomic background (`simulate_wgs.py`).
* `Phase 4`: Circle-Map detection on Circle-seq data.
* `Phase 5`: MicroDNA Map detection on WGS data.
* `Phase 6`: Circle-Map detection on WGS data (negative control).
* `Phase 7`: MicroDNA Map detection on Circle-seq data (cross-modality).
* `Phase 8`: Performance evaluation (Recall, Precision, F1, boundary error).
* `Phase 9`: Visualization (LOD curves, comparison bar plots) and report generation.

Common Arguments:

* `--config`: Path to benchmark configuration YAML. Default: `benchmark/config_real.yaml`.
* `--phase`: Comma-separated list of phase numbers to run (e.g., `0,1,2,5,7,8`). Runs all phases 0–9 if omitted.
* `--quick`: Fast test mode (restricts detection to chromosomes `chr21` and `chr22`).
* `--dry-run`: Print commands without executing.

```bash
# Quick test on selected phases
python benchmark/run_benchmark.py --config benchmark/config_real.yaml --quick --phase 1,2,5,7,8

# Complete full pipeline execution
python benchmark/run_benchmark.py --config benchmark/config_real.yaml

```

### 5. Junction Read Rescuer (`benchmark/microdna_junction_rescuer.py`)

Extracts circular head-to-tail junction split-reads from alignment BAMs and calculates statistical enrichment in target candidate ROIs against matched control regions.

Common Arguments:

* `-b, --bam`: Coordinate-sorted and indexed BAM file.
* `-t, --target-bed`: Candidate microDNA regions in BED format.
* `-r, --reference`: Reference genome FASTA (requires `.fai` index).
* `-o, --out-prefix`: Prefix for output files. Default: `rescued_microdna`.
* `--min-mapq`: Minimum MAPQ for alignments. Default: `20`.
* `--min-circle-len` / `--max-circle-len`: Length limits for circular DNA rescue (bp). Defaults: `150` / `3000`.
* `--depth-tolerance`: Sequencing depth tolerance ratio for matched controls. Default: `0.35`.

```bash
python benchmark/microdna_junction_rescuer.py \
    --bam results/detect/microdna_map/cn10/aligned.sorted.bam \
    --target-bed results/detect/microdna_map/cn10/detected_microdna.bed \
    --reference refs/hg19.fa \
    --out-prefix benchmark/results/rescue_cn10

```