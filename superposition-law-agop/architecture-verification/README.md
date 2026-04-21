# Architecture Verification

This folder contains a standalone verification pipeline for testing whether
open-weight language model families follow a power-law relation between loss and
AOFE-style coupling metrics.

It computes two metrics for each model:

- `agop_aofe_ratio`: projected output-space AGOP off-diagonal energy ratio
- `wtw_aofe_ratio`: off-diagonal energy ratio of the input-embedding Gram matrix

Both metrics are paired with a held-out WikiText cross-entropy estimate so you
can test relations such as:

- `loss ≈ a * (agop_aofe_ratio)^b`
- `loss ≈ a * (wtw_aofe_ratio)^b`

## Files

- `model_registry.py`: default Pythia and LLaMA model list
- `run_open_lm_verification.py`: batch runner, CSV writer, plotter, power-law fitter
- `requirements.txt`: extra Python dependencies for this folder
- `results/`: generated CSV, figures, and fit summaries

## Setup

```bash
cd /Users/wenjie1835/Documents/AOFE_Law/superposition-law-agop
python3 -m venv .venv
source .venv/bin/activate
pip install -r architecture-verification/requirements.txt
```

If you want to evaluate official Meta LLaMA checkpoints, make sure your Hugging
Face account has access first and run:

```bash
huggingface-cli login
```

## Default run

```bash
python architecture-verification/run_open_lm_verification.py \
  --device cuda \
  --skip_existing
```

This runs the default registry:

- Pythia: `70m, 160m, 410m, 1b, 1.4b, 2.8b, 6.9b, 12b`
- LLaMA: `3.2-1b, 3.2-3b, 3.1-8b`

## Faster smoke test

```bash
python architecture-verification/run_open_lm_verification.py \
  --models pythia-70m,pythia-160m,llama-3.2-1b \
  --agop_batch_size 2 \
  --agop_n_batches 2 \
  --agop_proj_samples 4 \
  --max_chars 80000 \
  --device cuda
```

## Plot-only regeneration

```bash
python architecture-verification/run_open_lm_verification.py \
  --plot_only \
  --csv_path architecture-verification/results/open_lm_loss_aofe_results.csv
```

## Outputs

The runner writes:

- `results/open_lm_loss_aofe_results.csv`
- `results/all_families_loss_vs_agop_aofe_ratio.png`
- `results/all_families_loss_vs_wtw_aofe_ratio.png`
- family-specific plots for LLaMA and Pythia
- `results/fit_summary.json`

## Practical notes

- `agop_aofe_ratio` is more expensive than `wtw_aofe_ratio` because it requires
  Jacobian-vector products through the full model.
- CUDA is strongly recommended for AGOP estimation.
- The default loss is token cross-entropy on `wikitext/wikitext-2-raw-v1:test`.
- If you want to match your Tiny Transformer setup more closely, switch to
  `--dataset_config wikitext-103-raw-v1`.
