# Superposition Law AGOP

This repository organizes four shape-sweep experiments for studying whether loss is primarily determined by parameter count while model shape is compensated through superposition, quantified here by AOFE/AGOP off-diagonal energy.

## Included experiments

- `cnn`: CIFAR-10 CNN shape sweep with AGOP measured in output space
- `mlp`: supervised PDE operator-learning MLP sweep
- `rnn`: Mackey-Glass sequence modeling sweep
- `transformer`: next-token Transformer sweep

## Project layout

```text
superposition-law-agop/
├── experiments/         # four core experiment scripts
├── results/             # generated outputs
├── data/                # downloaded datasets such as CIFAR-10
├── run_experiment.py    # unified launcher
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Unified usage

List available experiments:

```bash
python3 run_experiment.py list
```

Run one experiment:

```bash
python3 run_experiment.py cnn --device cuda
python3 run_experiment.py mlp --device cuda
python3 run_experiment.py rnn --device cuda
python3 run_experiment.py transformer --device cuda
```

Run all experiments sequentially:

```bash
python3 run_experiment.py all --device cuda
```

Pass extra arguments through to the underlying script with `--`:

```bash
python3 run_experiment.py transformer --device cuda -- --target_params 500000 --eval_every 500
python3 run_experiment.py mlp --device cpu -- --depth_list 3,4,5,6,7,8,9,10,11,12
```

Preview the resolved command without running it:

```bash
python3 run_experiment.py cnn --dry-run --device cuda
```

## Outputs

Each experiment writes into its own directory under `results/`, including:

- `results.csv` and `results.npy`
- AGOP/loss scatter plots
- `curves/` containing per-shape loss-vs-step CSV and PNG files

## Notes for open-sourcing

- The four scripts already enforce 10 shapes per network-task pair.
- Training now tries to stop from a fitted state rather than ending at an arbitrary fixed step.
- For synthetic tasks, the launcher preserves each script's auto-scaling defaults so dataset size and training budget better match the intended `D ≈ 20N` regime.

## Recommended GitHub steps

```bash
git init
git add .
git commit -m "Initial superposition law AGOP experiments"
```
