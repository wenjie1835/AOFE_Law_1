#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_registry import DEFAULT_MODEL_NAMES, ModelSpec, get_specs


if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = ROOT_DIR / "results"
DEFAULT_CSV_PATH = DEFAULT_OUT_DIR / "open_lm_loss_aofe_results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute projected AGOP AOFE_ratio, embedding WtW AOFE_ratio, and "
            "WikiText loss for Pythia/LLaMA families, then fit power laws."
        )
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODEL_NAMES),
        help="Comma-separated model keys from model_registry.py.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="loss,agop,wtw",
        help="Comma-separated subset of: loss,agop,wtw",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for CSV, figures, and fit summaries.",
    )
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Path to the rolling result CSV.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
        help="Hugging Face dataset name used for evaluation.",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="wikitext-2-raw-v1",
        help="Hugging Face dataset config name.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split used for evaluation.",
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=250_000,
        help="Maximum number of raw characters loaded from the dataset split.",
    )
    parser.add_argument(
        "--loss_max_length",
        type=int,
        default=1024,
        help="Context window for loss evaluation.",
    )
    parser.add_argument(
        "--loss_stride",
        type=int,
        default=512,
        help="Stride used in rolling loss evaluation.",
    )
    parser.add_argument(
        "--agop_seq_len",
        type=int,
        default=128,
        help="Sequence length used when estimating projected AGOP.",
    )
    parser.add_argument(
        "--agop_out",
        type=int,
        default=64,
        help="Output dimension after fixed random projection before AGOP.",
    )
    parser.add_argument(
        "--agop_batch_size",
        type=int,
        default=4,
        help="Batch size for AGOP estimation.",
    )
    parser.add_argument(
        "--agop_n_batches",
        type=int,
        default=4,
        help="Number of random batches used to estimate AGOP.",
    )
    parser.add_argument(
        "--agop_proj_samples",
        type=int,
        default=8,
        help="Number of JVP probes per AGOP batch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Execution device. CUDA is strongly recommended for AGOP.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Torch dtype for model loading.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to tokenizer/model loading.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip models that already exist in the CSV.",
    )
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="Skip model evaluation and regenerate figures from an existing CSV.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def pick_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda")
    if name == "mps":
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def cleanup_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def agop_offdiag_metrics(matrix: torch.Tensor) -> Tuple[float, float]:
    mat = matrix.detach().float()
    fro2 = float((mat * mat).sum().item()) + 1e-12
    diag2 = float((torch.diag(mat) ** 2).sum().item())
    offdiag = max(fro2 - diag2, 0.0)
    return offdiag, offdiag / fro2


def wtw_metrics_from_embedding(weight: torch.Tensor) -> Tuple[float, float]:
    emb = weight.detach().float().cpu()
    gram = (emb.T @ emb) / max(int(emb.shape[0]), 1)
    return agop_offdiag_metrics(gram)


def count_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def load_eval_text(dataset_name: str, dataset_config: str, split: str, max_chars: int) -> str:
    ds = load_dataset(dataset_name, dataset_config, split=split)
    text = "\n\n".join(x for x in ds["text"] if isinstance(x, str) and x.strip())
    if max_chars > 0:
        text = text[:max_chars]
    if not text:
        raise RuntimeError("Loaded dataset text is empty.")
    return text


def tokenize_text(tokenizer, text: str) -> torch.Tensor:
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"][0].to(torch.long).cpu()
    if input_ids.numel() < 4:
        raise RuntimeError("Tokenized dataset is too short.")
    return input_ids


@torch.no_grad()
def evaluate_loss(
    model,
    input_ids: torch.Tensor,
    device: torch.device,
    max_length: int,
    stride: int,
) -> float:
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    max_length = int(max_length)
    stride = int(stride)

    for begin in range(0, input_ids.size(0) - 1, stride):
        end = min(begin + max_length, input_ids.size(0))
        chunk = input_ids[begin:end].unsqueeze(0).to(device)
        targets = chunk.clone()
        if begin > 0:
            overlap = max_length - stride
            if overlap > 0:
                targets[:, :overlap] = -100
        outputs = model(input_ids=chunk, labels=targets, use_cache=False)
        valid = int((targets != -100).sum().item())
        total_nll += float(outputs.loss.item()) * max(valid, 1)
        total_tokens += valid
        if end >= input_ids.size(0):
            break
    return total_nll / max(total_tokens, 1)


def sample_windows(
    token_ids: torch.Tensor,
    seq_len: int,
    batch_size: int,
    n_batches: int,
    seed: int,
) -> List[torch.Tensor]:
    total = int(token_ids.numel())
    max_start = total - seq_len - 1
    if max_start <= 0:
        raise RuntimeError(
            f"Not enough tokens ({total}) for AGOP sequence length {seq_len}."
        )
    rng = np.random.default_rng(seed)
    windows: List[torch.Tensor] = []
    for _ in range(n_batches):
        starts = rng.integers(0, max_start, size=batch_size)
        batch = torch.stack([token_ids[s : s + seq_len] for s in starts], dim=0)
        windows.append(batch.to(torch.long))
    return windows


def estimate_projected_agop(
    model,
    windows: Sequence[torch.Tensor],
    device: torch.device,
    agop_out: int,
    proj_samples: int,
    seed: int,
) -> torch.Tensor:
    model.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    vocab_size = int(model.config.vocab_size)
    projector = torch.randn(
        agop_out,
        vocab_size,
        generator=generator,
        device=device,
        dtype=torch.float32,
    ) / math.sqrt(float(agop_out))

    agop = torch.zeros(agop_out, agop_out, device=device, dtype=torch.float32)
    updates = 0

    for batch_tokens in windows:
        batch_tokens = batch_tokens.to(device)
        inputs_embeds = model.get_input_embeddings()(batch_tokens).detach()

        def fwd(embeds: torch.Tensor) -> torch.Tensor:
            outputs = model(inputs_embeds=embeds, use_cache=False)
            logits = outputs.logits[:, -1, :].float()
            return logits @ projector.T

        for _ in range(proj_samples):
            tangent = torch.randn_like(inputs_embeds)
            _, jvp = torch.autograd.functional.jvp(
                fwd,
                (inputs_embeds,),
                (tangent,),
                create_graph=False,
                strict=False,
            )
            jvp = torch.nan_to_num(jvp.float(), nan=0.0, posinf=0.0, neginf=0.0)
            agop += (jvp.T @ jvp) / float(batch_tokens.size(0))
            updates += 1

    if updates == 0:
        return agop.cpu()
    agop = 0.5 * (agop + agop.T)
    return (agop / float(updates)).cpu()


def pearson(x: Iterable[float], y: Iterable[float]) -> float:
    xa = np.asarray(list(x), dtype=np.float64)
    ya = np.asarray(list(y), dtype=np.float64)
    if xa.size < 2:
        return float("nan")
    xa = xa - xa.mean()
    ya = ya - ya.mean()
    denom = np.sqrt(float((xa * xa).sum()) * float((ya * ya).sum())) + 1e-12
    return float((xa * ya).sum() / denom)


def _rank(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def spearman(x: Iterable[float], y: Iterable[float]) -> float:
    return pearson(_rank(list(x)), _rank(list(y)))


def fit_powerlaw(xs: np.ndarray, ys: np.ndarray) -> Tuple[float, float, float]:
    coef = np.polyfit(np.log(xs), np.log(ys), deg=1)
    exponent = float(coef[0])
    scale = float(np.exp(coef[1]))
    pred = scale * xs**exponent
    ss_res = float(np.square(ys - pred).sum())
    ss_tot = float(np.square(ys - ys.mean()).sum()) + 1e-12
    return scale, exponent, 1.0 - ss_res / ss_tot


def safe_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        return float(value)
    return float(value)


def read_existing_rows(csv_path: Path) -> List[Dict[str, object]]:
    if not csv_path.exists():
        return []
    rows: List[Dict[str, object]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "family": row.get("family", ""),
                    "name": row.get("name", ""),
                    "repo_id": row.get("repo_id", ""),
                    "param_count": safe_float(row.get("param_count")),
                    "loss": safe_float(row.get("loss")),
                    "agop_aofe": safe_float(row.get("agop_aofe")),
                    "agop_aofe_ratio": safe_float(row.get("agop_aofe_ratio")),
                    "wtw_aofe": safe_float(row.get("wtw_aofe")),
                    "wtw_aofe_ratio": safe_float(row.get("wtw_aofe_ratio")),
                    "loss_metric": row.get("loss_metric", ""),
                    "notes": row.get("notes", ""),
                }
            )
    return rows


def write_rows(csv_path: Path, rows: Sequence[Dict[str, object]]) -> None:
    ensure_dir(csv_path.parent)
    fieldnames = [
        "family",
        "name",
        "repo_id",
        "param_count",
        "loss",
        "agop_aofe",
        "agop_aofe_ratio",
        "wtw_aofe",
        "wtw_aofe_ratio",
        "loss_metric",
        "notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def filter_rows(
    rows: Sequence[Dict[str, object]],
    family: Optional[str],
    metric_key: str,
) -> List[Dict[str, object]]:
    filtered: List[Dict[str, object]] = []
    for row in rows:
        if family is not None and row.get("family") != family:
            continue
        metric = row.get(metric_key)
        loss = row.get("loss")
        if metric is None or loss is None:
            continue
        metric_value = float(metric)
        loss_value = float(loss)
        if not math.isfinite(metric_value) or not math.isfinite(loss_value):
            continue
        if metric_value <= 0.0 or loss_value <= 0.0:
            continue
        filtered.append(row)
    return filtered


def plot_powerlaw(
    rows: Sequence[Dict[str, object]],
    family: Optional[str],
    metric_key: str,
    out_path: Path,
    title: str,
    xlabel: str,
) -> Optional[Dict[str, float]]:
    sub = filter_rows(rows, family=family, metric_key=metric_key)
    if len(sub) < 2:
        return None

    xs = np.asarray([float(r[metric_key]) for r in sub], dtype=np.float64)
    ys = np.asarray([float(r["loss"]) for r in sub], dtype=np.float64)
    scale, exponent, r2 = fit_powerlaw(xs, ys)
    fit_x = np.linspace(xs.min() * 0.995, xs.max() * 1.005, 200)
    fit_y = scale * fit_x**exponent

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.scatter(xs, ys, s=80, color="tab:blue")
    ax.plot(
        fit_x,
        fit_y,
        "--",
        color="tab:red",
        lw=2.0,
        label=f"loss = {scale:.4g} * ratio^({exponent:.4f})",
    )
    for row in sub:
        ax.annotate(
            f"{row['family']}:{row['name']}",
            (float(row[metric_key]), float(row["loss"])),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("WikiText cross-entropy (nats/token)")
    ax.set_title(f"linear axes  $R^2$={r2:.3f}")
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.scatter(xs, ys, s=80, color="tab:blue")
    ax.plot(fit_x, fit_y, "--", color="tab:red", lw=2.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"{xlabel} (log)")
    ax.set_ylabel("loss (log)")
    ax.set_title("log-log view")
    ax.grid(True, alpha=0.35, which="both")

    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "n_points": float(len(sub)),
        "pearson": pearson(xs, ys),
        "spearman": spearman(xs, ys),
        "scale": scale,
        "exponent": exponent,
        "r2": r2,
    }


def summarize_to_json(rows: Sequence[Dict[str, object]], out_dir: Path) -> None:
    families = sorted({str(r["family"]) for r in rows})
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for family in [None, *families]:
        family_key = "all_families" if family is None else family
        summary[family_key] = {}
        for metric_key, xlabel in (
            ("agop_aofe_ratio", "Projected AGOP AOFE_ratio"),
            ("wtw_aofe_ratio", "Embedding WtW AOFE_ratio"),
        ):
            plot_name = f"{family_key}_loss_vs_{metric_key}.png"
            stats = plot_powerlaw(
                rows,
                family=family,
                metric_key=metric_key,
                out_path=out_dir / plot_name,
                title=f"{family_key}: loss vs {xlabel}",
                xlabel=xlabel,
            )
            if stats is not None:
                summary[family_key][metric_key] = stats
    with (out_dir / "fit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def load_model_and_tokenizer(
    spec: ModelSpec,
    device: torch.device,
    dtype: torch.dtype,
    trust_remote_code: bool,
):
    tokenizer = AutoTokenizer.from_pretrained(
        spec.repo_id,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        spec.repo_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    model.to(device)
    if getattr(model.config, "use_cache", None) is not None:
        model.config.use_cache = False
    return model, tokenizer


def evaluate_spec(
    spec: ModelSpec,
    raw_text: str,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> Dict[str, object]:
    metrics = {x.strip() for x in args.metrics.split(",") if x.strip()}
    row: Dict[str, object] = {
        "family": spec.family,
        "name": spec.name,
        "repo_id": spec.repo_id,
        "param_count": None,
        "loss": None,
        "agop_aofe": None,
        "agop_aofe_ratio": None,
        "wtw_aofe": None,
        "wtw_aofe_ratio": None,
        "loss_metric": "WikiText cross-entropy (nats/token)",
        "notes": spec.notes,
    }

    model = None
    tokenizer = None
    try:
        model, tokenizer = load_model_and_tokenizer(
            spec=spec,
            device=device,
            dtype=dtype,
            trust_remote_code=args.trust_remote_code,
        )
        row["param_count"] = count_params(model)
        token_ids = tokenize_text(tokenizer, raw_text)

        if "wtw" in metrics:
            wtw_aofe, wtw_ratio = wtw_metrics_from_embedding(
                model.get_input_embeddings().weight
            )
            row["wtw_aofe"] = wtw_aofe
            row["wtw_aofe_ratio"] = wtw_ratio

        if "loss" in metrics:
            row["loss"] = evaluate_loss(
                model=model,
                input_ids=token_ids,
                device=device,
                max_length=args.loss_max_length,
                stride=args.loss_stride,
            )

        if "agop" in metrics:
            spec_seed = args.seed + sum(ord(ch) for ch in spec.repo_id)
            windows = sample_windows(
                token_ids=token_ids,
                seq_len=args.agop_seq_len,
                batch_size=args.agop_batch_size,
                n_batches=args.agop_n_batches,
                seed=spec_seed,
            )
            agop = estimate_projected_agop(
                model=model,
                windows=windows,
                device=device,
                agop_out=args.agop_out,
                proj_samples=args.agop_proj_samples,
                seed=spec_seed + 42,
            )
            agop_aofe, agop_ratio = agop_offdiag_metrics(agop)
            row["agop_aofe"] = agop_aofe
            row["agop_aofe_ratio"] = agop_ratio
    finally:
        del model
        del tokenizer
        cleanup_memory(device)
    return row


def print_result_table(rows: Sequence[Dict[str, object]]) -> None:
    header = (
        f"{'family':<10}  {'name':<12}  {'params':>12}  {'loss':>10}  "
        f"{'agop_ratio':>12}  {'wtw_ratio':>12}"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        params = row["param_count"]
        loss = row["loss"]
        agop = row["agop_aofe_ratio"]
        wtw = row["wtw_aofe_ratio"]
        params_text = "NA" if params is None else f"{int(float(params)):,}"
        loss_text = "NA" if loss is None else f"{float(loss):.6f}"
        agop_text = "NA" if agop is None else f"{float(agop):.6f}"
        wtw_text = "NA" if wtw is None else f"{float(wtw):.6f}"
        print(
            f"{row['family']:<10}  {row['name']:<12}  {params_text:>12}  "
            f"{loss_text:>10}  {agop_text:>12}  {wtw_text:>12}"
        )


def main() -> None:
    args = parse_args()
    if args.csv_path == DEFAULT_CSV_PATH:
        args.csv_path = args.out_dir / DEFAULT_CSV_PATH.name
    ensure_dir(args.out_dir)
    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)
    model_specs = get_specs(args.models.split(","))

    rows = read_existing_rows(args.csv_path)
    existing_names = {str(r["name"]) for r in rows}

    if not args.plot_only:
        raw_text = load_eval_text(
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            split=args.split,
            max_chars=args.max_chars,
        )
        print(f"Using device={device} dtype={dtype}")
        print(
            f"Loaded {len(raw_text):,} raw characters from "
            f"{args.dataset_name}/{args.dataset_config}:{args.split}"
        )
        for spec in model_specs:
            if args.skip_existing and spec.name in existing_names:
                print(f"\nSkipping existing model: {spec.family}:{spec.name}")
                continue
            print(f"\n=== Evaluating {spec.family}:{spec.name} ({spec.repo_id}) ===")
            try:
                row = evaluate_spec(spec, raw_text, device, dtype, args)
            except Exception as exc:
                row = {
                    "family": spec.family,
                    "name": spec.name,
                    "repo_id": spec.repo_id,
                    "param_count": None,
                    "loss": None,
                    "agop_aofe": None,
                    "agop_aofe_ratio": None,
                    "wtw_aofe": None,
                    "wtw_aofe_ratio": None,
                    "loss_metric": "WikiText cross-entropy (nats/token)",
                    "notes": f"{spec.notes} | failed: {exc}",
                }
                print(f"FAILED: {exc}")
            rows = [r for r in rows if str(r["name"]) != spec.name]
            rows.append(row)
            rows.sort(key=lambda item: (str(item["family"]), str(item["name"])))
            write_rows(args.csv_path, rows)

    summarize_to_json(rows, args.out_dir)
    print_result_table(rows)
    print(f"\nSaved CSV to {args.csv_path}")
    print(f"Saved plots and fit summary to {args.out_dir}")


if __name__ == "__main__":
    main()
