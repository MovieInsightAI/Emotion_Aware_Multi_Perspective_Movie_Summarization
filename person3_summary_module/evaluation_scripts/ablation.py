"""
evaluation_scripts/ablation.py  (AAAI/TNNLS v2)
=================================================
Standalone ablation study runner — mandatory for AAAI submission.

Ablation conditions (7 required conditions):
  1. full_model       — complete CRGNN (baseline)
  2. no_emotion       — remove FiLM + cross-attention conditioning
  3. no_causal        — remove L_causal and L_cf
  4. no_projection    — single latent Z, no perspective subspaces
  5. no_mi            — orthogonality only (no MINE / NT-Xent)
  6. no_cf            — no counterfactual masking loss
  7. no_gnn           — GAT replaced by mean pooling (num_gat_layers=0)

Usage
-----
    # Quick test (5 epochs per condition):
    python evaluation_scripts/ablation.py --epochs 5

    # Paper-quality (50 epochs, GPU):
    python evaluation_scripts/ablation.py \\
        --subtitle_file data/my_film.srt \\
        --epochs 50 \\
        --device cuda \\
        --output_dir ablation_results/

Output
------
  ablation_results/
    ablation_results.json     ← all numeric results
    ablation_table.tex        ← LaTeX table for AAAI paper
    checkpoints/              ← per-condition model checkpoints

Console: ASCII performance table with Δ vs. full model
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

# Add parent directory so cross-module imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from training_pipeline import CRGNNConfig, CRGNNSystem, Trainer, AblationStudy
from event_graph_builder import NarrativeGraphBuilder, GraphStore
from evaluation_scripts.metrics import format_ablation_table

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")


# ── Ablation condition definitions (mirrors CRGNNConfig flags) ────────────────
ABLATION_CONFIGS: Dict[str, Dict] = {
    "full_model":    {},                                          # 1. baseline
    "no_emotion":    {"use_emotion": False},                      # 2. no FiLM/cross-attn
    "no_causal":     {"use_causal_loss": False,                   # 3. no causal objective
                      "use_cf_loss": False},
    "no_projection": {"use_projection": False},                   # 4. single Z, no heads
    "no_mi":         {"use_mi_loss": False},                      # 5. orth only, no MINE/NTXent
    "no_cf":         {"use_cf_loss": False},                      # 6. no counterfactual loss
    "no_gnn":        {"num_gat_layers": 0},                       # 7. no graph structure
}

# ── Built-in sample subtitle (10-scene film excerpt) ─────────────────────────
SAMPLE_SUBTITLE = "\n\n".join([
    f"{i}\n00:00:{i:02d},000 --> 00:00:{i+1:02d},000\n" + text
    for i, text in enumerate([
        "The detective enters the rain-soaked room.",
        "A shadowy figure stands near the shattered window.",
        "She confronts him with the crumpled envelope.",
        "He denies all knowledge of the disappearance.",
        "The inspector discovers a hidden door behind the bookcase.",
        "An informant whispers a name in the crowded market.",
        "She races through the alley as sirens wail overhead.",
        "The safe is empty — the documents are gone.",
        "A final message is left on the mirror in red ink.",
        "Hayes makes the arrest as dawn breaks over the city.",
    ], start=1)
])


def build_graph_store(
    subtitle_raw: str,
    d_emotion: int,
    cfg: CRGNNConfig,
    device: str = "cpu",
) -> Tuple[GraphStore, object]:
    """
    Parse subtitle → scene embeddings → narrative graph → GraphStore.

    Returns (graph_store, preprocessor) so the ablation runner can pass
    the preprocessor to AblationStudy.fit().
    """
    from subtitle_preprocessing import SubtitlePreprocessor

    proc = SubtitlePreprocessor(cfg.vocab_size, cfg.d_model, cfg.max_seq_len)
    scenes, token_ids, padding_mask = proc.process(subtitle_raw)
    N = len(scenes)
    if N == 0:
        raise ValueError("No scenes parsed. Check subtitle format (SRT or plain text).")
    logger.info("Parsed %d scenes from subtitle.", N)

    proc.embedding_module.eval()
    with torch.no_grad():
        scene_embs = proc.embedding_module(token_ids, padding_mask)

    # Random emotion vectors (replace with real VAD predictions in production)
    emotion_tensor = torch.rand(N, d_emotion)

    builder = NarrativeGraphBuilder(cfg.d_model, d_emotion, cfg.causal_threshold)
    builder.to(device).eval()
    with torch.no_grad():
        graph, causal_aff, pseudo_lbl = builder.build_graph(
            scene_embs, emotion_tensor, device)

    gs = GraphStore()
    gs.add("doc1", graph, causal_aff, pseudo_lbl)
    return gs, proc


def run_ablation(
    subtitle_raw: str,
    epochs: int,
    output_dir: str,
    device: str = "cpu",
) -> Dict[str, Dict]:
    """
    Run all 7 ablation conditions and return the results table.

    Parameters
    ----------
    subtitle_raw : str    Raw subtitle string (SRT or plain text)
    epochs       : int    Training epochs per condition
    output_dir   : str    Directory for JSON/LaTeX outputs
    device       : str    'cpu' or 'cuda'

    Returns
    -------
    results : {condition_name → metrics_dict}
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    d_emotion = 8

    base_cfg = CRGNNConfig(
        vocab_size=4096, d_model=128, d_hidden=128, d_latent=256,
        d_code=64, d_arc=64, d_persp=128, max_seq_len=128,
        num_gat_layers=3, gat_heads=4,
        max_epochs=epochs, patience=epochs,
        checkpoint_dir=str(Path(output_dir) / "checkpoints"),
    )

    logger.info("Building narrative graph store...")
    gs, proc = build_graph_store(subtitle_raw, d_emotion, base_cfg, device)

    # Inject our named configs into AblationStudy
    study = AblationStudy(base_cfg, gs, proc, device, epochs=epochs)
    study.ABLATIONS = ABLATION_CONFIGS

    logger.info("Starting ablation: %d conditions × %d epochs each.",
                len(ABLATION_CONFIGS), epochs)
    results = study.run()

    # ── Console table ────────────────────────────────────────────────────────
    _print_performance_table(results)

    # ── JSON results ─────────────────────────────────────────────────────────
    json_path = Path(output_dir) / "ablation_results.json"
    serializable = {
        name: {k: v for k, v in r.items() if k != "history"}
        for name, r in results.items()
    }
    json_path.write_text(json.dumps(serializable, indent=2))
    logger.info("JSON results → %s", json_path)

    # ── LaTeX table ───────────────────────────────────────────────────────────
    tex_path = Path(output_dir) / "ablation_table.tex"
    tex_path.write_text(format_ablation_table(results))
    logger.info("LaTeX table → %s", tex_path)

    return results


def _print_performance_table(results: Dict[str, Dict]) -> None:
    """
    Print AAAI-style ablation table to stdout.

    Shows absolute metrics and Δ vs. full_model for each condition.
    Positive Δ loss = worse than full model (expected for ablations).
    """
    full = results.get("full_model", results.get("full", {}))
    full_loss = full.get("final_loss", 0.0)
    full_f1   = full.get("graph_f1", 0.0)
    full_div  = full.get("perspective_divergence", 0.0)

    sep = "=" * 78
    print(f"\n{sep}")
    print("CRGNN Ablation Study Results")
    print(f"{sep}")
    print(f"{'Condition':<22} {'L_total ↓':>10} {'Δ loss':>8}  "
          f"{'Graph F1 ↑':>10} {'Persp Div ↑':>12}")
    print("-" * 78)

    for name, r in results.items():
        loss = r.get("final_loss", 0.0)
        f1   = r.get("graph_f1", 0.0)
        div  = r.get("perspective_divergence", 0.0)
        delta = loss - full_loss
        tag = " ← full" if name in ("full_model", "full") else ""
        print(f"{name:<22} {loss:>10.4f} {delta:>+8.4f}  "
              f"{f1:>10.4f} {div:>12.4f}{tag}")

    print(sep)
    print("Δ loss > 0 → ablation hurts performance (expected).")
    print("For paper: run ≥20 epochs per condition with real subtitle data.\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CRGNN Ablation Study — AAAI/TNNLS submission tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--subtitle_file", type=str, default=None,
        help="Path to subtitle .srt or plain-text file. "
             "Omit to use the built-in 10-scene sample.")
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="Training epochs per ablation condition. "
             "Use 20–50 for paper-quality results.")
    parser.add_argument(
        "--output_dir", type=str, default="ablation_results",
        help="Output directory for JSON and LaTeX files.")
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="Compute device.")
    args = parser.parse_args()

    if args.subtitle_file:
        path = Path(args.subtitle_file)
        if not path.exists():
            logger.error("Subtitle file not found: %s", path)
            sys.exit(1)
        subtitle_raw = path.read_text(encoding="utf-8")
        logger.info("Loaded subtitle: %s", path)
    else:
        subtitle_raw = SAMPLE_SUBTITLE
        logger.info("No subtitle_file provided — using built-in 10-scene sample.")

    run_ablation(
        subtitle_raw=subtitle_raw,
        epochs=args.epochs,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
