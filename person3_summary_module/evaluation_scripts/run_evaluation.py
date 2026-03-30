"""
evaluation_scripts/run_evaluation.py
=====================================
Command-line evaluation runner for the CRGNN system.

Usage
-----
    python evaluation_scripts/run_evaluation.py \
        --subtitle_file path/to/subtitles.srt \
        --emotion_file  path/to/emotions.csv \
        --checkpoint    checkpoints/best.pt \
        --output_dir    sample_outputs/

Generates:
  - JSON report of all metrics
  - PNG / HTML visualisations
  - Console summary table
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# Add parent directory to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation_scripts.metrics import (
    compute_rouge_scores,
    compute_bleu_scores,
    graph_alignment_score,
    perspective_divergence,
    emotion_consistency,
    latent_space_quality,
    run_evaluation,
    format_results,
)
from evaluation_scripts.visualizations import (
    plot_causal_graph_plotly,
    plot_causal_graph_matplotlib,
    plot_latent_scatter,
    plot_emotion_trajectory,
    plot_vad_3d,
    plot_salience_heatmap,
    build_summary_figure,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")


# ===========================================================================
# Data Loading Helpers
# ===========================================================================
def load_emotion_csv(path: str) -> np.ndarray:
    """
    Load scene-wise emotion vectors from CSV.
    Expected format: one row per scene, columns = emotion dimensions.
    """
    import csv
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)   # skip header if present
        if header is not None:
            try:
                _ = [float(v) for v in header]
                rows.append([float(v) for v in header])
            except ValueError:
                pass   # actual header row
        for row in reader:
            if row:
                rows.append([float(v) for v in row])
    if not rows:
        logger.warning("Empty emotion file: %s — using zeros.", path)
        return np.zeros((1, 8))
    return np.array(rows, dtype=np.float32)


def load_emotion_npy(path: str) -> np.ndarray:
    return np.load(path).astype(np.float32)


def load_emotion(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return load_emotion_npy(path)
    elif path.endswith(".csv"):
        return load_emotion_csv(path)
    else:
        raise ValueError(f"Unsupported emotion file format: {path}")


# ===========================================================================
# Evaluation Orchestrator
# ===========================================================================
def evaluate(
    subtitle_text: str,
    emotion_array: np.ndarray,
    checkpoint_path: Optional[str],
    output_dir: str,
    perspectives: Optional[List[str]] = None,
    device: str = "cpu",
) -> Dict:
    """
    Run full evaluation pipeline.

    Parameters
    ----------
    subtitle_text  : raw subtitle string
    emotion_array  : (N, d_emo) float32
    checkpoint_path: optional path to trained checkpoint
    output_dir     : directory for output files
    perspectives   : list of perspectives to evaluate

    Returns
    -------
    results : dict of all metrics
    """
    from training_pipeline import CRGNNConfig, CRGNNSystem, run_inference

    os.makedirs(output_dir, exist_ok=True)
    persp = perspectives or ["protagonist", "antagonist", "narrator"]

    # ── Build / load system ─────────────────────────────────────────────────
    cfg = CRGNNConfig()
    system = CRGNNSystem(cfg)

    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        # Partial load (ignore missing decoder keys before decoder init)
        system_state = {k: v for k, v in ckpt["system_state"].items()
                        if not k.startswith("summary_decoder")}
        system.load_state_dict(system_state, strict=False)
        logger.info("Loaded checkpoint: %s", checkpoint_path)
    else:
        logger.warning("No checkpoint provided — using random weights.")

    # ── Run inference ────────────────────────────────────────────────────────
    emotion_tensor = torch.tensor(emotion_array.tolist(), dtype=torch.float32)
    result = run_inference(system, subtitle_text, emotion_tensor,
                           perspectives=persp, device=device)

    if "error" in result:
        logger.error("Inference error: %s", result["error"])
        return result

    scenes = result["scenes"]
    scene_texts = [s["text"] for s in scenes]
    N = len(scenes)

    # ── Compute metrics ──────────────────────────────────────────────────────
    # Prepare hypotheses (generated summaries)
    hypotheses = {name: [result["summaries"][name]]
                  for name in persp if name in result["summaries"]}

    # Graph metrics
    causal_aff = result["causal_affinity"].numpy()
    # Use thresholded affinity as pseudo ground truth (evaluation mode)
    pseudo_true = (causal_aff >= 0.6).astype(float)
    graph_metrics = graph_alignment_score(
        torch.tensor(causal_aff.tolist(), dtype=torch.float32),
        torch.tensor(pseudo_true.tolist(), dtype=torch.float32),
    )

    # Perspective divergence
    persp_div = perspective_divergence(result["z_dict"])

    # Emotion consistency
    vad = result["vad"].numpy()
    emo_metrics = emotion_consistency(
        torch.tensor(vad.tolist(), dtype=torch.float32)
    )

    # Latent quality
    z_list = [result["z"]]
    lat_metrics = latent_space_quality(z_list)

    all_results = {
        "graph": graph_metrics,
        "perspective": persp_div,
        "emotion": emo_metrics,
        "latent": lat_metrics,
    }

    logger.info("\n%s", format_results(all_results))

    # ── Save JSON report ─────────────────────────────────────────────────────
    report_path = Path(output_dir) / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "metrics": all_results,
            "summaries": result["summaries"],
            "num_scenes": N,
        }, f, indent=2)
    logger.info("Report saved → %s", report_path)

    # ── Save visualisations ──────────────────────────────────────────────────
    edge_index = result["causal_affinity"]
    ei = result.get("causal_affinity")

    # Causal graph from affinity (threshold at 0.4)
    aff_np = causal_aff
    thr = 0.4
    mask = aff_np >= thr
    src_arr, tgt_arr = np.where(mask)
    ei_np = np.array([src_arr, tgt_arr])
    et_np = np.ones(len(src_arr), dtype=int)   # all causal
    # Add temporal edges
    if N > 1:
        t_src = np.arange(N - 1)
        t_tgt = t_src + 1
        ei_np = np.concatenate([np.array([t_src, t_tgt]), ei_np], axis=1)
        et_np = np.concatenate([np.zeros(N - 1, dtype=int), et_np])
    ew_np = np.ones(ei_np.shape[1])

    node_sal_np = result["node_salience"].numpy()
    emo_np = result["emotion_vecs"].numpy()

    # Plotly interactive graph
    fig_graph = plot_causal_graph_plotly(
        ei_np, et_np, ew_np, scene_texts, node_sal_np)
    fig_graph.write_html(str(Path(output_dir) / "causal_graph.html"))

    # Matplotlib static graph
    import matplotlib
    matplotlib.use("Agg")
    fig_mat = plot_causal_graph_matplotlib(
        ei_np, et_np, ew_np, scene_texts, node_sal_np)
    fig_mat.savefig(str(Path(output_dir) / "causal_graph.png"),
                    dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig_mat)

    # Latent scatter
    fig_latent = plot_latent_scatter(result["z_dict"])
    fig_latent.write_html(str(Path(output_dir) / "latent_scatter.html"))

    # Emotion trajectory
    fig_emo = plot_emotion_trajectory(emo_np)
    fig_emo.write_html(str(Path(output_dir) / "emotion_trajectory.html"))

    # VAD 3D
    fig_vad = plot_vad_3d(vad)
    fig_vad.write_html(str(Path(output_dir) / "vad_3d.html"))

    # Salience heatmap
    fig_sal = plot_salience_heatmap(node_sal_np, scene_texts)
    fig_sal.savefig(str(Path(output_dir) / "salience_heatmap.png"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig_sal)

    # Dashboard
    sal_dict = result["sal_dict"]
    fig_dash = build_summary_figure(emo_np, result["z_dict"], sal_dict, vad)
    fig_dash.write_html(str(Path(output_dir) / "dashboard.html"))

    logger.info("All visualisations saved to %s", output_dir)

    return all_results


# ===========================================================================
# CLI Entry Point
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="CRGNN Evaluation Runner")
    parser.add_argument("--subtitle_file", type=str, required=True,
                        help="Path to subtitle file (.srt or plain text)")
    parser.add_argument("--emotion_file", type=str, required=True,
                        help="Path to emotion vectors (.csv or .npy)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint .pt file")
    parser.add_argument("--output_dir", type=str, default="sample_outputs",
                        help="Directory for output files")
    parser.add_argument("--perspectives", type=str, nargs="+",
                        default=["protagonist", "antagonist", "narrator"],
                        help="Perspectives to evaluate")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu or cuda")
    args = parser.parse_args()

    # Load subtitle
    subtitle_path = Path(args.subtitle_file)
    if not subtitle_path.exists():
        logger.error("Subtitle file not found: %s", subtitle_path)
        sys.exit(1)
    subtitle_text = subtitle_path.read_text(encoding="utf-8")

    # Load emotions
    emotion_path = Path(args.emotion_file)
    if not emotion_path.exists():
        logger.error("Emotion file not found: %s", emotion_path)
        sys.exit(1)
    emotion_array = load_emotion(str(emotion_path))

    evaluate(
        subtitle_text=subtitle_text,
        emotion_array=emotion_array,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        perspectives=args.perspectives,
        device=args.device,
    )


if __name__ == "__main__":
    main()
