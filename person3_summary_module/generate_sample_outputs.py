"""
generate_sample_outputs.py
==========================
Generates all sample outputs (embeddings, summaries, graphs, plots)
without requiring a trained checkpoint or uploaded files.

Run from the project root:
    python generate_sample_outputs.py

Outputs written to ./sample_outputs/
"""

from __future__ import annotations

import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from training_pipeline import CRGNNConfig, CRGNNSystem, run_inference
from evaluation_scripts.visualizations import (
    plot_causal_graph_matplotlib,
    plot_causal_graph_plotly,
    plot_latent_scatter,
    plot_perspective_salience,
    plot_emotion_trajectory,
    plot_vad_3d,
    plot_salience_heatmap,
    plot_causal_affinity_heatmap,
    build_summary_figure,
)
from evaluation_scripts.metrics import (
    perspective_divergence, emotion_consistency,
    latent_space_quality, graph_alignment_score, format_results
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "sample_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Demo subtitle ─────────────────────────────────────────────────────────────
DEMO_SRT = """1
00:00:01,000 --> 00:00:05,000
Detective Sarah Hayes enters the dimly-lit evidence room.

2
00:00:06,000 --> 00:00:10,000
A shadowy figure — the informant — stands near the window.

3
00:00:11,000 --> 00:00:15,000
She finds a blood-stained envelope hidden behind the cabinet.

4
00:00:16,000 --> 00:00:20,000
The informant smiles coldly. "You're too late," he whispers.

5
00:00:21,000 --> 00:00:26,000
Hayes arrests him — but the real mastermind is still at large.
"""

# ── Demo emotion vectors ──────────────────────────────────────────────────────
DEMO_EMOTIONS = np.array([
    [0.1, 0.2, 0.1, 0.5, 0.3, 0.0, 0.4, 0.6],
    [0.0, 0.3, 0.2, 0.7, 0.4, 0.1, 0.2, 0.3],
    [0.0, 0.4, 0.3, 0.6, 0.8, 0.2, 0.1, 0.2],
    [0.0, 0.2, 0.8, 0.4, 0.1, 0.6, 0.0, 0.1],
    [0.3, 0.5, 0.4, 0.2, 0.3, 0.1, 0.5, 0.4],
], dtype=np.float32)

EMOTION_LABELS = ["Joy","Sadness","Anger","Fear","Surprise","Disgust","Trust","Anticipation"]

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("Building CRGNN system (small config for speed)…")
    cfg = CRGNNConfig(
        vocab_size=1024, d_model=64, d_hidden=64, d_latent=128,
        d_emotion=8, d_code=32, d_arc=32, d_persp=64,
        d_emb_dec=32, d_hidden_dec=128,
        max_seq_len=64, max_decode_len=24,
        num_gat_layers=2, gat_heads=4,
        use_vae=True, causal_threshold=0.4,
    )
    system = CRGNNSystem(cfg)
    system.eval()

    print("Running inference…")
    emotion_tensor = torch.tensor(DEMO_EMOTIONS.tolist(), dtype=torch.float32)
    result = run_inference(system, DEMO_SRT, emotion_tensor, device="cpu")

    if "error" in result:
        print(f"ERROR: {result['error']}")
        return

    scenes = result["scenes"]
    scene_texts = [s["text"] for s in scenes]
    N = len(scenes)
    emo_np = result["emotion_vecs"].numpy()
    vad_np = result["vad"].numpy()
    causal_aff = result["causal_affinity"].numpy()
    node_sal = result["node_salience"].numpy()
    z_dict = result["z_dict"]
    sal_dict = result["sal_dict"]

    # ── 1. Summaries JSON ─────────────────────────────────────────────────────
    summaries = result["summaries"]
    out_path = os.path.join(OUT_DIR, "perspective_summaries.json")
    with open(out_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"  ✓ {out_path}")

    for name, text in summaries.items():
        print(f"\n  [{name.upper()}]\n  {text[:200]}…")

    # ── 2. Latent embeddings JSON ─────────────────────────────────────────────
    embs = {k: v.numpy().tolist() for k, v in z_dict.items()}
    embs["z_global"] = result["z"].numpy().tolist()
    out_path = os.path.join(OUT_DIR, "latent_embeddings.json")
    with open(out_path, "w") as f:
        json.dump(embs, f, indent=2)
    print(f"\n  ✓ {out_path}")

    # ── 3. Causal graph PNG ───────────────────────────────────────────────────
    mask = causal_aff >= 0.4
    src_a, tgt_a = np.where(mask)
    valid = src_a != tgt_a
    src_a, tgt_a = src_a[valid], tgt_a[valid]
    t_src = np.arange(N - 1)
    t_tgt = t_src + 1
    ei_np = np.concatenate([np.array([t_src, t_tgt]),
                             np.array([src_a, tgt_a])], axis=1)
    et_np = np.concatenate([np.zeros(N-1, dtype=int),
                             np.ones(len(src_a), dtype=int)])
    ew_np = np.ones(ei_np.shape[1])

    fig_graph = plot_causal_graph_matplotlib(
        ei_np, et_np, ew_np, scene_texts, node_sal,
        title="Sample Causal Narrative Graph"
    )
    out_path = os.path.join(OUT_DIR, "causal_graph.png")
    fig_graph.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig_graph)
    print(f"  ✓ {out_path}")

    # Interactive HTML graph
    fig_plotly = plot_causal_graph_plotly(
        ei_np, et_np, ew_np, scene_texts, node_sal)
    out_path = os.path.join(OUT_DIR, "causal_graph.html")
    fig_plotly.write_html(out_path)
    print(f"  ✓ {out_path}")

    # ── 4. Causal affinity heatmap ────────────────────────────────────────────
    fig_heat = plot_causal_affinity_heatmap(
        causal_aff, [f"S{i+1}" for i in range(N)])
    out_path = os.path.join(OUT_DIR, "causal_affinity_heatmap.html")
    fig_heat.write_html(out_path)
    print(f"  ✓ {out_path}")

    # ── 5. Latent scatter ─────────────────────────────────────────────────────
    fig_scatter = plot_latent_scatter(z_dict)
    out_path = os.path.join(OUT_DIR, "latent_scatter.html")
    fig_scatter.write_html(out_path)
    print(f"  ✓ {out_path}")

    # ── 6. Perspective salience ───────────────────────────────────────────────
    fig_sal_bar = plot_perspective_salience(sal_dict)
    out_path = os.path.join(OUT_DIR, "perspective_salience.html")
    fig_sal_bar.write_html(out_path)
    print(f"  ✓ {out_path}")

    # ── 7. Salience heatmap PNG ───────────────────────────────────────────────
    fig_sal_h = plot_salience_heatmap(node_sal, scene_texts)
    out_path = os.path.join(OUT_DIR, "salience_heatmap.png")
    fig_sal_h.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig_sal_h)
    print(f"  ✓ {out_path}")

    # ── 8. Emotion trajectory ─────────────────────────────────────────────────
    fig_emo = plot_emotion_trajectory(
        emo_np, scene_ids=list(range(1, N+1)),
        emotion_labels=EMOTION_LABELS
    )
    out_path = os.path.join(OUT_DIR, "emotion_trajectory.html")
    fig_emo.write_html(out_path)
    print(f"  ✓ {out_path}")

    # ── 9. VAD 3D ─────────────────────────────────────────────────────────────
    fig_vad = plot_vad_3d(vad_np)
    out_path = os.path.join(OUT_DIR, "vad_3d.html")
    fig_vad.write_html(out_path)
    print(f"  ✓ {out_path}")

    # ── 10. Dashboard ─────────────────────────────────────────────────────────
    fig_dash = build_summary_figure(emo_np, z_dict, sal_dict, vad_np)
    out_path = os.path.join(OUT_DIR, "dashboard.html")
    fig_dash.write_html(out_path)
    print(f"  ✓ {out_path}")

    # ── 11. Evaluation metrics JSON ───────────────────────────────────────────
    pred_aff_t = torch.tensor(causal_aff.tolist(), dtype=torch.float32)
    pseudo_t   = torch.tensor((causal_aff >= 0.5.tolist(), dtype=torch.float32).astype(float))
    metrics = {
        "graph":       graph_alignment_score(pred_aff_t, pseudo_t),
        "perspective": perspective_divergence(z_dict),
        "emotion":     emotion_consistency(torch.tensor(vad_np.tolist(), dtype=torch.float32)),
        "latent":      latent_space_quality([result["z"]]),
    }
    out_path = os.path.join(OUT_DIR, "evaluation_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ {out_path}")
    print(f"\n{format_results(metrics)}")

    print(f"\n{'='*50}")
    print(f"  All sample outputs written to ./{OUT_DIR}/")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
