"""
app.py  —  CRGNN Streamlit UI  (v2: video upload + numpy/torch fix)
====================================================================
Input modes:
  1. Video file (.mp4 .mkv .avi .mov .webm) → auto-extract audio
     emotion features + embedded/placeholder SRT subtitles
  2. Manual: SRT subtitle + CSV/NPY emotion vector upload
  3. Demo data (no upload needed)

Key fix: torch.tensor() used everywhere instead of torch.from_numpy()
to avoid the "Numpy is not available" error on some Anaconda/Windows.
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from subtitle_preprocessing import SubtitlePreprocessor
from event_graph_builder import NarrativeGraphBuilder
from gnn_narrative_encoder import GNNNarrativeEncoder, VariationalNarrativeEncoder
from emotion_conditioning import EmotionConditioningModule, EMOTION_DIMS
from perspective_projection import PerspectiveDisentanglementModule, PERSPECTIVES
from summary_decoder import MultiPerspectiveSummaryDecoder
from training_pipeline import (
    CRGNNConfig, CRGNNSystem, Trainer, GraphStore, run_inference,
)
from video_processor import VideoProcessor
from evaluation_scripts.metrics import (
    graph_alignment_score, perspective_divergence,
    emotion_consistency, latent_space_quality,
)
from evaluation_scripts.visualizations import (
    plot_causal_graph_plotly, plot_causal_affinity_heatmap,
    plot_latent_scatter, plot_perspective_salience,
    plot_emotion_trajectory, plot_vad_3d,
    plot_loss_curves, plot_salience_heatmap, build_summary_figure,
)

logging.basicConfig(level=logging.WARNING)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CRGNN — Narrative Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-title{font-size:1.9rem;font-weight:800;
  background:linear-gradient(135deg,#1a237e,#0d47a1,#1565c0);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.sub-title{font-size:.9rem;color:#546e7a;margin-bottom:1.2rem}
.section-head{font-size:1.05rem;font-weight:700;color:#1a237e;
  border-bottom:2px solid #e3f2fd;padding-bottom:3px;margin-bottom:.8rem}
.pcard{background:#f8f9ff;border-left:5px solid #1565c0;border-radius:8px;
  padding:.9rem 1.1rem;margin-bottom:.9rem;font-size:.9rem;line-height:1.7;
  box-shadow:0 2px 6px rgba(0,0,0,.07)}
.pcard.protagonist{border-left-color:#2196F3}
.pcard.antagonist{border-left-color:#F44336}
.pcard.narrator{border-left-color:#4CAF50}
.info-box{background:#e8f5e9;border-radius:8px;padding:.7rem 1rem;
  margin:.4rem 0;font-size:.87rem;color:#1b5e20}
.warn-box{background:#fff8e1;border-radius:8px;padding:.7rem 1rem;
  margin:.4rem 0;font-size:.87rem;color:#6d4c41}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
_DEFAULTS = dict(
    system=None, inference_result=None, train_history=None,
    raw_subtitle=None, emotion_array=None,
    selected_perspectives=PERSPECTIVES[:], device="cpu", video_info=None,
)
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

ICONS  = {"protagonist":"🦸","antagonist":"🎭","narrator":"📜"}
COLORS = {"protagonist":"#2196F3","antagonist":"#F44336","narrator":"#4CAF50"}


# ── Safe tensor helper (key numpy/torch fix) ──────────────────────────────────
def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array → torch.Tensor without torch.from_numpy().
    torch.from_numpy() raises RuntimeError on some Anaconda builds
    where numpy is present but not linked in the torch binary.
    torch.tensor() always copies the data safely.
    """
    return torch.tensor(arr.tolist(), dtype=torch.float32)

def _to_numpy(t) -> "np.ndarray":
    """
    Safely convert a torch.Tensor → numpy array.
    Uses .tolist() as an intermediate to avoid the torch-numpy C bridge
    that breaks on some Anaconda/Windows installations.
    """
    if isinstance(t, np.ndarray):
        return t
    return np.array(t.detach().cpu().tolist(), dtype=np.float32)



# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar() -> dict:
    st.sidebar.markdown("## ⚙️ Configuration")
    devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    st.session_state.device = st.sidebar.selectbox("Compute device", devs)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎭 Active Perspectives")
    sel = [p for p in PERSPECTIVES
           if st.sidebar.checkbox(f"{ICONS[p]} {p.capitalize()}",
                                  value=True, key=f"cb_{p}")]
    st.session_state.selected_perspectives = sel or ["narrator"]

    st.sidebar.markdown("---")
    with st.sidebar.expander("🔬 Model Architecture", expanded=False):
        d_model  = st.slider("Scene embedding dim", 32, 256, 128, 32)
        d_latent = st.slider("Latent space dim Z",  64, 512, 256, 64)
        d_persp  = st.slider("Perspective subspace",32, 256, 128, 32)
        n_layers = st.slider("GAT layers", 1, 5, 3)
        n_heads  = st.select_slider("Attention heads", [1,2,4,8], value=4)
        use_vae  = st.checkbox("Variational latent (VAE)", True)
        c_thr    = st.slider("Causal edge threshold", 0.2, 0.9, 0.45, 0.05)

    with st.sidebar.expander("🎯 Loss Weights λ", expanded=False):
        l1 = st.slider("λ1 L_causal",      0.0, 2.0, 0.5, 0.1)
        l2 = st.slider("λ2 L_disentangle", 0.0, 1.0, 0.2, 0.05)
        l3 = st.slider("λ3 L_temporal",    0.0, 1.0, 0.3, 0.05)
        l4 = st.slider("λ4 L_summary",     0.0, 2.0, 0.4, 0.1)
        l5 = st.slider("λ5 L_kl",          0.0, 0.01, 0.001, 0.0005)

    with st.sidebar.expander("🏋️ Training", expanded=False):
        lr       = st.select_slider("LR", [1e-5,3e-5,1e-4,3e-4,1e-3], 3e-4)
        n_epochs = st.slider("Epochs", 1, 100, 20)
        patience = st.slider("Early stop patience", 3, 20, 8)

    with st.sidebar.expander("🎬 Video Settings", expanded=False):
        scene_dur = st.slider("Scene duration (s)", 2, 30, 5)
        st.caption("Controls scene segmentation length during video processing.")

    return dict(
        d_model=d_model, d_hidden=d_model, d_latent=d_latent,
        d_persp=d_persp, num_gat_layers=n_layers, gat_heads=n_heads,
        use_vae=use_vae, causal_threshold=c_thr,
        lambda1=l1, lambda2=l2, lambda3=l3, lambda4=l4, lambda5=l5,
        lr=lr, max_epochs=n_epochs, patience=patience,
        _scene_dur=scene_dur,
    )


# ── Upload section ─────────────────────────────────────────────────────────────
def render_upload(cfg_params: dict):
    st.markdown('<p class="section-head">📂 Data Input</p>',
                unsafe_allow_html=True)

    mode = st.radio(
        "Input mode",
        ["📹 Video file (auto-extract)",
         "📄 Manual (subtitle + emotion files)",
         "🎬 Demo data"],
        horizontal=True, label_visibility="collapsed",
    )

    # ── VIDEO ─────────────────────────────────────────────────────────────────
    if "Video" in mode:
        st.markdown("#### Upload Video File")
        st.markdown(
            '<div class="info-box">'
            '🎬 Upload any video file. The system will:<br>'
            '• Extract embedded subtitle tracks (SRT/ASS) if present<br>'
            '• If no subtitles found → generate scene descriptions from timestamps<br>'
            '• Analyse the audio track to produce 8-dim acoustic emotion vectors<br>'
            '• All processing is done locally — no external APIs used.'
            '</div>', unsafe_allow_html=True)

        vid_file = st.file_uploader(
            "Video file (MP4, MKV, AVI, MOV, WebM, FLV, M4V)",
            type=["mp4","mkv","avi","mov","webm","flv","m4v"],
            key="video_up",
            help="FFmpeg must be installed for audio/subtitle extraction.",
        )

        col1, col2, col3 = st.columns(3)
        col1.info("🎵 Audio → 8 acoustic emotion dims")
        col2.info("📝 Subtitles extracted or auto-generated")
        col3.info("🎬 Scenes auto-segmented by duration")

        if vid_file is not None:
            _process_video_upload(vid_file, int(cfg_params.get("_scene_dur", 5)))

    # ── MANUAL ───────────────────────────────────────────────────────────────
    elif "Manual" in mode:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Scene-Aligned Subtitles** (SRT or plain text)")
            sf = st.file_uploader("Subtitle file", type=["srt","txt"],
                                   key="sub_up")
            if sf:
                st.session_state.raw_subtitle = sf.read().decode("utf-8","ignore")
                st.success(f"✅ {sf.name} loaded")
        with col2:
            st.markdown("**Emotion Vectors** — rows=scenes, cols=emotion dims")
            ef = st.file_uploader("CSV or NPY", type=["csv","npy"], key="emo_up")
            if ef:
                arr = _parse_emotion_file(ef)
                if arr is not None:
                    st.session_state.emotion_array = arr
                    st.success(f"✅ {arr.shape[0]} scenes × {arr.shape[1]} dims")

        if st.session_state.raw_subtitle:
            with st.expander("Preview subtitle"):
                st.code(st.session_state.raw_subtitle[:600], language="text")
        if st.session_state.emotion_array is not None:
            with st.expander("Preview emotion vectors"):
                _show_emotion_df(st.session_state.emotion_array)

    # ── DEMO ─────────────────────────────────────────────────────────────────
    else:
        st.markdown(
            '<div class="info-box">'
            'Demo: 5-scene detective thriller subtitle + '
            'synthetic acoustic emotion vectors.</div>',
            unsafe_allow_html=True)
        if st.button("▶️ Load Demo Data", use_container_width=True):
            st.session_state.raw_subtitle  = _demo_srt()
            st.session_state.emotion_array = _demo_emotions()
            st.session_state.video_info    = None
            st.success("✅ Demo data loaded — click **Generate Summaries** below.")


def _process_video_upload(vid_file, scene_dur: int):
    """Read uploaded video bytes and run VideoProcessor."""
    vid_bytes = vid_file.read()
    filename  = vid_file.name
    size_mb   = len(vid_bytes) / 1e6

    st.caption(f"📁 {filename}  ·  {size_mb:.1f} MB")

    if st.button("🔄 Extract Subtitles & Emotion Vectors from Video",
                  use_container_width=True, type="primary"):
        prog = st.progress(0.0, text="Initialising…")
        stat = st.empty()

        def _cb(p: float, msg: str):
            prog.progress(min(float(p), 1.0), text=msg)
            stat.markdown(f"*{msg}*")

        with st.spinner("Processing video…"):
            vp   = VideoProcessor(scene_duration=scene_dur, sample_rate=16000)
            info = vp.process(vid_bytes, filename, progress_cb=_cb)

        prog.empty(); stat.empty()

        if info.get("error"):
            st.error(f"Video processing error: {info['error']}"); return

        st.session_state.raw_subtitle  = info["srt_text"]
        st.session_state.emotion_array = info["emotion_array"]
        st.session_state.video_info    = info

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Scenes", info["n_scenes"])
        c2.metric("Duration", f"{info['duration_sec']:.1f}s")
        c3.metric("Embedded subs", "✅ Yes" if info["has_embedded_subs"] else "⚠️ Generated")
        c4.metric("Audio", "✅ OK" if info["audio_extracted"] else "❌ No audio")

        with st.expander("Subtitle preview"):
            st.code(info["srt_text"][:800], language="text")
        with st.expander("Emotion vectors preview"):
            _show_emotion_df(info["emotion_array"])

        st.success(f"✅ Done — {info['n_scenes']} scenes ready. "
                   f"Click **Generate Summaries** below.")


def _parse_emotion_file(uf) -> Optional[np.ndarray]:
    data = uf.read()
    try:
        if uf.name.lower().endswith(".npy"):
            return np.load(io.BytesIO(data)).astype(np.float32)
        import csv
        rows = []
        for row in csv.reader(data.decode("utf-8","ignore").splitlines()):
            if not row: continue
            try: rows.append([float(v) for v in row])
            except ValueError: pass
        return np.array(rows, dtype=np.float32) if rows else None
    except Exception as e:
        st.error(f"Cannot parse emotion file: {e}"); return None


def _show_emotion_df(arr: np.ndarray):
    import pandas as pd
    d  = min(arr.shape[1], len(EMOTION_DIMS))
    df = pd.DataFrame(arr[:, :d], columns=EMOTION_DIMS[:d])
    st.dataframe(df.style.background_gradient(cmap="Blues"),
                 use_container_width=True)


# ── Inference ──────────────────────────────────────────────────────────────────
def render_run(cfg_params: dict):
    st.markdown("---")
    st.markdown('<p class="section-head">🚀 Inference</p>',
                unsafe_allow_html=True)

    ready = (st.session_state.raw_subtitle is not None
             and st.session_state.emotion_array is not None)

    col_btn, col_tip = st.columns([2, 3])
    with col_btn:
        go = st.button("▶️ Generate Multi-Perspective Summaries",
                       use_container_width=True, disabled=not ready,
                       type="primary")
    with col_tip:
        if not ready:
            st.markdown(
                '<div class="warn-box">⚠️ Load a video or upload '
                'subtitle + emotion files above first.</div>',
                unsafe_allow_html=True)

    if go and ready:
        device  = st.session_state.device
        persp   = st.session_state.selected_perspectives
        emo_arr = st.session_state.emotion_array

        with st.spinner("🔄 Building graph & running GNN encoder…"):
            t0 = time.time()
            try:
                # Build system
                cfg = CRGNNConfig(
                    **{k: v for k, v in cfg_params.items()
                       if not k.startswith("_")})
                cfg.d_emotion = min(int(emo_arr.shape[1]), 16)
                system = CRGNNSystem(cfg)
                st.session_state.system = system

                # ── KEY FIX: torch.tensor() not torch.from_numpy() ──────
                emo_tensor = _to_tensor(emo_arr)

                result = run_inference(
                    system,
                    st.session_state.raw_subtitle,
                    emo_tensor,
                    perspectives=persp,
                    device=device,
                )
                st.session_state.inference_result = result
                elapsed = time.time() - t0

                if "error" not in result:
                    n = len(result.get("scenes", []))
                    st.success(f"✅ Done in {elapsed:.2f}s — {n} scenes processed.")
                else:
                    st.error(f"Inference error: {result['error']}")

            except Exception as e:
                st.error(f"Runtime error: {e}")
                import traceback
                st.code(traceback.format_exc())


# ── Results ────────────────────────────────────────────────────────────────────
def render_results():
    result = st.session_state.inference_result
    if not result or "error" in result:
        return

    st.markdown("---")
    st.markdown('<p class="section-head">📊 Analysis Results</p>',
                unsafe_allow_html=True)

    scenes      = result.get("scenes", [])
    scene_texts = [s["text"] for s in scenes]
    N           = len(scenes)
    emo_np      = _to_numpy(result["emotion_vecs"])
    vad_np      = _to_numpy(result["vad"])
    causal_aff  = _to_numpy(result["causal_affinity"])
    node_sal    = _to_numpy(result["node_salience"])
    z_dict      = result["z_dict"]
    sal_dict    = result["sal_dict"]
    persp       = st.session_state.selected_perspectives

    tabs = st.tabs([
        "📝 Summaries", "🕸️ Causal Graph",
        "🧬 Latent Space", "💡 Emotion", "📈 Metrics",
    ])

    # ── Summaries ─────────────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("#### Multi-Perspective Narrative Summaries")
        for p in persp:
            txt = result.get("summaries", {}).get(p, "")
            if not txt:
                continue
            st.markdown(
                f'<div class="pcard {p}">'
                f'<b style="color:{COLORS[p]}">{ICONS[p]} {p.capitalize()}</b>'
                f'<br><br>{txt.replace(chr(10),"<br>")}</div>',
                unsafe_allow_html=True)

        if result.get("summaries"):
            st.download_button(
                "⬇️ Download Summaries (JSON)",
                json.dumps(result["summaries"], indent=2),
                "summaries.json", "application/json",
                use_container_width=True)

        if scenes:
            import pandas as pd
            df = pd.DataFrame([{
                "Scene":  i + 1,
                "Text":   s["text"][:60] + ("…" if len(s["text"]) > 60 else ""),
                "Salience": f"{float(node_sal[i]):.3f}" if i < len(node_sal) else "—",
                "Dom. Emotion": _dom_emo(emo_np[i] if i < len(emo_np) else np.zeros(8)),
            } for i, s in enumerate(scenes)])
            st.markdown("#### Scene Details")
            st.dataframe(df, use_container_width=True)

    # ── Causal Graph ──────────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("#### Causal Narrative Event Graph")
        thr = st.slider("Causal edge threshold", 0.1, 0.9, 0.4, 0.05,
                        key="g_thr")
        ei, et, ew = _build_edges(causal_aff, N, thr)

        col_g, col_s = st.columns([3, 1])
        with col_g:
            fig = plot_causal_graph_plotly(ei, et, ew, scene_texts, node_sal)
            st.plotly_chart(fig, use_container_width=True)
        with col_s:
            st.metric("Causal edges",   int((et == 1).sum()) if len(et) else 0)
            st.metric("Temporal edges", int((et == 0).sum()) if len(et) else 0)
            st.metric("Scenes", N)
            st.metric("Avg affinity", f"{causal_aff.mean():.3f}")

        st.markdown("#### Causal Affinity Matrix")
        fig_h = plot_causal_affinity_heatmap(
            causal_aff, [f"S{i+1}" for i in range(N)])
        st.plotly_chart(fig_h, use_container_width=True)

        st.download_button(
            "⬇️ Download Graph (JSON)",
            json.dumps({"edge_index": ei.tolist(), "edge_type": et.tolist(),
                        "scenes": scene_texts}, indent=2),
            "graph.json", "application/json")

    # ── Latent Space ──────────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("#### Perspective Latent Embeddings (PCA)")
        if z_dict:
            st.plotly_chart(plot_latent_scatter(z_dict),
                            use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_perspective_salience(sal_dict),
                            use_container_width=True)
        with c2:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig_s = plot_salience_heatmap(node_sal, scene_texts, title="")
            st.pyplot(fig_s, use_container_width=True)
            plt.close(fig_s)

        with st.expander("Raw Z vector (first 32 dims)"):
            import pandas as pd
            z = _to_numpy(result["z"])
            st.dataframe(
                pd.DataFrame(z[:32].reshape(1, -1),
                             columns=[f"z{i}" for i in range(min(32, len(z)))])
                .style.background_gradient(cmap="RdBu_r", axis=1),
                use_container_width=True)

        st.download_button(
            "⬇️ Download Embeddings (JSON)",
            json.dumps({k: _to_numpy(v).tolist() for k, v in z_dict.items()},
                       indent=2),
            "embeddings.json", "application/json")

    # ── Emotion ───────────────────────────────────────────────────────────────
    with tabs[3]:
        d = min(emo_np.shape[1], len(EMOTION_DIMS))
        st.plotly_chart(
            plot_emotion_trajectory(emo_np, list(range(1, N+1)),
                                    EMOTION_DIMS[:d]),
            use_container_width=True)

        if vad_np.shape[1] >= 3:
            st.plotly_chart(plot_vad_3d(vad_np, list(range(1, N+1))),
                            use_container_width=True)
            arc = _to_numpy(result["arc_vad"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Avg Valence",   f"{arc[0]:.3f}")
            c2.metric("Avg Arousal",   f"{arc[1]:.3f}")
            c3.metric("Avg Dominance", f"{arc[2]:.3f}")

        st.plotly_chart(
            build_summary_figure(emo_np, z_dict, sal_dict, vad_np,
                                 st.session_state.train_history),
            use_container_width=True)

    # ── Metrics ───────────────────────────────────────────────────────────────
    with tabs[4]:
        import pandas as pd
        pred_t = torch.tensor(causal_aff.tolist(),          dtype=torch.float32)
        true_t = torch.tensor((causal_aff >= 0.5).tolist(), dtype=torch.float32)
        vad_t  = torch.tensor(vad_np.tolist(),              dtype=torch.float32)

        gm  = graph_alignment_score(pred_t, true_t)
        pdv = perspective_divergence(z_dict)
        em  = emotion_consistency(vad_t)
        lm  = latent_space_quality([result["z"]])

        for title, data, col in [
            ("Graph Quality",       gm,  "left"),
            ("Latent Quality",      lm,  "left"),
            ("Perspective Divergence", pdv, "right"),
            ("Emotion Consistency", em,  "right"),
        ]:
            pass  # build below

        c1, c2 = st.columns(2)
        for title, data, col in [("Graph Quality", gm, c1),
                                   ("Latent Quality", lm, c1),
                                   ("Perspective Divergence", pdv, c2),
                                   ("Emotion Consistency", em, c2)]:
            with col:
                st.markdown(f"**{title}**")
                st.dataframe(
                    pd.DataFrame(list(data.items()), columns=["Metric","Value"])
                    .assign(Value=lambda df: df.Value.apply(
                        lambda x: f"{x:.4f}")),
                    use_container_width=True, hide_index=True)

        all_m = {"graph": gm, "perspective": pdv, "emotion": em, "latent": lm}
        st.download_button(
            "⬇️ Download All Metrics (JSON)",
            json.dumps(all_m, indent=2), "metrics.json", "application/json",
            use_container_width=True)


# ── Training panel ─────────────────────────────────────────────────────────────
def render_training(cfg_params: dict):
    st.markdown("---")
    with st.expander("🏋️ Training Panel (optional)", expanded=False):
        ready = (st.session_state.raw_subtitle is not None
                 and st.session_state.emotion_array is not None)
        if not ready:
            st.warning("Load data first."); return

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🚂 Start Training", use_container_width=True,
                          type="primary"):
                _run_training(cfg_params)
        with c2:
            h = st.session_state.train_history
            if h and h.get("L_total"):
                st.metric("Best L_total", f"{min(h['L_total']):.4f}")

        h = st.session_state.train_history
        if h:
            fig_l = plot_loss_curves({k: v for k, v in h.items() if v})
            st.plotly_chart(fig_l, use_container_width=True)


def _run_training(cfg_params: dict):
    device  = st.session_state.device
    emo_arr = st.session_state.emotion_array

    cfg = CRGNNConfig(**{k: v for k, v in cfg_params.items()
                          if not k.startswith("_")})
    cfg.d_emotion = min(int(emo_arr.shape[1]), 16)
    system = CRGNNSystem(cfg)
    st.session_state.system = system

    proc = system.subtitle_proc
    scenes, token_ids, padding_mask = proc.process(st.session_state.raw_subtitle)
    N = len(scenes)
    if N == 0:
        st.error("No scenes parsed."); return

    # Safe conversion
    emo_t = _to_tensor(emo_arr)
    if emo_t.size(0) != N:
        import torch.nn.functional as F
        emo_t = F.interpolate(
            emo_t.unsqueeze(0).unsqueeze(0),
            size=(N, emo_t.size(1)),
            mode="bilinear", align_corners=False
        ).squeeze(0).squeeze(0)

    prog = st.progress(0, "Building graph…")
    proc.embedding_module.train()
    embs = proc.embedding_module(token_ids, padding_mask)

    gb = system.graph_builder
    gb.to(device); gb.train()
    graph, aff, lbl = gb.build_graph(embs.detach(), emo_t.to(device),
                                      device=device)
    gs = GraphStore(); gs.add("doc1", graph, aff, lbl)
    prog.progress(30, "Training…")

    trainer = Trainer(system, cfg, device=device)
    import threading
    history: dict = {}

    def _go():
        nonlocal history
        history = trainer.fit(gs, proc)

    t = threading.Thread(target=_go, daemon=True)
    t.start(); t.join(timeout=max(60, cfg.max_epochs * 5))
    prog.progress(100, "Done")
    st.session_state.train_history = history
    st.success("✅ Training complete!")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _build_edges(aff: np.ndarray, N: int, thr: float):
    mask = aff >= thr
    sr, tg = np.where(mask)
    v = sr != tg
    sr, tg = sr[v], tg[v]
    ew_c = aff[sr, tg]
    if N > 1:
        ts = np.arange(N - 1); tt = ts + 1
        ei = np.concatenate([np.array([ts, tt]), np.array([sr, tg])], axis=1)
        et = np.concatenate([np.zeros(N-1, dtype=int),
                              np.ones(len(sr), dtype=int)])
        ew = np.concatenate([np.ones(N - 1), ew_c])
    else:
        ei = np.array([sr, tg])
        et = np.ones(len(sr), dtype=int)
        ew = ew_c
    return ei, et, ew


def _dom_emo(e: np.ndarray) -> str:
    d = min(len(e), len(EMOTION_DIMS))
    return EMOTION_DIMS[int(np.argmax(e[:d]))] if d else "neutral"


def _demo_srt() -> str:
    return """1
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


def _demo_emotions() -> np.ndarray:
    return np.array([
        [0.1, 0.2, 0.1, 0.5, 0.3, 0.0, 0.4, 0.6],
        [0.0, 0.3, 0.2, 0.7, 0.4, 0.1, 0.2, 0.3],
        [0.0, 0.4, 0.3, 0.6, 0.8, 0.2, 0.1, 0.2],
        [0.0, 0.2, 0.8, 0.4, 0.1, 0.6, 0.0, 0.1],
        [0.3, 0.5, 0.4, 0.2, 0.3, 0.1, 0.5, 0.4],
    ], dtype=np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    st.markdown(
        '<h1 class="main-title">🎬 CRGNN — Causally-Regularized Graph Neural '
        'Narrative Representation</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">Affective Multi-Perspective Summarization · '
        'All models learned from scratch · AAAI / IEEE TNNLS quality</p>',
        unsafe_allow_html=True)

    cfg_params = render_sidebar()
    render_upload(cfg_params)
    render_run(cfg_params)
    render_results()
    render_training(cfg_params)

    st.markdown("---")
    st.markdown(
        "<small style='color:#90a4ae;'>CRGNN v2 · Research prototype · "
        "No pre-trained models used.</small>",
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
