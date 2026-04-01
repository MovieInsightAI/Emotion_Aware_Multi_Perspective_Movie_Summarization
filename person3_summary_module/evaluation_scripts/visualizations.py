"""
evaluation_scripts/visualizations.py
======================================
Rich visualizations for the CRGNN system outputs.

Produces:
  1. Causal narrative graph (NetworkX + Matplotlib / Plotly)
  2. Latent embedding scatter (PCA / UMAP-style via PCA)
  3. Perspective projection comparison (radar / bar chart)
  4. Emotion trajectory (line chart per dimension)
  5. Training loss curves
  6. Attention / salience heatmap
  7. VAD emotion arc 3D scatter

All visualizations use Matplotlib, Seaborn, NetworkX, and Plotly only.
No external rendering services are used.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import torch

logger = logging.getLogger(__name__)

# Colour palette consistent with research paper figures
PALETTE = {
    "protagonist": "#2196F3",
    "antagonist":  "#F44336",
    "narrator":    "#4CAF50",
    "causal":      "#FF9800",
    "temporal":    "#9C27B0",
    "neutral":     "#607D8B",
}


# ===========================================================================
# 1.  Causal Narrative Graph
# ===========================================================================
def plot_causal_graph_matplotlib(
    edge_index: np.ndarray,        # (2, E) int
    edge_type: np.ndarray,         # (E,)  0=temporal, 1=causal
    edge_weight: np.ndarray,       # (E,)  float
    scene_texts: List[str],
    node_salience: Optional[np.ndarray] = None,  # (N,)
    title: str = "Causal Narrative Graph",
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Render causal + temporal narrative graph using NetworkX / Matplotlib.

    Returns a Matplotlib Figure object (embeddable in Streamlit via
    st.pyplot).
    """
    N = len(scene_texts)
    G = nx.DiGraph()

    # Add nodes
    for i, text in enumerate(scene_texts):
        label = f"S{i+1}: {text[:25]}…" if len(text) > 25 else f"S{i+1}: {text}"
        sal = float(node_salience[i]) if node_salience is not None else 0.5
        G.add_node(i, label=label, salience=sal)

    # Add edges
    edge_colors, edge_widths, edge_styles = [], [], []
    for e in range(edge_index.shape[1]):
        src, tgt = int(edge_index[0, e]), int(edge_index[1, e])
        if src >= N or tgt >= N:
            continue
        etype = int(edge_type[e])
        ew = float(edge_weight[e])
        G.add_edge(src, tgt, etype=etype, weight=ew)
        edge_colors.append(PALETTE["causal"] if etype == 1
                           else PALETTE["temporal"])
        edge_widths.append(1.0 + ew * 3.0)
        edge_styles.append("solid" if etype == 1 else "dashed")

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#FAFAFA")
    fig.patch.set_facecolor("white")

    # Layout
    try:
        pos = nx.kamada_kawai_layout(G)
    except Exception:
        pos = nx.spring_layout(G, seed=42)

    # Node size by salience
    node_sizes = []
    node_colors = []
    for i in range(N):
        if G.has_node(i):
            sal = G.nodes[i].get("salience", 0.5)
            node_sizes.append(600 + 1400 * sal)
            node_colors.append(plt.cm.Blues(0.4 + 0.6 * sal))

    valid_nodes = [i for i in range(N) if G.has_node(i)]

    nx.draw_networkx_nodes(G, pos, nodelist=valid_nodes,
                           node_size=node_sizes, node_color=node_colors,
                           alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos,
                            labels={i: G.nodes[i].get("label", str(i))
                                    for i in valid_nodes},
                            font_size=7, font_color="#212121",
                            ax=ax)

    # Draw edges by type separately (for dashed styling)
    causal_edges = [(u, v) for u, v, d in G.edges(data=True) if d["etype"] == 1]
    temporal_edges = [(u, v) for u, v, d in G.edges(data=True) if d["etype"] == 0]

    if causal_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=causal_edges,
            edge_color=PALETTE["causal"], width=2.0,
            arrows=True, arrowsize=18, ax=ax,
            connectionstyle="arc3,rad=0.1",
        )
    if temporal_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=temporal_edges,
            edge_color=PALETTE["temporal"], width=1.5,
            style="dashed", arrows=True, arrowsize=14, ax=ax,
        )

    # Legend
    patches = [
        mpatches.Patch(color=PALETTE["causal"], label="Causal edge"),
        mpatches.Patch(color=PALETTE["temporal"], label="Temporal edge"),
    ]
    ax.legend(handles=patches, loc="upper left", fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.axis("off")
    plt.tight_layout()
    return fig


def plot_causal_graph_plotly(
    edge_index: np.ndarray,
    edge_type: np.ndarray,
    edge_weight: np.ndarray,
    scene_texts: List[str],
    node_salience: Optional[np.ndarray] = None,
    title: str = "Interactive Causal Narrative Graph",
) -> go.Figure:
    """
    Interactive Plotly version of the causal graph.
    Hover over nodes to see full scene text.
    """
    import networkx as nx

    N = len(scene_texts)
    G = nx.DiGraph()
    for i in range(N):
        sal = float(node_salience[i]) if node_salience is not None else 0.5
        G.add_node(i, salience=sal)

    for e in range(edge_index.shape[1]):
        src, tgt = int(edge_index[0, e]), int(edge_index[1, e])
        if src < N and tgt < N:
            G.add_edge(src, tgt, etype=int(edge_type[e]),
                       weight=float(edge_weight[e]))

    try:
        pos = nx.kamada_kawai_layout(G)
    except Exception:
        pos = nx.spring_layout(G, seed=42)

    # Build Plotly traces
    fig_traces = []

    # Edge traces
    for etype, color, dash in [(0, PALETTE["temporal"], "dash"),
                                (1, PALETTE["causal"], "solid")]:
        x_edges, y_edges = [], []
        for u, v, d in G.edges(data=True):
            if d["etype"] == etype:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                x_edges += [x0, x1, None]
                y_edges += [y0, y1, None]
        if x_edges:
            label = "Temporal" if etype == 0 else "Causal"
            fig_traces.append(go.Scatter(
                x=x_edges, y=y_edges,
                mode="lines",
                line=dict(color=color, width=2, dash=dash),
                name=f"{label} edge",
                hoverinfo="skip",
            ))

    # Node trace
    node_x = [pos[i][0] for i in range(N) if G.has_node(i)]
    node_y = [pos[i][1] for i in range(N) if G.has_node(i)]
    node_text = [f"Scene {i+1}: {scene_texts[i]}" for i in range(N) if G.has_node(i)]
    node_sal = [G.nodes[i].get("salience", 0.5) for i in range(N) if G.has_node(i)]

    fig_traces.append(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=[f"S{i+1}" for i in range(len(node_x))],
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(
            size=[12 + 18 * s for s in node_sal],
            color=node_sal,
            colorscale="Blues",
            cmin=0, cmax=1,
            showscale=True,
            colorbar=dict(title="Salience"),
            line=dict(color="white", width=1),
        ),
        name="Events",
    ))

    fig = go.Figure(
        data=fig_traces,
        layout=go.Layout(
            title=dict(text=title, font=dict(size=16)),
            showlegend=True,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="#FAFAFA",
            paper_bgcolor="white",
            height=600,
        )
    )
    return fig


# ===========================================================================
# 2.  Latent Embedding Scatter (PCA)
# ===========================================================================
def plot_latent_scatter(
    z_dict: Dict[str, torch.Tensor],    # {persp: (d_persp,)}
    title: str = "Perspective Latent Space (PCA)",
) -> go.Figure:
    """
    2D PCA scatter of perspective-specific latent embeddings.
    """
    names = list(z_dict.keys())
    vecs = np.array(torch.stack([z_dict[n].flatten() for n in names]).detach().cpu().tolist(), dtype=np.float32)

    # Manual 2-component PCA (no sklearn dependency)
    vecs_c = vecs - vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs_c, rowvar=False)
    if cov.ndim < 2:
        cov = np.array([[cov, 0], [0, 0]])
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    pcs = vecs_c @ eigvecs[:, :2]

    fig = go.Figure()
    for i, name in enumerate(names):
        color = PALETTE.get(name, "#607D8B")
        fig.add_trace(go.Scatter(
            x=[pcs[i, 0]], y=[pcs[i, 1]],
            mode="markers+text",
            text=[name.capitalize()],
            textposition="top center",
            marker=dict(size=18, color=color,
                        line=dict(color="white", width=2)),
            name=name.capitalize(),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="PC 1",
        yaxis_title="PC 2",
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="white",
        height=450,
    )
    return fig


# ===========================================================================
# 3.  Perspective Salience Comparison Bar Chart
# ===========================================================================
def plot_perspective_salience(
    sal_dict: Dict[str, float],
    title: str = "Perspective Salience Scores",
) -> go.Figure:
    names = list(sal_dict.keys())
    values = [sal_dict[n] for n in names]
    colors = [PALETTE.get(n, "#607D8B") for n in names]

    fig = go.Figure(go.Bar(
        x=[n.capitalize() for n in names],
        y=values,
        marker_color=colors,
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        yaxis=dict(range=[0, max(values) * 1.3 if values else 1],
                   title="Salience"),
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="white",
        height=400,
    )
    return fig


# ===========================================================================
# 4.  Emotion Trajectory
# ===========================================================================
def plot_emotion_trajectory(
    emotion_vecs: np.ndarray,           # (N, d_emo)
    scene_ids: Optional[List[int]] = None,
    emotion_labels: Optional[List[str]] = None,
    title: str = "Scene-wise Emotion Trajectory",
) -> go.Figure:
    """
    Line chart of per-scene emotion intensities.
    """
    N, d = emotion_vecs.shape
    labels = emotion_labels or [
        "Joy", "Sadness", "Anger", "Fear",
        "Surprise", "Disgust", "Trust", "Anticipation"
    ][:d]
    x = scene_ids or list(range(1, N + 1))

    colors = px.colors.qualitative.Set2[:d]

    fig = go.Figure()
    for i, lbl in enumerate(labels):
        fig.add_trace(go.Scatter(
            x=x, y=emotion_vecs[:, i].tolist(),
            mode="lines+markers",
            name=lbl,
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=6),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Scene",
        yaxis_title="Emotion Intensity",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="white",
        height=450,
    )
    return fig


# ===========================================================================
# 5.  VAD 3D Scatter
# ===========================================================================
def plot_vad_3d(
    vad: np.ndarray,                    # (N, 3)  Valence, Arousal, Dominance
    scene_ids: Optional[List[int]] = None,
    title: str = "VAD Emotion Arc (3D)",
) -> go.Figure:
    N = vad.shape[0]
    x = scene_ids or list(range(1, N + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=vad[:, 0].tolist(),
        y=vad[:, 1].tolist(),
        z=vad[:, 2].tolist(),
        mode="lines+markers",
        text=[f"Scene {i}" for i in x],
        marker=dict(
            size=6,
            color=list(range(N)),
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Scene"),
        ),
        line=dict(color="grey", width=2),
    ))
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Valence",
            yaxis_title="Arousal",
            zaxis_title="Dominance",
        ),
        height=500,
    )
    return fig


# ===========================================================================
# 6.  Training Loss Curves
# ===========================================================================
def plot_loss_curves(
    history: Dict[str, List[float]],
    title: str = "Training Loss Curves",
) -> go.Figure:
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for idx, (key, values) in enumerate(history.items()):
        if not values:
            continue
        fig.add_trace(go.Scatter(
            x=list(range(1, len(values) + 1)),
            y=values,
            mode="lines+markers",
            name=key,
            line=dict(color=colors[idx % len(colors)], width=2),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title="Loss",
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="white",
        height=500,
        legend=dict(orientation="v"),
    )
    return fig


# ===========================================================================
# 7.  Causal Affinity Heatmap
# ===========================================================================
def plot_causal_affinity_heatmap(
    affinity: np.ndarray,           # (N, N)
    scene_labels: Optional[List[str]] = None,
    title: str = "Causal Affinity Matrix",
) -> go.Figure:
    N = affinity.shape[0]
    labels = scene_labels or [f"S{i+1}" for i in range(N)]

    fig = go.Figure(go.Heatmap(
        z=affinity.tolist(),
        x=labels,
        y=labels,
        colorscale="Oranges",
        zmin=0.0, zmax=1.0,
        colorbar=dict(title="P(causal)"),
        hovertemplate="Cause: %{y}<br>Effect: %{x}<br>P=%{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Effect Event",
        yaxis_title="Cause Event",
        height=450,
        yaxis=dict(autorange="reversed"),
    )
    return fig


# ===========================================================================
# 8.  Salience Heatmap (per scene)
# ===========================================================================
def plot_salience_heatmap(
    salience: np.ndarray,           # (N,)
    scene_texts: List[str],
    title: str = "Scene Salience",
) -> plt.Figure:
    N = len(salience)
    labels = [f"S{i+1}: {t[:30]}…" if len(t) > 30 else f"S{i+1}: {t}"
              for i, t in enumerate(scene_texts[:N])]

    fig, ax = plt.subplots(figsize=(10, max(3, N * 0.5)))
    data = salience.reshape(-1, 1)
    sns.heatmap(data, annot=True, fmt=".2f", cmap="Blues",
                yticklabels=labels, xticklabels=["Salience"],
                vmin=0, vmax=1, ax=ax, cbar_kws={"shrink": 0.5})
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


# ===========================================================================
# 9.  Combined Dashboard Figure
# ===========================================================================
def build_summary_figure(
    emotion_vecs: np.ndarray,
    z_dict: Dict[str, torch.Tensor],
    sal_dict: Dict[str, float],
    vad: np.ndarray,
    history: Optional[Dict[str, List[float]]] = None,
) -> go.Figure:
    """
    Composite Plotly dashboard with 4 sub-panels:
      (a) Emotion trajectory
      (b) Perspective salience
      (c) VAD first 2D (Valence vs Arousal)
      (d) Loss curve (if history provided)
    """
    rows = 2
    specs = [[{"type": "scatter"}, {"type": "bar"}],
             [{"type": "scatter"}, {"type": "scatter"}]]
    titles = ["Emotion Trajectory", "Perspective Salience",
              "Valence–Arousal", "Training Loss" if history else ""]

    fig = make_subplots(rows=rows, cols=2,
                        subplot_titles=titles,
                        specs=specs,
                        horizontal_spacing=0.12,
                        vertical_spacing=0.15)

    # Panel (a) — emotion trajectory (first 4 dims for readability)
    N = emotion_vecs.shape[0]
    labels = ["Joy", "Sadness", "Anger", "Fear"]
    colors_4 = ["#2196F3", "#F44336", "#FF9800", "#9C27B0"]
    for i, (lbl, col) in enumerate(zip(labels, colors_4)):
        if i >= emotion_vecs.shape[1]:
            break
        fig.add_trace(go.Scatter(
            x=list(range(1, N + 1)), y=emotion_vecs[:, i].tolist(),
            mode="lines+markers", name=lbl,
            line=dict(color=col, width=2),
        ), row=1, col=1)

    # Panel (b) — perspective salience
    names = list(sal_dict.keys())
    vals = [sal_dict[n] for n in names]
    bar_colors = [PALETTE.get(n, "#607D8B") for n in names]
    fig.add_trace(go.Bar(
        x=[n.capitalize() for n in names], y=vals,
        marker_color=bar_colors,
        showlegend=False,
    ), row=1, col=2)

    # Panel (c) — valence vs arousal scatter
    if vad.shape[1] >= 2:
        fig.add_trace(go.Scatter(
            x=vad[:, 0].tolist(), y=vad[:, 1].tolist(),
            mode="lines+markers",
            marker=dict(size=8, color=list(range(N)),
                        colorscale="Viridis", showscale=False),
            showlegend=False,
        ), row=2, col=1)

    # Panel (d) — loss curve
    if history and history.get("L_total"):
        vals_loss = history["L_total"]
        fig.add_trace(go.Scatter(
            x=list(range(1, len(vals_loss) + 1)), y=vals_loss,
            mode="lines", name="L_total",
            line=dict(color="#212121", width=2),
            showlegend=False,
        ), row=2, col=2)

    fig.update_layout(
        height=700,
        title_text="CRGNN System — Analysis Dashboard",
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="white",
    )
    fig.update_xaxes(title_text="Scene", row=1, col=1)
    fig.update_yaxes(title_text="Intensity", row=1, col=1)
    fig.update_xaxes(title_text="Valence", row=2, col=1)
    fig.update_yaxes(title_text="Arousal", row=2, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=2)
    fig.update_yaxes(title_text="Loss", row=2, col=2)
    return fig


# ===========================================================================
# Smoke test
# ===========================================================================
if __name__ == "__main__":
    import os, json

    os.makedirs("sample_outputs", exist_ok=True)

    N, d_emo = 5, 8
    emotion_vecs = np.random.rand(N, d_emo)
    vad = np.random.rand(N, 3) * 2 - 1

    scene_texts = [
        "The detective walks into a dimly lit room",
        "A shadowy figure stands near the window",
        "She pulls out her badge and demands answers",
        "The figure smiles and steps into the light",
        "It was her partner all along — a betrayal",
    ]
    salience = np.random.rand(N)

    edge_index = np.array([[0, 1, 2, 3, 1, 3], [1, 2, 3, 4, 3, 4]])
    edge_type = np.array([0, 0, 0, 0, 1, 1])
    edge_weight = np.random.rand(6)

    # Matplotlib causal graph
    fig_mat = plot_causal_graph_matplotlib(
        edge_index, edge_type, edge_weight, scene_texts, salience)
    fig_mat.savefig("sample_outputs/causal_graph.png", dpi=150,
                    bbox_inches="tight")
    print("Saved: sample_outputs/causal_graph.png")

    # Plotly causal graph
    fig_plotly = plot_causal_graph_plotly(
        edge_index, edge_type, edge_weight, scene_texts, salience)
    fig_plotly.write_html("sample_outputs/causal_graph.html")
    print("Saved: sample_outputs/causal_graph.html")

    # Latent scatter
    z_dict = {k: torch.randn(128) for k in ["protagonist", "antagonist", "narrator"]}
    fig_scatter = plot_latent_scatter(z_dict)
    fig_scatter.write_html("sample_outputs/latent_scatter.html")
    print("Saved: sample_outputs/latent_scatter.html")

    # Emotion trajectory
    fig_emo = plot_emotion_trajectory(emotion_vecs, emotion_labels=
        ["Joy","Sad","Anger","Fear","Surp","Disg","Trust","Anticip"])
    fig_emo.write_html("sample_outputs/emotion_trajectory.html")
    print("Saved: sample_outputs/emotion_trajectory.html")

    # Salience heatmap
    fig_sal = plot_salience_heatmap(salience, scene_texts)
    fig_sal.savefig("sample_outputs/salience_heatmap.png", dpi=150,
                    bbox_inches="tight")
    plt.close(fig_sal)
    print("Saved: sample_outputs/salience_heatmap.png")

    print("visualizations.py ✓")
