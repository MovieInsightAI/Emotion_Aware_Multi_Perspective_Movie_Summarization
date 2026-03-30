# CRGNN — Causally-Regularized Graph Neural Narrative Representation
## for Affective Multi-Perspective Summarization

> **Research prototype** — AAAI / IEEE TNNLS quality  
> All embeddings, encoders, and projections are **learned from scratch**.  
> No pre-trained language models or external embeddings are used.

---

## Architecture Overview

```
Subtitles (raw text)
    │
    ▼
subtitle_preprocessing.py
  └─ BPE Tokenizer (trained from scratch)
  └─ SceneEmbeddingModule (Transformer encoder, learned)
    │
    ▼
event_graph_builder.py
  └─ EventExtractionLayer  (scene → event nodes)
  └─ CausalEdgePredictor   (bilinear causal affinity)
  └─ TemporalEdgeBuilder   (sequential + emotion-gated)
    │
    ▼
gnn_narrative_encoder.py
  └─ TypedEdgeGATConv  ×  num_layers   (GATv2 + edge-type bias)
  └─ EmotionGatedResidual               (FiLM residual gating)
  └─ HierarchicalReadout                (mean+max+sum pool → Z)
    │
    ▼                    ◄─── emotion_conditioning.py
Latent Space Z ∈ ℝ^{d_latent}    (FiLM node modulation, edge re-weighting)
    │
    ▼
perspective_projection.py
  └─ PerspectiveProjector × 3   (spectral-norm linear, protagonist/antagonist/narrator)
  └─ Orthogonality + Contrastive + Emotion-prior losses
    │
    ▼
summary_decoder.py
  └─ NarrativeDecoder (GRU, learned vocabulary)
  └─ Surface realiser (template-based for UI)
    │
    ▼
Multi-Perspective Summaries + Causal Graph + Latent Embeddings
```

---

## Loss Function

```
L_total = L_repr
        + λ1 · L_causal          (causal graph edge prediction)
        + λ2 · L_disentangle     (orthogonality + contrastive + emotion-prior)
        + λ3 · L_temporal        (next-event prediction + masked reconstruction)
        + λ4 · L_summary         (decoder cross-entropy)
        + λ5 · L_kl              (VAE KL divergence, optional)
        + λ6 · L_align           (emotion alignment auxiliary)
```

---

## File Structure

```
causally_regularized_gnn/
├── app.py                        ← Streamlit UI
├── subtitle_preprocessing.py     ← BPE tokenizer + scene embedding
├── event_graph_builder.py        ← Event extraction + graph construction
├── gnn_narrative_encoder.py      ← GAT encoder + latent space
├── emotion_conditioning.py       ← FiLM conditioning + emotion arc
├── perspective_projection.py     ← Disentangled perspective projections
├── summary_decoder.py            ← GRU decoder + surface realiser
├── training_pipeline.py          ← Multi-loss training + inference
├── generate_sample_outputs.py    ← Generates all sample outputs
├── requirements.txt
├── setup.bat                     ← Windows: create venv + install deps
├── run.bat                       ← Windows: launch Streamlit UI
└── evaluation_scripts/
    ├── __init__.py
    ├── metrics.py                ← ROUGE, BLEU, graph alignment, etc.
    ├── visualizations.py         ← All Plotly + Matplotlib figures
    └── run_evaluation.py         ← CLI evaluation runner
```

---

## Quick Start (Windows)

```bat
REM 1. Setup (run once)
setup.bat

REM 2. Launch UI
run.bat
```

Then open http://localhost:8501 in your browser.

---

## Quick Start (Linux / macOS)

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install torch==2.2.0              # CPU version
pip install torch-geometric

# Launch UI
streamlit run app.py

# Or generate sample outputs directly
python generate_sample_outputs.py
```

---

## Streamlit UI

| Tab | Content |
|-----|---------|
| **Summaries** | Multi-perspective narrative summaries with scene table |
| **Causal Graph** | Interactive Plotly graph + causal affinity heatmap |
| **Latent Space** | PCA scatter, salience heatmap, raw Z vector |
| **Emotion** | Trajectory chart, VAD 3D arc, dashboard |
| **Metrics** | Graph alignment, perspective divergence, emotion consistency |

**Sidebar controls:**
- Device selection (CPU / CUDA)
- Perspective checkboxes (protagonist / antagonist / narrator)
- Model architecture sliders (dims, layers, heads)
- Loss weight sliders (λ1–λ5)
- Training panel with loss curve

---

## Evaluation CLI

```bash
python evaluation_scripts/run_evaluation.py \
    --subtitle_file data/movie.srt \
    --emotion_file  data/emotions.csv \
    --checkpoint    checkpoints/best.pt \
    --output_dir    sample_outputs/
```

Outputs: `evaluation_report.json`, `causal_graph.html/png`,
`latent_scatter.html`, `emotion_trajectory.html`, `dashboard.html`.

---

## Metrics

| Metric | Description |
|--------|-------------|
| **ROUGE-1/2/L** | N-gram overlap between generated and reference summaries |
| **BLEU-1/2/3/4** | Precision-based text similarity |
| **Graph Jaccard / F1** | Causal edge prediction quality |
| **SCA** | Structural Causal Alignment (TP−FP)/(TP+FN+FP) |
| **Perspective Divergence** | Pairwise L2 distance in perspective subspaces |
| **Emotion Consistency** | VAD arc cosine similarity |
| **Latent Isotropy** | min/max singular value ratio of Z embedding matrix |

---

## Research Notes

- **BPE tokenizer**: Sennrich et al., ACL 2016  
- **GATv2**: Brody et al., ICLR 2022  
- **FiLM conditioning**: Perez et al., AAAI 2018  
- **Orthogonality disentanglement**: Wang et al., EMNLP 2021  
- **Causal graph regularisation**: Tang et al., WWW 2023  
- **VAE latent space**: Kingma & Welling, ICLR 2014  

---

## Citation

```bibtex
@article{crgnn2024,
  title   = {Causally-Regularized Graph Neural Narrative Representation
             for Affective Multi-Perspective Summarization},
  author  = {[Author(s)]},
  journal = {Proceedings of AAAI / IEEE TNNLS},
  year    = {2025}
}
```
