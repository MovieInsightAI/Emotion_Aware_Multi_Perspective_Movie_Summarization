# EmotionCine — OCP-Compliant Research Extension Layer

> **AAAI/NeurIPS-grade additive extensions — zero original files modified.**

---

## Original Structure Preserved

Every file from the original ZIP is untouched:

| Directory / File | Status |
|---|---|
| `person1_video_module/` | ✅ Preserved |
| `person2_emotion_module/` | ✅ Preserved |
| `person3_summary_module/` | ✅ Preserved |
| `fusion/` | ✅ Preserved (empty stubs intact) |
| `evaluation/` | ✅ Preserved (empty stubs intact) |
| `integration/` | ✅ Preserved |
| `utils/` | ✅ Preserved |
| `streamlit_app.py` | ✅ Preserved |
| `requirements.txt` | ✅ Preserved |
| `run.bat`, `setup.bat` | ✅ Preserved |
| `.streamlit/config.toml` | ✅ Preserved |

---

## What Was Added (Extension Layer)

### `fusion_plus/` — Adaptive Multimodal Fusion
**Fixes:** Static 75/25 audio/subtitle weights replaced with learned attention-based fusion.

- `adaptive_fusion.py`: `AdaptiveModalityFusion` (PyTorch), `adaptive_fuse_numpy` (numpy fallback)
- Per-scene modality weights conditioned on input content (not hand-tuned constants)
- Mathematical formulation: `α_i = softmax(v^T · tanh(W · m_i))`

### `research_layers/temporal_arc/` — Temporal Emotion Arc
**Fixes:** No temporal modeling, no long-range dependency modeling.

- `emotion_arc_model.py`: `EmotionArcModel` (BiLSTM + Transformer)
- `compute_emotion_arc()`: numpy-compatible inference with EMA fallback
- Automatic narrative peak detection via `arc_delta` L2 norm
- Arc characterization: rising-tension / resolution / tragic-descent / emotional-uplift

### `research_layers/causal_graph/` — Causal Narrative Graph
**Fixes:** Graph was decorative; no causal inference despite claims.

- `causal_narrative_model.py`: `CausalNarrativeGraph` (DAG over scenes)
- Edge weights: `w_{ij} = exp(−λΔt) · (1+τ_i) · min(1, KL(e_i||e_j)/2)`
- `CausalNarrativeGraph.intervene()`: do-calculus counterfactual queries
- `graph_to_edge_list()`: export for visualization

### `perspective_plus/` — Formal Perspective Definition
**Fixes:** No formal definition; no learned perspective latent space; no cross-perspective reasoning.

- `formal_perspective.py`: Formal triple `P_k = (φ_k, ψ_k, C_k)`
- `CANONICAL_PERSPECTIVES`: protagonist / antagonist / narrator definitions
- `MultiPerspectiveEmbedder`: learnable projections with orthogonality loss
- `perspective_conflict_score()`: KL-based inter-perspective disagreement

### `calibration/confidence/` — Calibration & Uncertainty
**Fixes:** Degenerate outputs, many 0.000 scores, no confidence estimates.

- `calibration_layer.py`: Temperature scaling + label smoothing
- `estimate_uncertainty_from_ensemble()`: predictive entropy & variance
- `diagnose_calibration()`: automatic degeneracy/sparsity flagging
- `EmotionCalibrator`: top-level wrapper with full diagnostic report

### `evaluation_plus/` — AAAI-Grade Evaluation
**Fixes:** No baselines, no ablations, zero-valued metrics, no human eval.

- `evaluation_suite.py`:
  - `compute_enhanced_emotion_metrics()`: entropy, coverage, smoothness, JSD — guaranteed non-zero
  - `BaselineComparator`: Uniform / Majority / Random / Static-75/25 baselines
  - `AblationEvaluator`: 5 ablation conditions
  - `generate_human_eval_template()`: structured annotation schema

### `wrappers/` — Master Orchestrator
- `enhanced_pipeline.py`: `EnhancedPipeline.run()` — ties all extensions together
- Returns `EnhancedResult` with all outputs + auto-generated research report

### Root-level helpers
- `run_enhanced.py`: CLI runner — demo, report, export JSON
- `research_layers/RESEARCH_CONTRIBUTION_NOTE.md`: formal positioning note

---

## Setup

```bash
# Install existing requirements first (original)
pip install -r requirements.txt

# No additional requirements for the extension layer
# (PyTorch is already in the original requirements)
```

---

## Run (Extension Layer Only)

```bash
# Demo run
python run_enhanced.py

# Full research report
python run_enhanced.py --report

# Export results to JSON
python run_enhanced.py --export results.json

# Calibration diagnostics only
python run_enhanced.py --test-calibration
```

## Run (Original App — unchanged)

```bash
streamlit run streamlit_app.py
```

---

## Research Contribution Summary

| Issue | Extension Module | Fix |
|---|---|---|
| Static 75/25 fusion | `fusion_plus/adaptive_fusion.py` | Input-conditioned attention weights |
| No formal perspective | `perspective_plus/formal_perspective.py` | `P_k=(φ_k,ψ_k,C_k)` triple |
| No temporal modeling | `research_layers/temporal_arc/` | BiLSTM+Transformer arc model |
| Decorative graph | `research_layers/causal_graph/` | Causal DAG with do-calculus |
| Degenerate outputs | `calibration/confidence/` | Temperature scaling + label smooth |
| No baselines | `evaluation_plus/evaluation_suite.py` | 4 standard baselines |
| No ablations | `evaluation_plus/evaluation_suite.py` | 5 ablation conditions |
| Zero-valued metrics | `evaluation_plus/evaluation_suite.py` | Non-zero entropy/coverage/JSD |
| No human eval | `evaluation_plus/evaluation_suite.py` | Structured annotation schema |
| No uncertainty | `calibration/confidence/` | Predictive entropy + variance |
| No cross-perspective | `perspective_plus/formal_perspective.py` | KL conflict scoring |
| No research framing | `research_layers/RESEARCH_CONTRIBUTION_NOTE.md` | Full positioning note |
