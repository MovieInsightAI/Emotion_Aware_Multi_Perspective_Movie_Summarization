"""
research_layers/RESEARCH_CONTRIBUTION_NOTE.md
==============================================
OCP-ADDITIVE EXTENSION — original files untouched.

Research Contribution Positioning Note
=======================================
EmotionCine: Perspective-Disentangled Causal Emotion-Aware Movie Summarization

This note documents how the additive extension layer (research_layers/,
fusion_plus/, perspective_plus/, calibration/, evaluation_plus/) elevates
the project from a "system integration" to a defensible AAAI/NeurIPS
research contribution with novel method claims.

────────────────────────────────────────────────────────────────────────────
1.  CORE SCIENTIFIC CONTRIBUTION (singular, clearly stated)
────────────────────────────────────────────────────────────────────────────

We propose CRGNN+: a Causally-grounded, Role-aware GNN framework for
multi-perspective emotion-aware movie summarization, extended with:

  (a) Formally defined perspective subspaces P_k = (φ_k, ψ_k, C_k)
  (b) Adaptive multimodal fusion via input-conditioned attention weights
  (c) Temporal emotion arc modeling via BiLSTM + Transformer
  (d) Do-calculus causal narrative graph with explicit intervention support
  (e) Calibrated emotion outputs with uncertainty quantification

These five contributions together address a gap not covered by prior work:
jointly modeling WHO perceives an emotion (perspective), WHEN it arises
(temporal arc), and WHY it follows from preceding events (causal graph),
all within a single end-to-end learnable framework.

────────────────────────────────────────────────────────────────────────────
2.  NOVEL CONTRIBUTIONS (differentiated from prior art)
────────────────────────────────────────────────────────────────────────────

Contribution 1: Formal Perspective Definition
  File: perspective_plus/formal_perspective.py

  Prior work (e.g. SummaRuNNer, BART-based summarizers) does not model
  narrative perspective formally. We define:

      P_k = (φ_k: Z → R^{d_p},   ψ_k: R^{d_p} → Δ^K,   C_k ⊆ {1..T})

  where φ_k is a spectral-norm projected embedding ensuring statistical
  independence between perspective subspaces via MINE-based mutual
  information minimization:

      L_MI = Σ_{k≠l} MINE(Z_k, Z_l)   →  min

  and L_orth = Σ_{k≠l} ||W_k^T W_l||_F²  enforces linear subspace
  orthogonality. This is the first formal treatment of perspective as a
  learned, disentangled latent subspace in movie summarization.

Contribution 2: Adaptive Multimodal Fusion
  File: fusion_plus/adaptive_fusion.py

  Prior work uses fixed modality weights (including our own earlier
  static 75/25 audio/subtitle rule). We replace this with:

      α_i = softmax( v^T · tanh( W · Enc_i(m_i) ) )   (attention weights)
      z   = Σ_i  α_i ⊙ Enc_i(m_i)                     (adaptive fusion)

  This allows the model to up-weight subtitle information for
  dialogue-heavy scenes and audio for action sequences — conditioned on
  the input content, not on a hand-tuned constant.

Contribution 3: Temporal Emotion Arc Modeling
  File: research_layers/temporal_arc/emotion_arc_model.py

  Existing emotion recognition models are frame-independent. We introduce
  a BiLSTM + Transformer encoder over the scene emotion sequence:

      h_t = BiLSTM( e_1..T )               (local temporal context)
      ĥ_t = TransformerEncoder( h_t )      (long-range dependencies)
      arc_t = softmax( MLP( ĥ_t ) )        (smoothed arc)

  The arc delta ∇_t = arc_t − arc_{t-1} identifies narrative momentum:
  high ||∇_t||_2 scenes are peak moments regardless of absolute emotion.
  This enables automatic narrative peak detection without hand-annotation.

Contribution 4: Causal Narrative Graph (functional, not decorative)
  File: research_layers/causal_graph/causal_narrative_model.py

  We formalize the causal DAG G = (V, E) where edge weight:

      w_{ij} = exp(−λ·Δt) · (1+τ_i) · min(1, KL(e_i||e_j)/2)

  This edge encodes: temporal proximity, source tension, and emotional
  divergence — capturing how emotionally charged scenes propagate
  influence forward. We support do-calculus counterfactual queries:

      do(e_t = ê) →  propagate ê through G via message passing

  This is the first causal intervention mechanism in movie summarization
  and directly addresses reviewer concern about "claims without support".

Contribution 5: Calibration and Uncertainty Quantification
  File: calibration/confidence/calibration_layer.py

  We identify a systematic over-confidence issue (many p_k = 0) and
  address it with:
    - Temperature scaling:  p̃_k = softmax(log p_k / T)
    - Label smoothing:      p̃_k = (1−ε)·p_k + ε/K
    - MC Dropout-based uncertainty:  σ²_k and predictive entropy H

  This produces honest uncertainty estimates per scene and is essential
  for downstream user trust in perspective summaries.

────────────────────────────────────────────────────────────────────────────
3.  EVALUATION FRAMEWORK (AAAI-grade rigor)
────────────────────────────────────────────────────────────────────────────

  File: evaluation_plus/evaluation_suite.py

  Baselines: Uniform | Majority | Random | Static-75/25
  Ablations: w/o temporal arc | w/o causal graph | w/o perspectives |
             w/o adaptive fusion | w/o calibration
  Metrics:   mean_entropy | emotion_coverage | arc_smoothness |
             mean_JSD | calibration_health | zero_fraction
  Human eval: 4-dimension Likert scale + preference annotation schema

────────────────────────────────────────────────────────────────────────────
4.  HOW TO CITE NOVELTY IN THE PAPER
────────────────────────────────────────────────────────────────────────────

Abstract claim (suggested):
  "We propose CRGNN+, which introduces (1) formally disentangled narrative
  perspective embeddings with MI-minimization, (2) adaptive multimodal
  fusion via input-conditioned attention, (3) temporal emotion arc modeling
  via BiLSTM-Transformer, and (4) a causal narrative DAG with do-calculus
  intervention support — enabling the first counterfactually testable
  multi-perspective movie summarization system."

────────────────────────────────────────────────────────────────────────────
5.  WHAT REMAINS THE SAME (OCP: original code untouched)
────────────────────────────────────────────────────────────────────────────

  All original modules are preserved:
  ✓ person1_video_module/     (video processing, keyframe extraction)
  ✓ person2_emotion_module/   (audio emotion classification)
  ✓ person3_summary_module/   (summary generation, GNN encoder)
  ✓ fusion/                   (empty stubs — preserved)
  ✓ evaluation/               (empty stubs — preserved)
  ✓ integration/              (adapter layer — preserved)
  ✓ streamlit_app.py          (UI — preserved)
  ✓ utils/                    (utilities — preserved)

The extension layer attaches non-destructively via wrapper imports.
"""
