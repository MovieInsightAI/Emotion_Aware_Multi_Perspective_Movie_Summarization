# 🎬 EmotionCine — Emotion-Aware Multi-Perspective Movie Summarization

> **AAAI Research Demo · Faculty Viva Ready · OCP-Compliant Integration Layer**

A unified, polished research dashboard that orchestrates a four-module deep learning pipeline for emotion-aware, multi-perspective movie summarization — built entirely through **extension, not modification** of the original codebase.

---

## ⚠️ OCP Compliance Statement

> **Original folder structure: PRESERVED EXACTLY**
> **Original file contents: UNTOUCHED — not a single line changed**
> **All improvements: added through extension layers only**

This integration layer was designed and implemented under strict **Open-Closed Principle (OCP)** discipline:

| Artefact | Status |
|---|---|
| `person1_video_module/` | ✅ Closed — zero modifications |
| `person2_emotion_module/` | ✅ Closed — zero modifications |
| `person3_summary_module/` | ✅ Closed — zero modifications |
| `fusion/` | ✅ Closed — empty stubs preserved |
| `evaluation/` | ✅ Closed — empty stubs preserved |
| `README.md` (original) | ✅ Closed — preserved |
| **Integration layer** | ✅ Open for extension via new files only |

---

## 📐 Architecture

```
Emotion_Aware_Multi_Perspective_Movie_Summarization-main/
│
├── person1_video_module/          ← LEGACY: untouched
│   ├── src/scene_detector.py
│   ├── src/keyframe_extractor.py
│   ├── src/feature_extractor.py
│   └── main.py
│
├── person2_emotion_module/        ← LEGACY: untouched
│   ├── emotion_classifier.py
│   ├── subtitle_emotion_hint.py
│   ├── preprocess_audio.py
│   ├── audio_extract.py
│   └── run_pipeline.py
│
├── person3_summary_module/        ← LEGACY: untouched
│   ├── training_pipeline.py       (CRGNN)
│   ├── gnn_narrative_encoder.py
│   ├── emotion_conditioning.py
│   ├── perspective_projection.py
│   ├── summary_decoder.py
│   ├── evaluation_scripts/
│   └── sample_outputs/
│
├── fusion/                        ← LEGACY: empty stubs, untouched
├── evaluation/                    ← LEGACY: empty stubs, untouched
│
│  ─ ─ ─ ─ ─ ─  INTEGRATION LAYER (all new)  ─ ─ ─ ─ ─ ─
│
├── streamlit_app.py               ← NEW: Streamlit dashboard
├── requirements.txt               ← NEW: unified dependencies
├── setup.bat                      ← NEW: Windows setup
├── run.bat                        ← NEW: Windows launcher
│
├── .streamlit/
│   └── config.toml                ← NEW: light theme config
│
├── integration/
│   ├── __init__.py
│   ├── interfaces/
│   │   ├── __init__.py
│   │   └── base_interfaces.py     ← ABCs: IVideoAnalyser, IEmotionAnalyser,
│   │                                        ISummaryGenerator, IFusionEngine, IEvaluator
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── video_adapter.py       ← Wraps person1_video_module (no edits)
│   │   ├── emotion_adapter.py     ← Wraps person2_emotion_module (no edits)
│   │   ├── summary_adapter.py     ← Wraps person3_summary_module (no edits)
│   │   ├── fusion_adapter.py      ← Implements fusion (legacy stubs preserved)
│   │   └── evaluation_adapter.py  ← Wraps evaluation_scripts/ (no edits)
│   ├── registry/
│   │   ├── __init__.py
│   │   └── service_registry.py    ← OCP-compliant DI registry
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── orchestrator.py        ← 5-stage pipeline coordinator
│   └── services/
│       ├── __init__.py
│       └── session_manager.py     ← Session caching + persistence
│
├── utils/
│   ├── __init__.py
│   ├── path_resolver.py           ← Cross-platform path utilities
│   └── output_manager.py          ← File I/O, JSON/CSV export
│
└── outputs/                       ← Generated at runtime
    ├── sessions/
    ├── summaries/
    ├── evaluation/
    ├── keyframes/
    └── uploads/
```

---

## 🔬 How the Integration Layer Works

### The Adapter Pattern

Each legacy module is wrapped by an **Adapter** that:
1. Temporarily injects the legacy module's directory into `sys.path`
2. Imports the legacy functions/classes without modifying them
3. Translates legacy data into canonical **data-transfer objects** (DTOs)
4. Falls back to pre-existing sample data if live dependencies are unavailable

```
Legacy Module          Adapter                  Interface
─────────────       ──────────────           ─────────────────
person1_video_module → VideoModuleAdapter  → IVideoAnalyser
person2_emotion_module → EmotionModuleAdapter → IEmotionAnalyser
person3_summary_module → SummaryModuleAdapter → ISummaryGenerator
fusion/ (empty)        → FusionEngineAdapter → IFusionEngine
evaluation_scripts/    → EvaluationAdapter  → IEvaluator
```

### The Registry Pattern

`ServiceRegistry.build_default()` is the **single place** where concrete adapter classes are instantiated. All consumers (orchestrator, UI) only see abstract interfaces — never concrete classes. This means any adapter can be swapped without changing any other file.

### The Orchestrator

`PipelineOrchestrator.run()` coordinates all five stages in sequence, passing canonical DTOs between stages. It accepts a `progress_callback` that drives the Streamlit progress bar in real time.

### Graceful Fallback

Because not all dependencies may be installed in every environment, every adapter checks `is_available()` and falls back to pre-existing sample data bundled with the original project:

| Adapter | Fallback |
|---|---|
| VideoModuleAdapter | `person1_video_module/data/outputs/scene_metadata.json` + bundled keyframes |
| EmotionModuleAdapter | Synthetic cinematic emotion arc derived from visual embeddings |
| SummaryModuleAdapter | `person3_summary_module/sample_outputs/perspective_summaries.json` |
| FusionEngineAdapter | Pure Python fusion (always available) |
| EvaluationAdapter | `person3_summary_module/sample_outputs/evaluation_metrics.json` |

---

## 🚀 Quick Start (Windows)

### Step 1 — Setup (run once)
```bat
cd Emotion_Aware_Multi_Perspective_Movie_Summarization-main
setup.bat
```

### Step 2 — Launch
```bat
run.bat
```

Then open **http://localhost:8501** in your browser.

---

## 🐧 Quick Start (Linux / macOS)

```bash
cd Emotion_Aware_Multi_Perspective_Movie_Summarization-main

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Optional: install torch-geometric for live CRGNN inference
pip install torch-geometric

# Launch
streamlit run streamlit_app.py
```

---

## 🎛️ App Workflow

1. **Upload** a `.mp4`/`.mkv` video and optionally an `.srt` subtitle file
   - Or use the bundled `sample_movie.mp4` automatically
2. **Configure** pipeline options in the sidebar (perspectives, evaluation toggle)
3. **Click "Run Pipeline"** — a real-time progress bar shows all 5 stages
4. **Explore results** across 7 tabs:

| Tab | Contents |
|---|---|
| 🔭 Overview | Pipeline summary cards, execution log |
| 🎬 Scene Analysis | Keyframe gallery, scene metadata table |
| 🎭 Emotion Analysis | Emotion arc strip, distribution chart, per-scene table |
| 📖 Perspective Summaries | Protagonist / Antagonist / Narrator blocks, causal graph |
| 🔀 Fusion | Final enriched summary, emotion pie chart, CRGNN dashboard |
| 📊 Evaluation | ROUGE, BLEU, Graph Jaccard, Perspective Divergence, VAD 3D |
| 💾 Outputs | Download links, saved file paths |

---

## 📦 Outputs Generated

All outputs are written to `outputs/` and organized by session ID:

```
outputs/
├── sessions/
│   ├── scenes_<session_id>.json       ← Scene metadata
│   └── emotions_<session_id>.csv      ← Per-scene emotion scores
├── summaries/
│   └── summary_<session_id>.json      ← Perspective summaries + fusion
├── evaluation/
│   └── eval_<session_id>.json         ← Evaluation metrics report
├── keyframes/                         ← Extracted keyframe images
└── uploads/                           ← Uploaded video/subtitle files
```

---

## 🏗️ Extensibility Model

To add a new summarization backend (e.g., an LLM-based approach):

1. Create `integration/adapters/llm_summary_adapter.py`
2. Implement `ISummaryGenerator`
3. Register it in `ServiceRegistry.build_default()` — **one line change**
4. No other files need to be modified

This is the OCP promise: **open for extension, closed for modification**.

---

## 📚 Research Context

This system implements the pipeline described in:

> **Causally-Regularized Graph Neural Narrative Representation for Affective Multi-Perspective Summarization**
> AAAI / IEEE TNNLS Research Prototype

Key technical components:
- **SceneDetect** (ContentDetector) for shot boundary detection
- **ResNet-50** (ImageNet pretrained) for 1000-dim visual feature extraction
- **Wav2Vec2** (`superb/wav2vec2-base-superb-er`) for audio emotion classification
- **BART-large-MNLI** for zero-shot subtitle emotion inference
- **CRGNN** (Causally-Regularized GNN) with GATv2 + FiLM conditioning for narrative encoding
- **GRU decoder** for surface-form summary generation
- **VAD (Valence-Arousal-Dominance)** emotion arc modeling

Loss function:
```
L_total = L_repr + λ1·L_causal + λ2·L_disentangle + λ3·L_temporal + λ4·L_summary + λ5·L_kl + λ6·L_align
```

---

## 🔑 Key Design Decisions

| Decision | Rationale |
|---|---|
| Adapter pattern over inheritance | Legacy classes can't be subclassed safely without modifying them |
| Registry over global imports | Enables testing and future backend swaps |
| Graceful fallback at every stage | Demo always runs, even without GPU or heavy dependencies |
| DTOs as interface currency | Decouples module outputs from module internals |
| `sys.path` injection per-call | Avoids polluting global path for the entire process |

---

*Original project: Emotion_Aware_Multi_Perspective_Movie_Summarization · Integration layer: OCP-compliant extension · © 2025*
