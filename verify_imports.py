"""
verify_imports.py
==================
Import verification script called by setup.bat step 9.
Checks every project module is importable and prints a summary.
OCP-additive: does not modify any existing file.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODULES = [
    # Third-party dependencies
    ("streamlit",           "import streamlit"),
    ("torch",               "import torch"),
    ("numpy",               "import numpy"),
    ("networkx",            "import networkx"),
    ("plotly",              "import plotly"),
    ("sklearn",             "import sklearn"),
    ("transformers",        "import transformers"),

    # integration/ -- OCP core layer
    ("integration",                       "from integration import ServiceRegistry, PipelineOrchestrator"),
    ("integration.interfaces",            "from integration.interfaces.base_interfaces import IVideoAnalyser, SceneRecord"),
    ("integration.adapters.video",        "from integration.adapters.video_adapter import VideoModuleAdapter"),
    ("integration.adapters.emotion",      "from integration.adapters.emotion_adapter import EmotionModuleAdapter"),
    ("integration.adapters.summary",      "from integration.adapters.summary_adapter import SummaryModuleAdapter"),
    ("integration.adapters.fusion",       "from integration.adapters.fusion_adapter import FusionEngineAdapter"),
    ("integration.adapters.evaluation",   "from integration.adapters.evaluation_adapter import EvaluationAdapter"),
    ("integration.registry",              "from integration.registry.service_registry import ServiceRegistry"),
    ("integration.pipeline",              "from integration.pipeline.orchestrator import PipelineOrchestrator"),
    ("integration.services",              "from integration.services.session_manager import SessionManager"),

    # utils/
    ("utils",                             "from utils import resolve_legacy_path, get_outputs_root, ensure_outputs_dirs"),

    # OCP extension layers
    ("fusion_plus",                       "from fusion_plus import adaptive_fuse_numpy"),
    ("calibration",                       "from calibration import EmotionCalibrator, diagnose_calibration"),
    ("research_layers.temporal_arc",      "from research_layers.temporal_arc import compute_emotion_arc, ArcResult"),
    ("research_layers.causal_graph",      "from research_layers.causal_graph import CausalNarrativeGraph, graph_to_edge_list"),
    ("research_layers.adaptive_fusion",   "from research_layers.adaptive_fusion import adaptive_fuse_numpy"),
    ("research_layers.perspective_formal","from research_layers.perspective_formal import perspective_conflict_score"),
    ("perspective_plus",                  "from perspective_plus import perspective_conflict_score, CANONICAL_PERSPECTIVES"),
    ("evaluation_plus",                   "from evaluation_plus import compute_enhanced_emotion_metrics, BaselineComparator"),
    ("metrics_plus",                      "from metrics_plus import temporal_consistency, cross_modal_agreement, narrative_coherence"),
    ("wrappers",                          "from wrappers import EnhancedPipeline, EnhancedResult"),

    # Legacy stubs -- must be importable without error
    ("fusion.scene_representation",       "import fusion.scene_representation"),
    ("fusion.merge_modalities",           "import fusion.merge_modalities"),
    ("fusion.final_generation",           "import fusion.final_generation"),
    ("evaluation.summary_metrics",        "import evaluation.summary_metrics"),
    ("evaluation.emotion_metrics",        "import evaluation.emotion_metrics"),
    ("evaluation.human_eval_form",        "import evaluation.human_eval_form"),
]

ok = fail = 0
for label, code in MODULES:
    try:
        exec(code, {})
        print(f"  [OK]  {label}")
        ok += 1
    except Exception as e:
        print(f"  [--]  {label}  ({type(e).__name__}: {e})")
        fail += 1

print()
print(f"  Result: {ok}/{ok+fail} modules verified", end="")
if fail == 0:
    print("  -- All modules ready.")
else:
    print(f"  -- {fail} unavailable (graceful fallbacks will activate).")
